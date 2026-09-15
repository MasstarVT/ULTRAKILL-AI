using System.Collections.Generic;
using HarmonyLib;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace UltrakillAIBridge.Obs
{
    /// <summary>
    /// Builds a raw, structured snapshot of the game state. Python turns it into the policy's
    /// fixed-size vector and computes rewards, so tuning never requires rebuilding the mod.
    /// All game-specific reads live here.
    /// </summary>
    public sealed class ObservationBuilder
    {
        public int MaxEnemies = 16;
        public int HorizontalRays = 16;
        public int GroundRays = 8;
        public float RayLength = 50f;
        public float GroundRayRadius = 4f;
        public float GroundRayLength = 30f;

        // Resolved lazily so a game update renaming the private field only loses enemies_left.
        private static AccessTools.FieldRef<EndlessGrid, ActivateNextWave> anwRef;
        private static bool anwResolved;

        private static ActivateNextWave GetAnw(EndlessGrid grid)
        {
            if (!anwResolved)
            {
                anwResolved = true;
                try
                {
                    anwRef = AccessTools.FieldRefAccess<EndlessGrid, ActivateNextWave>("anw");
                }
                catch (System.Exception e)
                {
                    Plugin.Log.LogWarning($"EndlessGrid.anw not found, enemies_left unavailable: {e.Message}");
                }
            }
            return anwRef?.Invoke(grid);
        }

        private readonly List<(EnemyIdentifier eid, float dist)> sorted = new List<(EnemyIdentifier, float)>();

        public void Configure(JObject cfg)
        {
            if (cfg == null) return;
            MaxEnemies = cfg["max_enemies"]?.Value<int>() ?? MaxEnemies;
            HorizontalRays = cfg["horizontal_rays"]?.Value<int>() ?? HorizontalRays;
            GroundRays = cfg["ground_rays"]?.Value<int>() ?? GroundRays;
            RayLength = cfg["ray_length"]?.Value<float>() ?? RayLength;
            GroundRayRadius = cfg["ground_ray_radius"]?.Value<float>() ?? GroundRayRadius;
            GroundRayLength = cfg["ground_ray_length"]?.Value<float>() ?? GroundRayLength;
        }

        public static bool PlayerReady()
        {
            var nm = MonoSingleton<NewMovement>.Instance;
            return nm != null && nm.activated && !nm.dead && MonoSingleton<CameraController>.Instance != null
                   && string.IsNullOrEmpty(SceneHelper.PendingScene);
        }

        public JObject Build(int step, string eventName = null)
        {
            var obs = new JObject
            {
                ["type"] = "obs",
                ["step"] = step,
                ["frame"] = Time.frameCount,
                ["time"] = Time.time,
                ["scene"] = SceneHelper.CurrentScene,
                ["ready"] = PlayerReady(),
            };
            if (eventName != null) obs["event"] = eventName;

            var nm = MonoSingleton<NewMovement>.Instance;
            var cc = MonoSingleton<CameraController>.Instance;
            if (nm == null || cc == null)
            {
                obs["player"] = null;
                obs["enemies"] = new JArray();
                return obs;
            }

            var cam = cc.cam != null ? cc.cam.transform : cc.transform;
            var playerPos = nm.transform.position;
            var envMask = LayerMaskDefaults.Get(LMD.Environment);

            obs["player"] = BuildPlayer(nm, cc, cam);
            obs["enemies"] = BuildEnemies(cam, envMask);
            obs["rays"] = BuildHorizontalRays(nm, playerPos, envMask);
            obs["ground_rays"] = BuildGroundRays(nm, playerPos, envMask);
            obs["stats"] = BuildStats(nm);

            var grid = MonoSingleton<EndlessGrid>.Instance;
            if (grid != null)
            {
                var anw = GetAnw(grid);
                var cg = new JObject
                {
                    ["wave"] = grid.currentWave,
                    ["enemies_left"] = anw != null ? Mathf.Max(0, grid.tempEnemyAmount - anw.deadEnemies) : -1,
                };
                // The trigger that starts wave 1 when the player enters it (disabled once waves start).
                var trigger = grid.GetComponent<Collider>();
                if (trigger != null && trigger.enabled)
                {
                    cg["start_trigger"] = new JObject
                    {
                        ["center"] = Vec(trigger.bounds.center),
                        ["size"] = Vec(trigger.bounds.size),
                    };
                }
                obs["cybergrind"] = cg;
            }

            return obs;
        }

        private static JObject BuildPlayer(NewMovement nm, CameraController cc, Transform cam)
        {
            var vel = nm.rb != null ? nm.rb.velocity : Vector3.zero;
            var gun = MonoSingleton<GunControl>.Instance;
            return new JObject
            {
                ["pos"] = Vec(nm.transform.position),
                ["vel"] = Vec(vel),
                ["local_vel"] = Vec(nm.transform.InverseTransformDirection(vel)),
                ["forward"] = Vec(cam.forward),
                ["yaw"] = cc.rotationY,
                ["pitch"] = cc.rotationX,
                ["hp"] = nm.hp,
                ["anti_hp"] = nm.antiHp,
                ["stamina"] = nm.boostCharge,
                ["grounded"] = nm.gc != null && nm.gc.onGround,
                ["sliding"] = nm.sliding,
                ["dead"] = nm.dead,
                ["activated"] = nm.activated,
                ["level_over"] = nm.levelOver,
                ["weapon_slot"] = gun != null ? gun.currentSlotIndex : -1,
                ["weapon_variation"] = gun != null ? gun.currentVariationIndex : -1,
            };
        }

        private JArray BuildEnemies(Transform cam, int envMask)
        {
            var arr = new JArray();
            var tracker = MonoSingleton<EnemyTracker>.Instance;
            if (tracker == null) return arr;

            sorted.Clear();
            foreach (var eid in tracker.GetCurrentEnemies())
            {
                if (eid == null || eid.dead || eid.blessed) continue;
                sorted.Add((eid, Vector3.Distance(cam.position, Center(eid))));
            }
            sorted.Sort((a, b) => a.dist.CompareTo(b.dist));

            for (int i = 0; i < sorted.Count && i < MaxEnemies; i++)
            {
                var (eid, dist) = sorted[i];
                var center = Center(eid);
                bool visible = !Physics.Linecast(cam.position, center, envMask, QueryTriggerInteraction.Ignore);
                arr.Add(new JObject
                {
                    ["id"] = eid.GetInstanceID(),
                    ["type"] = (int)eid.enemyType,
                    ["type_name"] = eid.enemyType.ToString(),
                    ["health"] = eid.health,
                    ["pos"] = Vec(center),
                    ["rel"] = Vec(cam.InverseTransformPoint(center)),
                    ["dist"] = dist,
                    ["visible"] = visible,
                });
            }
            return arr;
        }

        private static Vector3 Center(EnemyIdentifier eid)
        {
            var c = eid.GetCenter();
            return c != null ? c.position : eid.transform.position;
        }

        /// <summary>Distances to walls in a ring around the player, starting straight ahead and going clockwise.</summary>
        private JArray BuildHorizontalRays(NewMovement nm, Vector3 origin, int envMask)
        {
            var arr = new JArray();
            origin += nm.transform.up * 0.5f;
            for (int i = 0; i < HorizontalRays; i++)
            {
                var dir = Quaternion.AngleAxis(360f * i / HorizontalRays, nm.transform.up) * nm.transform.forward;
                arr.Add(Physics.Raycast(origin, dir, out var hit, RayLength, envMask, QueryTriggerInteraction.Ignore) ? hit.distance : RayLength);
            }
            return arr;
        }

        /// <summary>
        /// Height of the ground below points on a ring around the player. A value near the max means a pit.
        /// </summary>
        private JArray BuildGroundRays(NewMovement nm, Vector3 origin, int envMask)
        {
            var arr = new JArray();
            var up = nm.transform.up;
            for (int i = 0; i < GroundRays; i++)
            {
                var offset = Quaternion.AngleAxis(360f * i / GroundRays, up) * nm.transform.forward * GroundRayRadius;
                var start = origin + offset + up;
                arr.Add(Physics.Raycast(start, -up, out var hit, GroundRayLength, envMask, QueryTriggerInteraction.Ignore) ? hit.distance - 1f : GroundRayLength);
            }
            return arr;
        }

        private static JObject BuildStats(NewMovement nm)
        {
            var sm = MonoSingleton<StatsManager>.Instance;
            if (sm == null) return new JObject();
            return new JObject
            {
                ["kills"] = sm.kills,
                ["style"] = sm.stylePoints,
                ["seconds"] = sm.seconds,
                ["restarts"] = sm.restarts,
                ["level_complete"] = sm.infoSent || nm.levelOver,
            };
        }

        private static JArray Vec(Vector3 v) => new JArray(v.x, v.y, v.z);
    }
}
