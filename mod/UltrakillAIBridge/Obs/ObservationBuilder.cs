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

        // Same idea for NewMovement.crouching, which is private: a rename costs the crouching flag, not the obs.
        private static AccessTools.FieldRef<NewMovement, bool> crouchingRef;
        private static bool crouchingResolved;

        private static readonly List<EnemyIdentifier> EmptyEnemies = new List<EnemyIdentifier>();

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

        private static bool GetCrouching(NewMovement nm)
        {
            if (!crouchingResolved)
            {
                crouchingResolved = true;
                try
                {
                    crouchingRef = AccessTools.FieldRefAccess<NewMovement, bool>("crouching");
                }
                catch (System.Exception e)
                {
                    Plugin.Log.LogWarning($"NewMovement.crouching not found, reported as false: {e.Message}");
                }
            }
            return crouchingRef != null && crouchingRef.Invoke(nm);
        }

        private readonly List<(EnemyIdentifier eid, float dist)> sorted = new List<(EnemyIdentifier, float)>();
        private readonly CampaignObserver campaign = new CampaignObserver();

        // The enemies BuildEnemies already fetched from EnemyTracker this step, reused for the campaign
        // block's arena_enemies_alive so CampaignObserver doesn't re-walk and re-allocate the same list.
        private List<EnemyIdentifier> currentEnemies = new List<EnemyIdentifier>();

        // Set once a campaign-block failure has been logged, so a persistent failure at 150-250 steps/s
        // across five instances can't flood the log for hours; the failure itself repeats every step.
        private bool campaignBuildFailWarned;

        public void Configure(JObject cfg)
        {
            if (cfg == null) return;
            MaxEnemies = cfg["max_enemies"]?.Value<int>() ?? MaxEnemies;
            HorizontalRays = cfg["horizontal_rays"]?.Value<int>() ?? HorizontalRays;
            GroundRays = cfg["ground_rays"]?.Value<int>() ?? GroundRays;
            RayLength = cfg["ray_length"]?.Value<float>() ?? RayLength;
            GroundRayRadius = cfg["ground_ray_radius"]?.Value<float>() ?? GroundRayRadius;
            GroundRayLength = cfg["ground_ray_length"]?.Value<float>() ?? GroundRayLength;
            ReportMemory = cfg["report_memory"]?.Value<bool>() ?? ReportMemory;
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
            obs["ground_ray_center"] = GroundRayCenter(nm, playerPos, envMask);
            obs["stats"] = BuildStats(nm);
            if (ReportMemory) obs["mem"] = BuildMemory();

            var sm = MonoSingleton<StatsManager>.Instance;
            if (CampaignObserver.IsCampaignScene(sm))
            {
                // A campaign-block bug must not become an obs-build exception: EpisodeController.EndOfFrame's
                // outer catch would turn the whole reply into a "type":"error", which the Python client raises
                // as BridgeError, killing that SubprocVecEnv worker and the whole training run. Degrade instead:
                // omit the block for this step, same as GetAnw degrades a missing field.
                try
                {
                    obs["campaign"] = campaign.Build(nm, sm, currentEnemies);
                }
                catch (System.Exception e)
                {
                    if (!campaignBuildFailWarned)
                    {
                        campaignBuildFailWarned = true;
                        Plugin.Log.LogWarning($"CampaignObserver.Build failed, omitting campaign block: {e}");
                    }
                }
            }

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
                ["slow_mode"] = nm.slowMode,
                ["heavy_fall"] = nm.gc != null && nm.gc.heavyFall,
                ["crouching"] = GetCrouching(nm),
                ["dead"] = nm.dead,
                ["activated"] = nm.activated,
                ["level_over"] = nm.levelOver,
                ["weapon_slot"] = gun != null ? gun.currentSlotIndex : -1,
                ["weapon_variation"] = gun != null ? gun.currentVariationIndex : -1,
                ["slot_counts"] = SlotCounts(gun),
                ["soft_deaths"] = Env.TrainingSpeed.SoftDeaths,
                ["soft_death_instakill"] = Env.TrainingSpeed.LastSoftDeathInstakill,
            };
        }

        /// <summary>Weapons in each slot, slot 1 first (empty until GunControl has started).</summary>
        private static JArray SlotCounts(GunControl gun)
        {
            var arr = new JArray();
            if (gun == null || gun.slots == null) return arr;
            foreach (var slot in gun.slots)
            {
                arr.Add(slot != null ? slot.Count : 0);
            }
            return arr;
        }

        private JArray BuildEnemies(Transform cam, int envMask)
        {
            var arr = new JArray();
            var tracker = MonoSingleton<EnemyTracker>.Instance;
            // Fetched once here and kept in currentEnemies for CampaignObserver to reuse for
            // arena_enemies_alive, since GetCurrentEnemies() allocates and refills a list every call.
            currentEnemies = tracker != null ? tracker.GetCurrentEnemies() : EmptyEnemies;
            if (tracker == null) return arr;

            sorted.Clear();
            foreach (var eid in currentEnemies)
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

        /// <summary>
        /// Height of the ground directly below the player, same convention as <see cref="BuildGroundRays"/>
        /// (the ray length exactly when nothing is hit, negative when the ground is above the player's
        /// feet). The ring's minimum can be a ledge four metres away rather than the floor underfoot, so
        /// the centre is the measure that says where the player actually is.
        /// </summary>
        private float GroundRayCenter(NewMovement nm, Vector3 origin, int envMask)
        {
            var up = nm.transform.up;
            return Physics.Raycast(origin + up, -up, out var hit, GroundRayLength, envMask, QueryTriggerInteraction.Ignore)
                ? hit.distance - 1f
                : GroundRayLength;
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

        /// <summary>
        /// Three floats, without the params-array and the boxing the obvious version costs.
        ///
        /// `new JArray(v.x, v.y, v.z)` binds JArray(params object[]) -- an object[3] plus THREE BOXED FLOATS
        /// per call, on top of the JArray, its backing List and the three JValues. This runs ~40 times a step
        /// at ~20 steps a second forever, and the per-step JSON tree is the mod's one allocation of the right
        /// order of magnitude to matter on Unity 2022.3's non-compacting Boehm GC (~1.8 MB/s, ~35 GB over a
        /// 5.5 h session). Nothing here is RETAINED -- see docs/notes/2026-09-18-memory.md -- but the heap's
        /// high-water mark is a function of the churn, so the churn is worth cutting where it is free.
        /// </summary>
        internal static JArray Vec(Vector3 v)
        {
            var a = new JArray();
            a.Add(new JValue(v.x));
            a.Add(new JValue(v.y));
            a.Add(new JValue(v.z));
            return a;
        }

        /// <summary>
        /// Whether the observation carries the `mem` block. Off by default: it is a diagnostic for the leak
        /// hunt, set through the config key `report_memory`, and Python ignores an unknown obs field either way.
        /// </summary>
        public static bool ReportMemory { get; set; }

        /// <summary>
        /// Unity's own memory counters, which is what decides WHERE the games' 1.2-1.4 GB an hour goes.
        ///
        /// The mod retains almost nothing across scene loads (audited 2026-09-18), so the growth is one of two
        /// things and they need opposite fixes: the Mono heap's high-water mark growing under per-step JSON
        /// churn, or native asset memory accumulating because nothing on the game's level-change path ever
        /// calls Resources.UnloadUnusedAssets (grep over all 1178 decompiled files hits one file, and it is
        /// the sandbox saver). Logging these four beside the process's private bytes across 60 fresh level
        /// loads separates them in one session:
        ///
        ///     mono_heap tracks private bytes      -> the JSON churn; stop building a JObject tree
        ///     native_reserved tracks private bytes -> assets and bundles; sweep them after a level load
        ///
        /// All four are cheap native reads of counters Unity already maintains.
        /// </summary>
        private static JObject BuildMemory()
        {
            return new JObject
            {
                ["mono_used"] = UnityEngine.Profiling.Profiler.GetMonoUsedSizeLong(),
                ["mono_heap"] = UnityEngine.Profiling.Profiler.GetMonoHeapSizeLong(),
                ["native_alloc"] = UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong(),
                ["native_reserved"] = UnityEngine.Profiling.Profiler.GetTotalReservedMemoryLong(),
            };
        }
    }
}
