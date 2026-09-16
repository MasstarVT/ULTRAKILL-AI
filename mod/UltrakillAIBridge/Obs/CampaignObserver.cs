using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UltrakillAIBridge.Env;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.SceneManagement;
using Object = UnityEngine.Object;

namespace UltrakillAIBridge.Obs
{
    /// <summary>
    /// Builds the obs "campaign" block in the 35 main levels: the exit, checkpoints, a NavMesh path hint to
    /// the exit, locked doors, arena enemies, milestone keys and rank thresholds (keys in docs/protocol.md).
    ///
    /// The exit and later checkpoints can sit in rooms that start inactive, so scene objects are found with
    /// FindObjectsOfType(includeInactive). That also returns the disabled room templates CheckPoint.Start
    /// keeps (moved +10000 on X from wherever the room was, so no X threshold separates them), so anything
    /// under an entry of some checkpoint's defaultRooms is skipped. The search is cached and repeated every
    /// <see cref="RescanEvery"/> builds, on a new scene, after a checkpoint respawn (which destroys and
    /// re-creates the rooms it owns), and when a checkpoint changes state: CheckPoint.Start and a repeated
    /// ActivateCheckPoint (InheritRoom) turn live rooms into templates, and a stale cache would report the
    /// objects inside them at +10000 X under new ids.
    /// </summary>
    public sealed class CampaignObserver
    {
        private const int RescanEvery = 30;
        private const int PathEvery = 4;
        private const int MaxLockedDoors = 4;
        private const float PlayerSnapDistance = 6f;
        private const float ExitSnapDistance = 20f;
        private const float CornerReachedDistance = 1.5f;

        private readonly List<FinalPit> pits = new List<FinalPit>();
        private readonly List<CheckPoint> checkpoints = new List<CheckPoint>();
        private readonly List<Door> doors = new List<Door>();
        private readonly HashSet<Transform> templates = new HashSet<Transform>();
        private readonly List<(Door door, float dist)> lockedDoors = new List<(Door, float)>();

        private bool scanned;
        private int sceneHandle;
        private int lastRestarts;
        private int checkpointSignature;
        private int builds;

        // NavMesh path to the exit, recalculated every PathEvery builds (pathStatus "none" is a cached result too).
        private NavMeshPath navPath;
        private bool pathCached;
        private string pathStatus = "none";
        private float pathLength;
        private Vector3 nextCorner;

        public static bool IsCampaignScene(StatsManager sm)
        {
            var scene = SceneHelper.CurrentScene;
            return sm != null && sm.levelNumber >= 1 && sm.levelNumber <= 35
                   && scene != null && scene.StartsWith("Level ", StringComparison.Ordinal);
        }

        public JObject Build(NewMovement nm, StatsManager sm)
        {
            int handle = SceneManager.GetActiveScene().handle;
            if (!scanned || handle != sceneHandle || sm.restarts != lastRestarts)
            {
                // A new scene, or a checkpoint respawn that re-created rooms and moved the player.
                sceneHandle = handle;
                builds = 0;
                pathCached = false;
            }
            else if (CheckpointSignature(sm) != checkpointSignature)
            {
                // A checkpoint activated, took over rooms or became current: live rooms may now be templates.
                builds = 0;
            }
            if (builds % RescanEvery == 0) Scan(sm);

            var playerPos = nm.transform.position;
            var exit = ChooseExit();
            if (!pathCached || builds % PathEvery == 0) UpdatePath(playerPos, exit);
            builds++;

            var prefs = MonoSingleton<PrefsManager>.Instance;
            var gsm = GameStateManager.Instance;
            return new JObject
            {
                ["mission"] = sm.levelNumber,
                ["difficulty"] = prefs != null ? prefs.GetInt("difficulty") : -1,
                ["seconds"] = sm.seconds,
                ["timer_running"] = sm.timer,
                ["level_started"] = sm.levelStarted,
                ["level_over"] = nm.levelOver,
                ["restarts"] = sm.restarts,
                ["input_locked"] = (gsm != null && gsm.PlayerInputLocked) || !nm.activated,
                ["exit"] = exit != null
                    ? new JObject { ["pos"] = ObservationBuilder.Vec(exit.transform.position), ["active"] = exit.gameObject.activeInHierarchy }
                    : null,
                ["checkpoints"] = BuildCheckpoints(sm),
                ["path"] = BuildPath(),
                ["locked_doors"] = BuildLockedDoors(playerPos),
                ["arena_enemies_alive"] = ArenaEnemiesAlive(),
                ["cleared_arenas"] = Strings(CampaignPatches.ClearedArenas),
                ["unlocked_doors"] = Strings(CampaignPatches.UnlockedDoors),
                ["ranks"] = new JObject
                {
                    ["time"] = Ints(sm.timeRanks),
                    ["kills"] = Ints(sm.killRanks),
                    ["style"] = Ints(sm.styleRanks),
                },
            };
        }

        private void Scan(StatsManager sm)
        {
            scanned = true;
            lastRestarts = sm.restarts;

            var allCheckpoints = Object.FindObjectsOfType<CheckPoint>(true);
            templates.Clear();
            foreach (var cp in allCheckpoints)
            {
                if (cp == null || cp.defaultRooms == null) continue;
                foreach (var room in cp.defaultRooms)
                {
                    if (room != null) templates.Add(room.transform);
                }
            }

            checkpoints.Clear();
            foreach (var cp in allCheckpoints)
            {
                if (cp != null && !IsTemplate(cp.transform)) checkpoints.Add(cp);
            }
            pits.Clear();
            foreach (var pit in Object.FindObjectsOfType<FinalPit>(true))
            {
                if (pit != null && !pit.fakeEnd && !pit.secondPit && !pit.rankless && !IsTemplate(pit.transform)) pits.Add(pit);
            }
            doors.Clear();
            foreach (var door in Object.FindObjectsOfType<Door>(true))
            {
                if (door != null && !IsTemplate(door.transform)) doors.Add(door);
            }
            checkpointSignature = CheckpointSignature(sm);
        }

        /// <summary>
        /// Summary of the cached checkpoints' state (current checkpoint, activated flags, owned room counts,
        /// destroyed entries). It only changes on checkpoint events, so comparing it every build is cheap.
        /// </summary>
        private int CheckpointSignature(StatsManager sm)
        {
            unchecked
            {
                int sig = sm.currentCheckPoint != null ? sm.currentCheckPoint.GetInstanceID() : 0;
                foreach (var cp in checkpoints)
                {
                    if (cp == null)
                    {
                        sig = sig * 31 + 1;
                        continue;
                    }
                    sig = sig * 31 + (cp.activated ? 3 : 2);
                    sig = sig * 31 + (cp.defaultRooms != null ? cp.defaultRooms.Count : 0);
                }
                return sig;
            }
        }

        /// <summary>True when the object or any ancestor is a room template kept by a checkpoint.</summary>
        private bool IsTemplate(Transform t)
        {
            for (; t != null; t = t.parent)
            {
                if (templates.Contains(t)) return true;
            }
            return false;
        }

        /// <summary>The real exit; an active pit is preferred over one in a room that hasn't loaded yet.</summary>
        private FinalPit ChooseExit()
        {
            FinalPit inactive = null;
            foreach (var pit in pits)
            {
                if (pit == null) continue;
                if (pit.gameObject.activeInHierarchy) return pit;
                if (inactive == null) inactive = pit;
            }
            return inactive;
        }

        /// <summary>
        /// NavMesh path from the player to the exit, both ends snapped onto the mesh. The mesh only covers
        /// walkable ground (jumps and gaps are not linked, doors carry obstacles), so a partial path is normal.
        /// </summary>
        private void UpdatePath(Vector3 playerPos, FinalPit exit)
        {
            pathCached = true;
            pathStatus = "none";
            if (exit == null) return;
            if (!NavMesh.SamplePosition(playerPos, out var from, PlayerSnapDistance, NavMesh.AllAreas)) return;
            if (!NavMesh.SamplePosition(exit.transform.position, out var to, ExitSnapDistance, NavMesh.AllAreas)) return;

            if (navPath == null) navPath = new NavMeshPath();
            if (!NavMesh.CalculatePath(from.position, to.position, NavMesh.AllAreas, navPath)) return;
            if (navPath.status == NavMeshPathStatus.PathInvalid) return;
            var corners = navPath.corners;
            if (corners.Length == 0) return;

            pathLength = Vector3.Distance(playerPos, corners[0]);
            for (int i = 1; i < corners.Length; i++)
            {
                pathLength += Vector3.Distance(corners[i - 1], corners[i]);
            }
            nextCorner = corners[corners.Length - 1];
            foreach (var corner in corners)
            {
                var flat = corner - playerPos;
                flat.y = 0f;
                if (flat.magnitude > CornerReachedDistance)
                {
                    nextCorner = corner;
                    break;
                }
            }
            pathStatus = navPath.status == NavMeshPathStatus.PathComplete ? "complete" : "partial";
        }

        private JObject BuildPath()
        {
            if (pathStatus == "none") return new JObject { ["status"] = "none" };
            return new JObject
            {
                ["status"] = pathStatus,
                ["length"] = pathLength,
                ["next_corner"] = ObservationBuilder.Vec(nextCorner),
            };
        }

        private JArray BuildCheckpoints(StatsManager sm)
        {
            var arr = new JArray();
            foreach (var cp in checkpoints)
            {
                if (cp == null) continue;
                var pos = cp.transform.position;
                arr.Add(new JObject
                {
                    ["id"] = CampaignPatches.Key(pos),
                    ["pos"] = ObservationBuilder.Vec(pos),
                    ["activated"] = cp.activated,
                    ["current"] = sm.currentCheckPoint == cp,
                });
            }
            return arr;
        }

        private JArray BuildLockedDoors(Vector3 playerPos)
        {
            lockedDoors.Clear();
            foreach (var door in doors)
            {
                if (door == null || !door.locked || !door.gameObject.activeInHierarchy) continue;
                lockedDoors.Add((door, Vector3.Distance(playerPos, door.transform.position)));
            }
            lockedDoors.Sort((a, b) => a.dist.CompareTo(b.dist));

            var arr = new JArray();
            for (int i = 0; i < lockedDoors.Count && i < MaxLockedDoors; i++)
            {
                var (door, dist) = lockedDoors[i];
                arr.Add(new JObject
                {
                    ["pos"] = ObservationBuilder.Vec(door.transform.position),
                    ["dist"] = dist,
                });
            }
            return arr;
        }

        /// <summary>Live enemies belonging to an arena wave that hasn't been cleared yet.</summary>
        private static int ArenaEnemiesAlive()
        {
            var tracker = MonoSingleton<EnemyTracker>.Instance;
            if (tracker == null) return 0;
            int alive = 0;
            foreach (var eid in tracker.GetCurrentEnemies())
            {
                if (eid == null || eid.dead) continue;
                var wave = eid.GetComponentInParent<ActivateNextWave>();
                if (wave != null && !wave.activated) alive++;
            }
            return alive;
        }

        private static JArray Strings(IEnumerable<string> values)
        {
            var arr = new JArray();
            foreach (var value in values)
            {
                arr.Add(value);
            }
            return arr;
        }

        private static JArray Ints(int[] values)
        {
            var arr = new JArray();
            if (values == null) return arr;
            foreach (var value in values)
            {
                arr.Add(value);
            }
            return arr;
        }
    }
}
