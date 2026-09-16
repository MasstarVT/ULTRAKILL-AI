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
        // The player end is snapped too, and ULTRAKILL is played in the air: jumping, dashing and falling are
        // most of a run. At 6 m an airborne player can miss the mesh, and a missed sample reports the whole
        // path as "none", withholding the path reward and the policy's next-corner input until they land.
        // 25 m covers a jump or a drop between floors while still snapping to ground the player is above.
        // (This is not why Level 0-1 reads "none" at spawn: measured there, the player is grounded and snaps
        // fine, but the mesh island holding the exit is not connected to the start area, so CalculatePath
        // finds nothing at all. The path hint only appears once the agent is far enough through the level.)
        private const float PlayerSnapDistance = 25f;
        private const float ExitSnapDistance = 20f;
        private const float CornerReachedDistance = 1.5f;

        /// <summary>
        /// Escalating search radii tried when <see cref="ExitSnapDistance"/> finds nothing at a
        /// <see cref="FinalPit"/>'s own position. A FinalPit's transform sits inside the drop it triggers,
        /// not on walkable ground, so the nearest NavMesh isn't at a fixed offset from it: logging every
        /// NavMesh.SamplePosition candidate in game (Level 0-1, Level 1-1) showed the nearest point is 47.5 m
        /// away straight down in one level and 84.6 m away up and 46.8 m to the side in the other -- not "a
        /// few metres above" as first assumed from the pit's raw depth (17.1 m / 76.1 m below the floor).
        /// A wider search radius at the unchanged pit position finds the same nearest point regardless of
        /// its direction, so this replaces directional probing. Both measured points connect back to the
        /// player's side of the mesh only as NavMeshPathStatus.PathPartial (BuildPath already reports that
        /// as a valid "partial" status), which matches a FinalPit deliberately sitting off the walkable graph.
        /// 55/95 clears both measured cases with a few metres of slack.
        /// </summary>
        private static readonly float[] ExitSnapRadii = { 55f, 95f };

        private readonly List<FinalPit> pits = new List<FinalPit>();
        private readonly List<CheckPoint> checkpoints = new List<CheckPoint>();
        // Id strings for checkpoints, same index as checkpoints. Computed once per Scan() rather than
        // once per Build(): checkpoint positions don't move between scans, so re-formatting the same
        // "x,y,z" string every step (up to 250/s across five instances) is a pure waste.
        private readonly List<string> checkpointIds = new List<string>();
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

        /// <param name="enemies">
        /// The enemies ObservationBuilder.BuildEnemies already fetched from EnemyTracker this step, reused for
        /// arena_enemies_alive instead of querying the tracker again (it allocates and refills a list per call).
        /// </param>
        public JObject Build(NewMovement nm, StatsManager sm, List<EnemyIdentifier> enemies)
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
                // The path itself didn't necessarily move, but treat it the same as the scan invalidation
                // above so both branches leave the cache in the same state.
                builds = 0;
                pathCached = false;
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
                ["arena_enemies_alive"] = ArenaEnemiesAlive(enemies),
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
            checkpointIds.Clear();
            foreach (var cp in allCheckpoints)
            {
                if (cp == null || IsTemplate(cp.transform)) continue;
                checkpoints.Add(cp);
                checkpointIds.Add(CampaignPatches.Key(cp.transform.position));
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
        /// The exit end is snapped by <see cref="SampleExit"/>, which may land far from the pit's own
        /// position (see <see cref="ExitSnapRadii"/>); <c>pathLength</c> is measured through
        /// <c>NavMeshPath.corners</c> to that snapped point, i.e. it always stays "distance the player still
        /// has to walk" to reach the nearest standable point back on the mesh, never a straight line into the
        /// pit itself.
        /// </summary>
        private void UpdatePath(Vector3 playerPos, FinalPit exit)
        {
            pathCached = true;
            pathStatus = "none";
            if (exit == null) return;
            if (!NavMesh.SamplePosition(playerPos, out var from, PlayerSnapDistance, NavMesh.AllAreas)) return;
            if (!SampleExit(exit.transform.position, out var to)) return;

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

        /// <summary>
        /// Samples the NavMesh point nearest a <see cref="FinalPit"/>, escalating the search radius instead
        /// of guessing a direction: see <see cref="ExitSnapRadii"/> for why (the nearest point measured in
        /// game was straight down for one pit and diagonally up and sideways for another). Tries
        /// <see cref="ExitSnapDistance"/> at the raw position first, so a pit that already sits on/near the
        /// mesh keeps working exactly as before, then each wider radius in turn, stopping at the first
        /// NavMesh hit. Bounded to at most 1 + <c>ExitSnapRadii.Length</c> SamplePosition calls, cheap enough
        /// for the 150-250 steps/s this runs at across five games.
        /// </summary>
        private static bool SampleExit(Vector3 exitPos, out NavMeshHit hit)
        {
            if (NavMesh.SamplePosition(exitPos, out hit, ExitSnapDistance, NavMesh.AllAreas)) return true;
            foreach (var radius in ExitSnapRadii)
            {
                if (NavMesh.SamplePosition(exitPos, out hit, radius, NavMesh.AllAreas)) return true;
            }
            return false;
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
            for (int i = 0; i < checkpoints.Count; i++)
            {
                var cp = checkpoints[i];
                if (cp == null) continue;
                arr.Add(new JObject
                {
                    ["id"] = checkpointIds[i],
                    ["pos"] = ObservationBuilder.Vec(cp.transform.position),
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

        /// <summary>
        /// Live enemies belonging to an arena wave that hasn't been cleared yet. Takes the enemies
        /// ObservationBuilder.BuildEnemies already fetched this step rather than querying EnemyTracker
        /// again, since GetCurrentEnemies() allocates and refills a list on every call.
        /// </summary>
        private static int ArenaEnemiesAlive(List<EnemyIdentifier> enemies)
        {
            if (enemies == null) return 0;
            int alive = 0;
            foreach (var eid in enemies)
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
