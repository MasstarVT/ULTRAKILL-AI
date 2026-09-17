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
    /// the exit, locked doors, arena enemies, the door-graph route ("gates"), milestone keys and rank
    /// thresholds (keys in docs/protocol.md).
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

        /// <summary>
        /// Most gates reported per level. Measured candidate counts are 11 (0-1), 17 (0-2), 12 (0-3),
        /// 7 (0-4), 3 (0-5) and 13 (1-1), so this is only a wire-size guard; a level that exceeds it keeps
        /// the gates nearest the player (the ones the agent needs first) and sets "gates_truncated".
        /// </summary>
        private const int MaxGates = 64;
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

        // The door-graph route. The doors, the room graph, the keys and the hop counts are all decided in
        // Scan(); a step only reads four live flags off each Door (see BuildGates).
        private readonly List<Gate> gates = new List<Gate>();
        private bool gatesOrdered;
        private bool gatesTruncated;
        // hops per gate key for THIS level load. Rules 7 and 12 of the spec: a Scan() that lands while a
        // checkpoint respawn is re-creating the exit's room finds no goal room, and renumbering the route
        // under the agent (or blanking it) would move the target for reasons the player had nothing to do
        // with. Keys are stable across re-instantiation, so a key that has ever had a hops value keeps it
        // until the scene changes.
        private readonly Dictionary<string, int> hopsByKey = new Dictionary<string, int>();
        private bool duplicateKeysWarned;

        // Scratch for the room graph, reused across scans.
        private readonly HashSet<string> roomNodes = new HashSet<string>();
        private readonly Dictionary<string, List<string>> roomEdges = new Dictionary<string, List<string>>();
        private readonly Dictionary<string, int> roomHops = new Dictionary<string, int>();
        private readonly Queue<string> bfs = new Queue<string>();
        private readonly HashSet<int> roomIds = new HashSet<int>();
        private readonly HashSet<string> gateKeys = new HashSet<string>();

        // The FinalPit chosen for this level load. Rule 4 of the spec: the preference order is
        // time-varying (a pit's room activates mid-run) and every hops value depends on which pit was
        // chosen, so re-running it mid-load could silently renumber the route.
        private FinalPit chosenExit;

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
            if (!scanned || handle != sceneHandle)
            {
                // A new scene: the route, the chosen exit and the duplicate-key warning are all
                // level-load scoped (a checkpoint respawn must NOT clear them, see hopsByKey).
                sceneHandle = handle;
                builds = 0;
                pathCached = false;
                hopsByKey.Clear();
                chosenExit = null;
                duplicateKeysWarned = false;
            }
            else if (sm.restarts != lastRestarts)
            {
                // A checkpoint respawn that re-created rooms and moved the player.
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
            var playerPos = nm.transform.position;
            if (builds % RescanEvery == 0) Scan(sm, playerPos);

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
                ["gates_ordered"] = gatesOrdered,
                ["gates_truncated"] = gatesTruncated,
                ["gates"] = BuildGates(),
                ["ranks"] = new JObject
                {
                    ["time"] = Ints(sm.timeRanks),
                    ["kills"] = Ints(sm.killRanks),
                    ["style"] = Ints(sm.styleRanks),
                },
            };
        }

        private void Scan(StatsManager sm, Vector3 playerPos)
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
                if (pit == null || pit.fakeEnd || pit.secondPit || pit.rankless || IsTemplate(pit.transform)) continue;
                // A pit with no target, or one that drops into a secret level ("Level 0-S"), is never the
                // mission exit. Measured on 0-2 and 1-1, which each carry both kinds; after this filter
                // exactly one pit is left on every level checked (0-1..0-5, 1-1).
                var target = pit.targetLevelName;
                if (string.IsNullOrEmpty(target) || target.EndsWith("-S", StringComparison.OrdinalIgnoreCase)) continue;
                pits.Add(pit);
            }
            doors.Clear();
            foreach (var door in Object.FindObjectsOfType<Door>(true))
            {
                if (door != null && !IsTemplate(door.transform)) doors.Add(door);
            }
            checkpointSignature = CheckpointSignature(sm);
            ScanGates(playerPos);
        }

        /// <summary>
        /// One door that connects two or more rooms, with everything about it a scan decides: its
        /// closed-position key and position, its hop count to the exit and its DoorControllers.
        /// </summary>
        private sealed class Gate
        {
            public Door Door;
            public DoorController[] Controllers;
            public string Key;
            public Vector3 Pos;
            public int? Hops;
            public List<string> Rooms;
            public float Dist;
        }

        /// <summary>
        /// Rebuilds the route: which doors are gates, the room graph they form, and each gate's hop count
        /// from the room holding the exit. Runs inside <see cref="Scan"/>, i.e. at most once every
        /// <see cref="RescanEvery"/> builds, so a step never walks the scene or the graph again.
        ///
        /// A door is a gate when its activatedRooms holds at least two distinct GameObjects; the rooms of
        /// those doors (and nothing else) are the graph's nodes, identified by their rounded world position
        /// rather than by reference, because CheckPoint.ResetRoom destroys and re-instantiates a room on
        /// every respawn. Two rooms are adjacent when one gate lists both, and a BFS from the exit's own
        /// room gives every room its hop count; a gate takes the lowest hop count among its rooms.
        ///
        /// The JSON objects themselves are built per step by <see cref="BuildGates"/> rather than cached:
        /// a JToken that already has a parent is deep-copied when it is assigned to the next step's obs,
        /// so caching them would pay for a clone of the whole array every step and look like it didn't.
        /// </summary>
        private void ScanGates(Vector3 playerPos)
        {
            gates.Clear();
            roomNodes.Clear();
            roomEdges.Clear();
            gateKeys.Clear();

            int duplicates = 0;
            foreach (var door in doors)
            {
                if (door == null || door.activatedRooms == null) continue;
                roomIds.Clear();
                List<string> rooms = null;
                foreach (var room in door.activatedRooms)
                {
                    if (room == null || !roomIds.Add(room.GetInstanceID())) continue;
                    if (rooms == null) rooms = new List<string>(door.activatedRooms.Length);
                    var node = CampaignPatches.Key(room.transform.position);
                    if (!rooms.Contains(node)) rooms.Add(node);
                }
                if (roomIds.Count < 2) continue;

                var pos = ClosedPosition(door);
                var key = CampaignPatches.Key(pos);
                // CampaignPatches.Key rounds to whole metres, so two doors within a metre would collide,
                // and Python uses the key alone to tell one gate from another.
                if (!gateKeys.Add(key))
                {
                    duplicates++;
                    var collided = key;
                    for (int n = 2; ; n++)
                    {
                        key = collided + "#" + n.ToString(System.Globalization.CultureInfo.InvariantCulture);
                        if (gateKeys.Add(key)) break;
                    }
                }
                gates.Add(new Gate
                {
                    Door = door,
                    Controllers = FindControllers(door),
                    Key = key,
                    Pos = pos,
                    Rooms = rooms,
                    Dist = Vector3.Distance(playerPos, pos),
                });

                foreach (var node in rooms)
                {
                    roomNodes.Add(node);
                    if (!roomEdges.TryGetValue(node, out var neighbours))
                    {
                        neighbours = new List<string>();
                        roomEdges[node] = neighbours;
                    }
                    foreach (var other in rooms)
                    {
                        if (other != node && !neighbours.Contains(other)) neighbours.Add(other);
                    }
                }
            }
            if (duplicates > 0 && !duplicateKeysWarned)
            {
                duplicateKeysWarned = true;
                Plugin.Log.LogWarning($"{duplicates} gate position key(s) collide in {SceneHelper.CurrentScene}, suffixed with #N");
            }

            gatesTruncated = gates.Count > MaxGates;
            if (gatesTruncated)
            {
                // Keep the ones nearest the player: the agent needs the gates around it long before the
                // ones near the exit, so dropping the highest hop counts would drop exactly the wrong end.
                gates.Sort((a, b) => a.Dist.CompareTo(b.Dist));
                gates.RemoveRange(MaxGates, gates.Count - MaxGates);
            }

            ComputeHops();
            gates.Sort(CompareGates);
            // Sorted hops-first, so only the first entry has to be checked.
            gatesOrdered = gates.Count > 0 && gates[0].Hops.HasValue;
        }

        /// <summary>
        /// BFS from the exit's own room over the room graph, writing each gate's "hops". A gate keeps a
        /// hops value it was given earlier in this level load (see <see cref="hopsByKey"/>).
        /// </summary>
        private void ComputeHops()
        {
            roomHops.Clear();
            var goal = GoalRoom();
            if (goal != null)
            {
                roomHops[goal] = 0;
                bfs.Clear();
                bfs.Enqueue(goal);
                while (bfs.Count > 0)
                {
                    var room = bfs.Dequeue();
                    if (!roomEdges.TryGetValue(room, out var neighbours)) continue;
                    int next = roomHops[room] + 1;
                    foreach (var neighbour in neighbours)
                    {
                        if (roomHops.ContainsKey(neighbour)) continue;
                        roomHops[neighbour] = next;
                        bfs.Enqueue(neighbour);
                    }
                }
            }

            foreach (var gate in gates)
            {
                int hops = int.MaxValue;
                if (gate.Rooms != null)
                {
                    foreach (var room in gate.Rooms)
                    {
                        if (roomHops.TryGetValue(room, out var h) && h < hops) hops = h;
                    }
                }
                if (hopsByKey.TryGetValue(gate.Key, out var known)) hops = known;
                else if (hops != int.MaxValue) hopsByKey[gate.Key] = hops;
                gate.Hops = hops == int.MaxValue ? (int?)null : hops;
            }
        }

        /// <summary>
        /// The room the exit sits in: the first room node on the chosen FinalPit's own transform chain,
        /// starting at the pit. Never "every room under a common root" -- a level whose rooms share one
        /// container would then report every gate at hops 0, which looks valid and means nothing.
        /// </summary>
        private string GoalRoom()
        {
            var exit = ChooseExit();
            if (exit == null) return null;
            for (var t = exit.transform; t != null; t = t.parent)
            {
                var key = CampaignPatches.Key(t.position);
                if (roomNodes.Contains(key)) return key;
            }
            return null;
        }

        /// <summary>Lowest hops first, nulls last, then by key, so Python sees a stable order.</summary>
        private static int CompareGates(Gate a, Gate b)
        {
            if (a.Hops.HasValue != b.Hops.HasValue) return a.Hops.HasValue ? -1 : 1;
            if (a.Hops.HasValue)
            {
                int cmp = a.Hops.Value.CompareTo(b.Hops.Value);
                if (cmp != 0) return cmp;
            }
            return string.CompareOrdinal(a.Key, b.Key);
        }

        /// <summary>
        /// A door's CLOSED world position. Door.Update moves a Normal door's own transform to
        /// closedPos + openPos while it opens (measured (0, 5.75, 0) on every gate door of 0-1..0-5), so
        /// reading transform.position during a rescan while the player stands in the proximity trigger
        /// would capture the open position and change the door's key mid-level. A door whose Awake has not
        /// run yet is still sitting at its closed position, which is what the fallback reads.
        /// </summary>
        private static Vector3 ClosedPosition(Door door)
        {
            if (!door.gotPos) return door.transform.position;
            var parent = door.transform.parent;
            return parent != null ? parent.TransformPoint(door.closedPos) : door.closedPos;
        }

        /// <summary>
        /// The door's DoorControllers, found the way Door.Awake finds them but including inactive ones:
        /// Awake's own search skips inactive objects, so <c>Door.docons</c> is empty for exactly the doors
        /// whose controller an ActivateArena has not switched on yet -- which is the case
        /// "controller_active" exists to report.
        /// </summary>
        private static DoorController[] FindControllers(Door door)
        {
            if (door.doorType != DoorType.Normal) return door.GetComponentsInChildren<DoorController>(true);
            var parent = door.transform.parent;
            return parent != null ? parent.GetComponentsInChildren<DoorController>(true) : new DoorController[0];
        }

        /// <summary>
        /// The gates array with this step's live flags. "open" is transient (a DoorController closes the
        /// door as soon as the player and every enemy leave, and opening one door force-closes the others),
        /// so it means "currently open or opening", never "has been passed". A destroyed door stays in the
        /// array with every flag false rather than disappearing from it.
        /// </summary>
        private JArray BuildGates()
        {
            var arr = new JArray();
            foreach (var gate in gates)
            {
                var door = gate.Door;
                bool alive = door != null;
                arr.Add(new JObject
                {
                    ["key"] = gate.Key,
                    ["pos"] = ObservationBuilder.Vec(gate.Pos),
                    ["hops"] = gate.Hops.HasValue ? new JValue(gate.Hops.Value) : JValue.CreateNull(),
                    ["open"] = alive && door.open,
                    ["locked"] = alive && door.locked,
                    ["active"] = alive && door.gameObject.activeInHierarchy,
                    ["controller_active"] = alive && ControllerActive(gate.Controllers),
                });
            }
            return arr;
        }

        private static bool ControllerActive(DoorController[] controllers)
        {
            if (controllers == null) return false;
            foreach (var controller in controllers)
            {
                if (controller != null && controller.gameObject.activeInHierarchy) return true;
            }
            return false;
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

        /// <summary>
        /// The real exit, chosen once per level load. Decoys and secret-level pits are already out of
        /// <see cref="pits"/> (see Scan), so this only has to prefer the pit that leads to the next
        /// mission, then an active pit over one whose room hasn't loaded yet. The choice is frozen because
        /// "active" changes as the level plays and every gate's hop count depends on which pit was picked.
        /// </summary>
        private FinalPit ChooseExit()
        {
            if (chosenExit != null) return chosenExit;

            var current = MissionNumber(SceneHelper.CurrentScene);
            FinalPit best = null;
            int bestRank = int.MaxValue;
            foreach (var pit in pits)
            {
                if (pit == null) continue;
                bool successor = current.HasValue && IsSuccessor(current.Value, MissionNumber(pit.targetLevelName));
                int rank = (successor ? 0 : 2) + (pit.gameObject.activeInHierarchy ? 0 : 1);
                if (rank >= bestRank) continue; // ties keep FindObjectsOfType order
                best = pit;
                bestRank = rank;
            }
            chosenExit = best;
            return chosenExit;
        }

        /// <summary>"Level 3-2" to (3, 2); null for anything that doesn't parse (a secret, the menu, ...).</summary>
        private static (int act, int mission)? MissionNumber(string scene)
        {
            if (string.IsNullOrEmpty(scene) || !scene.StartsWith("Level ", StringComparison.Ordinal)) return null;
            var parts = scene.Substring(6).Split('-');
            if (parts.Length != 2) return null;
            if (!int.TryParse(parts[0].Trim(), out var act) || !int.TryParse(parts[1].Trim(), out var mission)) return null;
            return (act, mission);
        }

        /// <summary>The next mission in campaign order: 0-1 to 0-2, 0-5 to 1-1. No level table needed.</summary>
        private static bool IsSuccessor((int act, int mission) current, (int act, int mission)? target)
        {
            if (!target.HasValue) return false;
            var t = target.Value;
            return (t.act == current.act && t.mission == current.mission + 1)
                   || (t.act == current.act + 1 && t.mission == 1);
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
