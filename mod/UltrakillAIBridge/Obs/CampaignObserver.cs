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
    /// the exit, locked doors, arena enemies, the door-graph route ("gates"), the skull altars and carryable
    /// items that lock parts of it, milestone keys and rank thresholds (keys in docs/protocol.md).
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

        /// <summary>
        /// Most altars (ItemPlaceZone with a real accepted item) and carryable items reported per level.
        /// Both are wire-size guards only: the largest measured counts are 7 altars (1-1) and 30
        /// ItemIdentifiers (8-2), and 12 of the 33 shipped levels have neither. A level that exceeded one
        /// would keep the entries nearest the player, the same rule the gates array uses.
        /// </summary>
        private const int MaxAltars = 64;
        private const int MaxItems = 64;
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

        /// <summary>
        /// Heights above the pit that <see cref="UpdateExitGround"/> samples from, nearest first, looking for
        /// the ground the player stands on before dropping in. Sized from the two measured gaps: 61-75 m on
        /// <c>Level 0-2</c> (its completions triggered at y -25 to -27.5 against a pit at y -86.1) and 70 m on
        /// <c>Level 0-3</c>. 0 is first so a pit that already sits on walkable ground is unchanged.
        /// </summary>
        private static readonly float[] ExitGroundHeights = { 0f, 20f, 40f, 60f, 80f, 100f };
        private const float ExitGroundRadius = 30f;         // generous: the ledge can be well off the shaft's axis
        private const float ExitGroundBelowTolerance = 1f;  // a hit this far under the pit still counts as level with it

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
        private bool duplicateDoorKeysWarned;
        private bool duplicateAltarKeysWarned;
        private bool exitTieWarned;

        // One key per Door in the level, gates and altars[].doors both read out of it, so the strings match
        // by construction (rule A2 of the spec). Built in Scan() over every non-template door, not only the
        // gate candidates, because a skull-locked door is usually a one-room door that is no gate at all.
        private readonly Dictionary<Door, string> doorKeys = new Dictionary<Door, string>();
        private readonly List<(Door door, Vector3 pos)> doorKeyOrder = new List<(Door, Vector3)>();

        // Altars (ItemPlaceZone) and carryable items (ItemIdentifier). Everything but the per-step flags in
        // AltarInfo/ItemInfo is decided in Scan().
        private readonly List<AltarInfo> altars = new List<AltarInfo>();
        private readonly List<ItemInfo> items = new List<ItemInfo>();
        private readonly Dictionary<ItemPlaceZone, string> altarKeys = new Dictionary<ItemPlaceZone, string>();
        // Item keys memoized by instance id, so a carried skull keeps its key while it moves (rule A4).
        // Level-load scoped, like hopsByKey: CheckPoint.ResetRoom re-instantiates items on a respawn.
        private readonly Dictionary<int, string> itemKeys = new Dictionary<int, string>();
        private readonly List<int> staleItemKeys = new List<int>();
        private readonly HashSet<int> liveItemIds = new HashSet<int>();

        // Scratch for the room graph, reused across scans.
        private readonly HashSet<string> roomNodes = new HashSet<string>();
        private readonly Dictionary<string, List<string>> roomEdges = new Dictionary<string, List<string>>();
        private readonly Dictionary<string, int> roomHops = new Dictionary<string, int>();
        private readonly Queue<string> bfs = new Queue<string>();
        private readonly HashSet<int> roomIds = new HashSet<int>();
        private readonly HashSet<string> keyScratch = new HashSet<string>();
        private readonly HashSet<Door> gateDoors = new HashSet<Door>();
        private readonly Dictionary<Door, int> gateIndexByDoor = new Dictionary<Door, int>();

        // The FinalPit chosen for this level load, RE-RESOLVED on every Scan (see ResolveExit). The
        // decision is still made once -- rule 4 of the spec: the preference order is time-varying (a pit's
        // room activates mid-run) and every hops value depends on which pit was chosen, so re-running the
        // RULE mid-load could silently renumber the route -- but the reference it hands out is not frozen,
        // because CheckPoint.Start replaces the exit's room with a live clone and banishes the original.
        private FinalPit chosenExit;

        // The frozen decision: what the rule picked, as an identity that survives the room being replaced.
        // Level-load scoped, like hopsByKey.
        private bool exitChosen;
        private string exitTarget;   // the chosen pit's targetLevelName
        private Vector3 exitAnchor;  // where it was when the rule picked it, i.e. the true pit position
        private bool exitReresolveLogged;

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

        // The NavMesh point nearest the chosen FinalPit, i.e. the nearest STANDABLE ground to the exit.
        // Refreshed on the same cadence as the path hint, from the same SampleExit call. Reported as
        // exit.ground_pos, and null when NavMesh has nothing within the widest search radius.
        private Vector3 exitGround;
        private bool exitGroundValid;

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
                exitGroundValid = false;  // another level's standable point must never be reported here
                hopsByKey.Clear();
                itemKeys.Clear();
                chosenExit = null;
                exitChosen = false;
                exitTarget = null;
                exitReresolveLogged = false;
                duplicateDoorKeysWarned = false;
                duplicateAltarKeysWarned = false;
                exitTieWarned = false;
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
            // The live flags of the altars and items, read once here because BuildGates (needs_item),
            // BuildAltars and BuildItems all read the same ones.
            UpdateAltarsAndItems();

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
                    ? new JObject
                    {
                        ["pos"] = ObservationBuilder.Vec(exit.transform.position),
                        ["active"] = exit.gameObject.activeInHierarchy,
                        // The nearest standable ground to the pit, or null. `pos` is the pit's own
                        // transform, which sits far below anything walkable; see UpdateExitGround.
                        ["ground_pos"] = exitGroundValid ? ObservationBuilder.Vec(exitGround) : null,
                    }
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
                ["altars"] = BuildAltars(),
                ["items"] = BuildItems(),
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
                // A pit with no target, one that drops into a secret level ("Level 0-S") and one that drops
                // into a Prime Sanctum ("Level P-1", "Level P-2") is never the mission exit. Measured on
                // 0-2 and 1-1, which each carry a secret pit; 6-2 carries two Prime Sanctum pits and would
                // otherwise tie them against its real Intermission2 exit and pick by FindObjectsOfType
                // order. Campaign-wide only 3-1 and 6-2 ship a "Level P-" pit and no main level's real
                // successor starts with it.
                var target = pit.targetLevelName;
                if (string.IsNullOrEmpty(target)
                    || target.EndsWith("-S", StringComparison.OrdinalIgnoreCase)
                    || target.StartsWith("Level P-", StringComparison.OrdinalIgnoreCase)) continue;
                pits.Add(pit);
            }
            doors.Clear();
            foreach (var door in Object.FindObjectsOfType<Door>(true))
            {
                if (door != null && !IsTemplate(door.transform)) doors.Add(door);
            }
            // Before ScanGates, which asks ExitRoom() (and so ChooseExit()) which room the route starts from.
            ResolveExit();
            ScanDoorKeys();
            ScanAltars(playerPos);
            ScanItems(playerPos);
            checkpointSignature = CheckpointSignature(sm);
            ScanGates(playerPos);
        }

        /// <summary>
        /// One key per Door in the level (rule A2), shared by the gates array and by <c>altars[].doors</c>
        /// so the two string-match by construction. Keys are the same rounded "x,y,z" as everywhere else,
        /// taken from the door's CLOSED position (see <see cref="ClosedPosition"/>).
        ///
        /// Suffixing is deterministic (rule A3): doors are keyed in ascending closed position, falling back
        /// to the instance id so two doors authored at the same point still have a strict order. The old
        /// code suffixed in FindObjectsOfType order, which is not stable across rescans while
        /// <see cref="hopsByKey"/> is keyed on the suffixed string -- so a respawn could hand a gate the
        /// other gate's hops. Measured over all 33 shipped levels, no gate key collides with a non-gate
        /// door key, so widening the namespace to every door leaves every existing gate key unchanged;
        /// 1-2's two doors at "0,20,380" are the only pair this reorders.
        /// </summary>
        private void ScanDoorKeys()
        {
            doorKeys.Clear();
            keyScratch.Clear();
            doorKeyOrder.Clear();
            foreach (var door in doors)
            {
                if (door != null) doorKeyOrder.Add((door, ClosedPosition(door)));
            }
            doorKeyOrder.Sort(CompareDoorPosition);

            int duplicates = 0;
            foreach (var (door, pos) in doorKeyOrder)
            {
                var baseKey = CampaignPatches.Key(pos);
                if (keyScratch.Contains(baseKey)) duplicates++;
                doorKeys[door] = UniqueKey(baseKey, keyScratch);
            }
            if (duplicates > 0 && !duplicateDoorKeysWarned)
            {
                duplicateDoorKeysWarned = true;
                Plugin.Log.LogWarning($"{duplicates} door position key(s) collide in {SceneHelper.CurrentScene}, suffixed with #N");
            }
        }

        private static int CompareDoorPosition((Door door, Vector3 pos) a, (Door door, Vector3 pos) b)
        {
            int cmp = a.pos.x.CompareTo(b.pos.x);
            if (cmp != 0) return cmp;
            cmp = a.pos.y.CompareTo(b.pos.y);
            if (cmp != 0) return cmp;
            cmp = a.pos.z.CompareTo(b.pos.z);
            if (cmp != 0) return cmp;
            return a.door.GetInstanceID().CompareTo(b.door.GetInstanceID());
        }

        /// <summary><paramref name="baseKey"/>, or it with "#2", "#3", ... until it is not already in <paramref name="used"/>.</summary>
        private static string UniqueKey(string baseKey, HashSet<string> used)
        {
            if (used.Add(baseKey)) return baseKey;
            for (int n = 2; ; n++)
            {
                var key = baseKey + "#" + n.ToString(System.Globalization.CultureInfo.InvariantCulture);
                if (used.Add(key)) return key;
            }
        }

        /// <summary>Inactive GameObjects on a transform's own chain, itself included (rule A6).</summary>
        private static int InactiveAncestors(Transform t)
        {
            int inactive = 0;
            for (; t != null; t = t.parent)
            {
                if (!t.gameObject.activeSelf) inactive++;
            }
            return inactive;
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
            /// <summary>Phase 2: a door only an altar opens, which the two-room test rejects (see ScanGates).</summary>
            public bool AltarOnly;
            /// <summary>Indices into <see cref="altars"/> of the altars that OPEN this door; null when none does.</summary>
            public List<int> Altars;
        }

        /// <summary>
        /// Rebuilds the route: which doors are gates, the room graph they form, and each gate's hop count
        /// from the room holding the exit. Runs inside <see cref="Scan"/>, i.e. at most once every
        /// <see cref="RescanEvery"/> builds, so a step never walks the scene or the graph again.
        ///
        /// Phase 1: a door is a gate when its activatedRooms holds at least two distinct GameObjects; the
        /// rooms of those doors (and nothing else) are the graph's nodes, identified by their rounded world
        /// position rather than by reference, because CheckPoint.ResetRoom destroys and re-instantiates a
        /// room on every respawn. Two rooms are adjacent when one gate lists both, and a BFS from the exit's
        /// own room gives every room its hop count; a gate takes the lowest hop count among its rooms.
        ///
        /// Phase 2 then appends the doors an altar OPENS that phase 1 rejected, marked "altar_only". A
        /// skull-locked door is overwhelmingly a one-room streaming door (measured: 6 of 28 altar-driven
        /// forward doors campaign-wide are gates today), so without this a skull leg is invisible to the
        /// route. Phase 2 adds no node and no edge, so no existing hop count can change; the appended door
        /// reads its hops off the rooms it does list, which leaves it null everywhere the rooms are outside
        /// the graph (measured, it gives a usable hops only on 1-1, 5-3 and 8-1).
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
            gateDoors.Clear();

            foreach (var door in doors)
            {
                if (door == null || door.activatedRooms == null) continue;
                var rooms = RoomNodes(door, out var distinctRooms);
                if (distinctRooms < 2) continue;
                if (!doorKeys.TryGetValue(door, out var key)) continue;
                gateDoors.Add(door);

                var pos = ClosedPosition(door);
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

            // Phase 2: doors an altar opens that phase 1 rejected. A dead-branch altar (rule A6: more than
            // one inactive GameObject on its own chain) can never activate or fill, so it drives nothing;
            // measured, no door in the campaign is driven only by dead zones, so skipping them never loses
            // a real lock.
            foreach (var altar in altars)
            {
                if (altar.Zone == null || altar.DoorObjects == null) continue;
                if (IsDeadTwin(altar)) continue;
                foreach (var door in altar.DoorObjects)
                {
                    if (door == null || !doorKeys.TryGetValue(door, out var key)) continue;
                    if (!gateDoors.Add(door)) continue;
                    var pos = ClosedPosition(door);
                    gates.Add(new Gate
                    {
                        Door = door,
                        Controllers = FindControllers(door),
                        Key = key,
                        Pos = pos,
                        Rooms = RoomNodes(door, out _),
                        Dist = Vector3.Distance(playerPos, pos),
                        AltarOnly = true,
                    });
                }
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
            LinkGateAltars();
        }

        /// <summary>
        /// The distinct room nodes a door's activatedRooms names, identified by rounded world position, or
        /// null when it names none. <paramref name="distinctRooms"/> counts distinct room GameObjects, which
        /// is what decides whether the door is a phase-1 gate: two rooms can round to the same node key.
        /// </summary>
        private List<string> RoomNodes(Door door, out int distinctRooms)
        {
            roomIds.Clear();
            List<string> rooms = null;
            var activated = door.activatedRooms;
            if (activated != null)
            {
                foreach (var room in activated)
                {
                    if (room == null || !roomIds.Add(room.GetInstanceID())) continue;
                    if (rooms == null) rooms = new List<string>(activated.Length);
                    var node = CampaignPatches.Key(room.transform.position);
                    if (!rooms.Contains(node)) rooms.Add(node);
                }
            }
            distinctRooms = roomIds.Count;
            return rooms;
        }

        /// <summary>
        /// Records, per gate, which altars OPEN it, so a step only has to read those altars' "filled" flag
        /// to answer "needs_item". reverseDoors are deliberately not wired: placing the item CLOSES those,
        /// so treating one as a lock would send the agent to fetch a skull that shuts the route.
        /// </summary>
        private void LinkGateAltars()
        {
            gateIndexByDoor.Clear();
            for (int i = 0; i < gates.Count; i++)
            {
                gates[i].Altars = null;
                if (gates[i].Door != null) gateIndexByDoor[gates[i].Door] = i;
            }
            for (int a = 0; a < altars.Count; a++)
            {
                var doorObjects = altars[a].DoorObjects;
                if (doorObjects == null) continue;
                foreach (var door in doorObjects)
                {
                    if (door == null || !gateIndexByDoor.TryGetValue(door, out var index)) continue;
                    var gate = gates[index];
                    if (gate.Altars == null) gate.Altars = new List<int>(1);
                    if (!gate.Altars.Contains(a)) gate.Altars.Add(a);
                }
            }
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
                var needs = NeedsItem(gate);
                var obj = new JObject
                {
                    ["key"] = gate.Key,
                    ["pos"] = ObservationBuilder.Vec(gate.Pos),
                    ["hops"] = gate.Hops.HasValue ? new JValue(gate.Hops.Value) : JValue.CreateNull(),
                    ["open"] = alive && door.open,
                    ["locked"] = alive && door.locked,
                    ["active"] = alive && door.gameObject.activeInHierarchy,
                    ["controller_active"] = alive && ControllerActive(gate.Controllers),
                    ["needs_item"] = needs != null ? new JValue(needs) : JValue.CreateNull(),
                };
                if (gate.AltarOnly) obj["altar_only"] = true;
                arr.Add(obj);
            }
            return arr;
        }

        /// <summary>
        /// The item type an UNFILLED, non-dead altar wants before this door will open, or null. With several
        /// such altars the lowest type name wins, so the answer is deterministic.
        ///
        /// A dead twin is excluded because it can never fill: counting it would leave the lock set after the
        /// live twin has already been filled, and the agent would be sent to punch the skull back out of the
        /// altar it just filled -- which ItemPlaceZone.CheckItem answers by closing the door again. Measured,
        /// 20 of the campaign's 104 functional zones are such dead twins. See <see cref="IsDeadTwin"/>.
        /// </summary>
        private string NeedsItem(Gate gate)
        {
            if (gate.Altars == null) return null;
            string needs = null;
            foreach (var index in gate.Altars)
            {
                var altar = altars[index];
                if (altar.Zone == null || altar.Filled || IsDeadTwin(altar)) continue;
                if (needs == null || string.CompareOrdinal(altar.Item, needs) < 0) needs = altar.Item;
            }
            return needs;
        }

        /// <summary>
        /// Whether an altar is the dead half of a duplicated ItemPlaceZone pair, judged RELATIVE to its twins:
        /// a zone is dead when another zone with the same item type, the same door set and the same rounded
        /// position reports STRICTLY FEWER inactive ancestors.
        ///
        /// The test used to be the absolute <c>InactiveAncestors &gt; 1</c>, which is the M14 bug. That number
        /// is not a property of the zone: it is the zone's own chain PLUS however much of the room above it
        /// happens to be switched off, so it shifts when the room lights. Measured in game on Level 1-1, the
        /// live altar 81,-4,251 and its twin read 1 and 2 on a fresh load and 0 and 1 after the checkpoint
        /// respawn switched that room on -- so the twin stopped being filtered exactly when the player
        /// arrived, and the gate kept needs_item set however often the puzzle was solved.
        ///
        /// A shared room contributes the same count to both halves of a pair, so the relative test cannot be
        /// shifted by one lighting up. The minimum of each group always survives, so a group can never be
        /// filtered away entirely and a real lock can never be lost. Re-validated offline against all 21 altar
        /// levels: it drops the same 20 of 104 zones the absolute rule did, loses no lock anywhere, and takes
        /// the doors left stuck after a solved puzzle from 11 to 0 once a room lights.
        ///
        /// Position is part of the identity because twins are co-located (~0.2 m apart, which is why the keys
        /// need the #N suffix at all). Without it the rule would also drop five zones that are not twins but
        /// two separate altars of one item type that drive no door: 4-2 (x2), 4-3, 5-3 and 7-1.
        /// </summary>
        private bool IsDeadTwin(AltarInfo altar)
        {
            foreach (var other in altars)
            {
                if (ReferenceEquals(other, altar) || other.Zone == null) continue;
                if (other.InactiveAncestors >= altar.InactiveAncestors) continue;
                if (!string.Equals(other.Item, altar.Item, System.StringComparison.Ordinal)) continue;
                if (!string.Equals(other.PosKey, altar.PosKey, System.StringComparison.Ordinal)) continue;
                if (SameDoors(other.Doors, altar.Doors)) return true;
            }
            return false;
        }

        /// <summary>Whether two altars drive the same set of doors (both lists are 0-2 entries long).</summary>
        private static bool SameDoors(List<DoorRef> a, List<DoorRef> b)
        {
            int na = a != null ? a.Count : 0, nb = b != null ? b.Count : 0;
            if (na != nb) return false;
            for (int i = 0; i < na; i++)
            {
                bool found = false;
                for (int j = 0; j < nb && !found; j++)
                {
                    found = string.Equals(a[i].Key, b[j].Key, System.StringComparison.Ordinal);
                }
                if (!found) return false;
            }
            return true;
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
        /// One altar: an ItemPlaceZone that accepts a real item type. Everything but the three per-step
        /// flags at the bottom is decided in <see cref="ScanAltars"/>.
        /// </summary>
        private sealed class AltarInfo
        {
            public ItemPlaceZone Zone;
            public string Key;
            /// <summary>The rounded position key WITHOUT the #N suffix, i.e. what twins share (IsDeadTwin).</summary>
            public string PosKey;
            public Vector3 Pos;
            /// <summary>Where a punch must be aimed to fill this zone. See <see cref="AimPoint"/>.</summary>
            public Vector3 AimPos;
            /// <summary>ItemType name: SkullBlue, SkullRed, SkullGreen, Readable, Torch, Soap, CustomKey1..3.</summary>
            public string Item;
            /// <summary>The doors this altar OPENS, as wire-ready key/pos pairs.</summary>
            public List<DoorRef> Doors;
            /// <summary>The doors this altar CLOSES.</summary>
            public List<DoorRef> ReverseDoors;
            /// <summary>The same forward doors as objects, for gate phase 2 and the gate wiring.</summary>
            public List<Door> DoorObjects;
            public float Dist;

            public bool Filled;
            public bool Active;
            public int InactiveAncestors;
        }

        /// <summary>One carryable item, an ItemIdentifier with a real item type. Positions move, so most of this is per step.</summary>
        private sealed class ItemInfo
        {
            public ItemIdentifier Id;
            public string Key;
            public string Item;
            public float Dist;

            public Vector3 Pos;
            public bool Held;
            public bool Placed;
            public string PlacedIn;
            public bool Active;
            public bool ActiveSelf;
            public int InactiveAncestors;
        }

        /// <summary>A door named by an altar, as it goes on the wire.</summary>
        private sealed class DoorRef
        {
            public string Key;
            public Vector3 Pos;
        }

        /// <summary>
        /// Every ItemPlaceZone in the level that accepts a real item type, room templates excluded. Zones in
        /// rooms that have not streamed in yet are included (FindObjectsOfType with inactive), because the
        /// point of the block is to give the agent a heading before it gets there.
        ///
        /// Dead branches are reported too, with their inactive_ancestors, so Python can apply rule A6 with
        /// the same numbers the mod used; only "needs_item" filters them mod-side.
        /// </summary>
        private void ScanAltars(Vector3 playerPos)
        {
            altars.Clear();
            altarKeys.Clear();
            foreach (var zone in Object.FindObjectsOfType<ItemPlaceZone>(true))
            {
                if (zone == null || zone.acceptedItemType == ItemType.None || IsTemplate(zone.transform)) continue;
                var pos = zone.transform.position;
                altars.Add(new AltarInfo
                {
                    Zone = zone,
                    Pos = pos,
                    AimPos = AimPoint(zone),
                    Item = zone.acceptedItemType.ToString(),
                    Doors = DoorRefs(zone.doors),
                    ReverseDoors = DoorRefs(zone.reverseDoors),
                    DoorObjects = DoorObjects(zone.doors),
                    // Read here as well as in UpdateAltarsAndItems, because ScanGates' phase 2 runs before the
                    // first UpdateAltarsAndItems of a scan and its dead-twin test needs the whole set.
                    InactiveAncestors = InactiveAncestors(zone.transform),
                    Dist = Vector3.Distance(playerPos, pos),
                });
            }
            if (altars.Count > MaxAltars)
            {
                altars.Sort((a, b) => a.Dist.CompareTo(b.Dist));
                altars.RemoveRange(MaxAltars, altars.Count - MaxAltars);
            }

            // Keys in ascending position, #N on collision, the same deterministic rule the doors use:
            // several levels ship coincident duplicate zones ~0.2 m apart (1-1 has three such pairs).
            altars.Sort(CompareAltarPosition);
            keyScratch.Clear();
            int duplicates = 0;
            foreach (var altar in altars)
            {
                var baseKey = CampaignPatches.Key(altar.Pos);
                if (keyScratch.Contains(baseKey)) duplicates++;
                altar.PosKey = baseKey;
                altar.Key = UniqueKey(baseKey, keyScratch);
                altarKeys[altar.Zone] = altar.Key;
            }
            if (duplicates > 0 && !duplicateAltarKeysWarned)
            {
                duplicateAltarKeysWarned = true;
                Plugin.Log.LogWarning($"{duplicates} altar position key(s) collide in {SceneHelper.CurrentScene}, suffixed with #N");
            }
            altars.Sort((a, b) => string.CompareOrdinal(a.Key, b.Key));
        }

        /// <summary>
        /// Where a punch has to be aimed to fill a zone: the centre of the zone's OWN collider.
        ///
        /// Punch.AltHit (decompiled/Punch.cs:1335) only places when the ray hits the very GameObject that
        /// carries the ItemPlaceZone -- it calls target.GetComponents&lt;ItemPlaceZone&gt;() on the transform
        /// the raycast returned -- and ItemPlaceZone.Start reads its own GetComponent&lt;Collider&gt;(), so
        /// that collider is the thing to hit. It is NOT centred on the transform: every one of the campaign's
        /// 104 zones is the same prefab, a trigger BoxCollider of local size (2.2, 3.5, 2.2) whose local
        /// centre is (0, -1.25, 0), on a transform scaled (0.9, 0.8, 0.8). So the box sits 1 m below
        /// transform.position (103 of 104; the one exception is 0.625 m) and spans pos.y-2.4 .. pos.y+0.4:
        /// the position the block reports is only 0.4 m under the lid, with no margin for aim error, while
        /// the centre has 1.4 m of it. Measured in game on 1-1, aiming at the reported position does not
        /// place and aiming ~1.25 m below it does.
        ///
        /// TransformPoint rather than Collider.bounds, because a zone in a room that has not streamed in yet
        /// is inactive and its bounds are not meaningful, while the transform maths always is. bounds is the
        /// fallback for a collider type with no centre of its own (no shipped zone uses one).
        /// </summary>
        private static Vector3 AimPoint(ItemPlaceZone zone)
        {
            var t = zone.transform;
            var box = zone.GetComponent<BoxCollider>();
            if (box != null) return t.TransformPoint(box.center);
            var sphere = zone.GetComponent<SphereCollider>();
            if (sphere != null) return t.TransformPoint(sphere.center);
            var capsule = zone.GetComponent<CapsuleCollider>();
            if (capsule != null) return t.TransformPoint(capsule.center);
            var col = zone.GetComponent<Collider>();
            if (col != null && zone.gameObject.activeInHierarchy) return col.bounds.center;
            return t.position;
        }

        private static int CompareAltarPosition(AltarInfo a, AltarInfo b)
        {
            int cmp = a.Pos.x.CompareTo(b.Pos.x);
            if (cmp != 0) return cmp;
            cmp = a.Pos.y.CompareTo(b.Pos.y);
            if (cmp != 0) return cmp;
            cmp = a.Pos.z.CompareTo(b.Pos.z);
            if (cmp != 0) return cmp;
            return a.Zone.GetInstanceID().CompareTo(b.Zone.GetInstanceID());
        }

        private List<DoorRef> DoorRefs(Door[] doorArray)
        {
            if (doorArray == null) return null;
            List<DoorRef> refs = null;
            foreach (var door in doorArray)
            {
                if (door == null || !doorKeys.TryGetValue(door, out var key)) continue;
                if (refs == null) refs = new List<DoorRef>(doorArray.Length);
                refs.Add(new DoorRef { Key = key, Pos = ClosedPosition(door) });
            }
            return refs;
        }

        private List<Door> DoorObjects(Door[] doorArray)
        {
            if (doorArray == null) return null;
            List<Door> list = null;
            foreach (var door in doorArray)
            {
                if (door == null || !doorKeys.ContainsKey(door)) continue;
                if (list == null) list = new List<Door>(doorArray.Length);
                list.Add(door);
            }
            return list;
        }

        /// <summary>
        /// Every ItemIdentifier in the level with a real item type, room templates excluded. infiniteSource
        /// items are skipped: Punch.AltHit mints a fresh ItemIdentifier per punch for one of those, which
        /// would be an unbounded key stream. (Measured: no shipped level has one, so this is a guard.)
        ///
        /// Keys are memoized by instance id (rule A4), because a carried skull moves every step while the
        /// scan only runs every RescanEvery builds. Python does not depend on a key surviving a respawn;
        /// the only requirement is that keys are unique within a step.
        /// </summary>
        private void ScanItems(Vector3 playerPos)
        {
            items.Clear();
            foreach (var identifier in Object.FindObjectsOfType<ItemIdentifier>(true))
            {
                if (identifier == null || identifier.itemType == ItemType.None || identifier.infiniteSource) continue;
                if (IsTemplate(identifier.transform)) continue;
                var pos = identifier.transform.position;
                items.Add(new ItemInfo
                {
                    Id = identifier,
                    Item = identifier.itemType.ToString(),
                    Pos = pos,
                    Dist = Vector3.Distance(playerPos, pos),
                });
            }
            if (items.Count > MaxItems)
            {
                items.Sort((a, b) => a.Dist.CompareTo(b.Dist));
                items.RemoveRange(MaxItems, items.Count - MaxItems);
            }

            keyScratch.Clear();
            liveItemIds.Clear();
            foreach (var item in items)
            {
                int id = item.Id.GetInstanceID();
                liveItemIds.Add(id);
                if (itemKeys.TryGetValue(id, out var known) && keyScratch.Add(known)) item.Key = known;
            }
            items.Sort(CompareItemPosition);
            foreach (var item in items)
            {
                if (item.Key != null) continue;
                item.Key = UniqueKey(CampaignPatches.Key(item.Pos), keyScratch);
                itemKeys[item.Id.GetInstanceID()] = item.Key;
            }

            // Drop memoized keys of instances that no longer exist (a respawn re-instantiates rooms), so the
            // table cannot grow without bound over a level load.
            staleItemKeys.Clear();
            foreach (var pair in itemKeys)
            {
                if (!liveItemIds.Contains(pair.Key)) staleItemKeys.Add(pair.Key);
            }
            foreach (var id in staleItemKeys)
            {
                itemKeys.Remove(id);
            }

            items.Sort((a, b) => string.CompareOrdinal(a.Key, b.Key));
        }

        private static int CompareItemPosition(ItemInfo a, ItemInfo b)
        {
            int cmp = a.Pos.x.CompareTo(b.Pos.x);
            if (cmp != 0) return cmp;
            cmp = a.Pos.y.CompareTo(b.Pos.y);
            if (cmp != 0) return cmp;
            cmp = a.Pos.z.CompareTo(b.Pos.z);
            if (cmp != 0) return cmp;
            return a.Id.GetInstanceID().CompareTo(b.Id.GetInstanceID());
        }

        /// <summary>
        /// The live flags of every altar and item, read once per step.
        ///
        /// "filled" is computed exactly as ItemPlaceZone.CheckItem does (decompiled/ItemPlaceZone.cs:160):
        /// GetComponentInChildren&lt;ItemIdentifier&gt;() WITHOUT includeInactive, then the type test. With
        /// includeInactive every "Altar (... Skull) Variant" would report as filled at load, because it
        /// carries a disabled decoration skull. The consequence to accept is that an altar in a room that
        /// has not streamed in reads filled: false (and active: false) -- conservative for routing, and it
        /// also makes this cheap, since GetComponentInChildren on an inactive object returns immediately.
        /// </summary>
        private void UpdateAltarsAndItems()
        {
            foreach (var altar in altars)
            {
                var zone = altar.Zone;
                if (zone == null)
                {
                    altar.Filled = false;
                    altar.Active = false;
                    altar.InactiveAncestors = 0;
                    continue;
                }
                altar.Active = zone.gameObject.activeInHierarchy;
                altar.InactiveAncestors = InactiveAncestors(zone.transform);
                var placed = zone.GetComponentInChildren<ItemIdentifier>();
                altar.Filled = placed != null && placed.itemType == zone.acceptedItemType;
            }

            foreach (var item in items)
            {
                var identifier = item.Id;
                if (identifier == null)
                {
                    item.Held = false;
                    item.Placed = false;
                    item.PlacedIn = null;
                    item.Active = false;
                    item.ActiveSelf = false;
                    item.InactiveAncestors = 0;
                    continue;
                }
                item.Pos = identifier.transform.position;
                item.Held = identifier.pickedUp;
                // ItemPlaceZone.Start is what sets ipz, and Start has not run while the room is off, so a
                // pedestal skull would read placed: false at load without the parent search (rule A5).
                var zone = identifier.ipz != null ? identifier.ipz : identifier.GetComponentInParent<ItemPlaceZone>(true);
                item.Placed = zone != null;
                item.PlacedIn = zone != null && altarKeys.TryGetValue(zone, out var key) ? key : null;
                item.Active = identifier.gameObject.activeInHierarchy;
                item.ActiveSelf = identifier.gameObject.activeSelf;
                item.InactiveAncestors = InactiveAncestors(identifier.transform);
            }
        }

        private JArray BuildAltars()
        {
            var arr = new JArray();
            foreach (var altar in altars)
            {
                arr.Add(new JObject
                {
                    ["key"] = altar.Key,
                    ["pos"] = ObservationBuilder.Vec(altar.Pos),
                    ["aim_pos"] = ObservationBuilder.Vec(altar.AimPos),
                    ["item"] = altar.Item,
                    ["filled"] = altar.Filled,
                    ["active"] = altar.Active,
                    ["inactive_ancestors"] = altar.InactiveAncestors,
                    ["doors"] = BuildDoorRefs(altar.Doors),
                    ["reverse_doors"] = BuildDoorRefs(altar.ReverseDoors),
                });
            }
            return arr;
        }

        private JArray BuildItems()
        {
            var arr = new JArray();
            foreach (var item in items)
            {
                arr.Add(new JObject
                {
                    ["key"] = item.Key,
                    ["pos"] = ObservationBuilder.Vec(item.Pos),
                    ["item"] = item.Item,
                    ["held"] = item.Held,
                    ["placed"] = item.Placed,
                    ["placed_in"] = item.PlacedIn != null ? new JValue(item.PlacedIn) : JValue.CreateNull(),
                    ["active"] = item.Active,
                    ["active_self"] = item.ActiveSelf,
                    ["inactive_ancestors"] = item.InactiveAncestors,
                });
            }
            return arr;
        }

        private static JArray BuildDoorRefs(List<DoorRef> refs)
        {
            var arr = new JArray();
            if (refs == null) return arr;
            foreach (var reference in refs)
            {
                arr.Add(new JObject
                {
                    ["key"] = reference.Key,
                    ["pos"] = ObservationBuilder.Vec(reference.Pos),
                });
            }
            return arr;
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
        /// The exit for this step: the frozen decision, re-resolved onto whichever live <see cref="FinalPit"/>
        /// now carries it. Cheap -- <see cref="ResolveExit"/> does the work once per <see cref="Scan"/>.
        /// </summary>
        private FinalPit ChooseExit()
        {
            if (!exitChosen) ResolveExit();
            return chosenExit;
        }

        /// <summary>
        /// Re-resolves <see cref="chosenExit"/> against the live scene, running the choice RULE only the
        /// first time it succeeds in a level load.
        ///
        /// **Why a frozen reference was wrong.** <c>CheckPoint.Start</c> (decompiled/CheckPoint.cs:120-132)
        /// clones every room the checkpoint owns, activates the clone at the room's own position, and moves
        /// the ORIGINAL to <c>x + 10000</c>; <c>ResetRoom</c> (:621) does it again on every respawn. A
        /// checkpoint's Start only runs once its own room is live, so on a level where the exit's room belongs
        /// to a later checkpoint the first scan sees the original -- picks it, and used to keep that reference
        /// for the whole load. The moment that checkpoint activated, the reported exit jumped 10,000 m along
        /// x while the real trigger stayed put. Measured on <c>Level 0-2</c>: the exit read
        /// (9801, -86.1, 277) once checkpoint <c>-55,-11,277</c> activated, against a true pit at
        /// (-199, -86.1, 277). Python's <c>ExitGuard</c> is the stopgap that rejects the jump; this is the fix.
        ///
        /// **What stays frozen.** The rule's verdict, not the object: the pit's <c>targetLevelName</c> and the
        /// position it held when the rule picked it. Re-running the rule itself would be unsafe, because its
        /// rank includes <c>activeInHierarchy</c>, which changes as rooms stream in, and every gate's hop
        /// count is measured from the chosen pit's room.
        ///
        /// **How the live twin is told from the banished one.** By distance to the anchor. The clone is
        /// instantiated at the original's exact position, so it sits 0 m away; the banished original sits
        /// 10,000 m away, a gap no legitimate move approaches. (It is usually gone from <see cref="pits"/>
        /// anyway once its room is in some checkpoint's <c>defaultRooms</c> and <see cref="IsTemplate"/>
        /// catches it -- but that is a consequence of the same Start, so it must not be the only defence.)
        /// A scan that finds no match at all keeps the last resolved pit rather than reporting no exit: a
        /// respawn re-instantiates the room, and a scan landing inside that window is a blink, not a change.
        /// </summary>
        private void ResolveExit()
        {
            if (!exitChosen)
            {
                var picked = ChooseExitByRule();
                if (picked == null) return;  // nothing to freeze yet; try again on the next scan
                exitChosen = true;
                exitTarget = picked.targetLevelName;
                exitAnchor = picked.transform.position;
                chosenExit = picked;
                return;
            }

            FinalPit best = null;
            float bestDist = float.MaxValue;
            foreach (var pit in pits)
            {
                if (pit == null) continue;
                if (!string.Equals(pit.targetLevelName, exitTarget, StringComparison.Ordinal)) continue;
                var dist = (pit.transform.position - exitAnchor).sqrMagnitude;
                if (dist >= bestDist) continue;  // ties keep FindObjectsOfType order, as the rule does
                best = pit;
                bestDist = dist;
            }
            if (best == null) return;  // mid-respawn blink: keep the pit we already have
            if (!ReferenceEquals(best, chosenExit) && !exitReresolveLogged)
            {
                exitReresolveLogged = true;
                var was = chosenExit != null ? CampaignPatches.Key(chosenExit.transform.position) : "none";
                Plugin.Log.LogInfo(
                    $"Exit re-resolved in {SceneHelper.CurrentScene}: {was} -> "
                    + $"{CampaignPatches.Key(best.transform.position)} (target '{exitTarget}')");
            }
            chosenExit = best;
        }

        /// <summary>
        /// The choice rule, run once per level load by <see cref="ResolveExit"/>. Decoys, secret-level pits
        /// and Prime Sanctum pits are already out of <see cref="pits"/> (see Scan), so this only has to
        /// prefer the pit that leads onward, then an active pit over one whose room hasn't loaded yet.
        ///
        /// An "Intermission*" target counts as leading onward: an act finale's real exit drops into an
        /// intermission, not into the next mission, and MissionNumber cannot parse it, so IsSuccessor has
        /// no string to test and the rule cannot live inside it. Measured, 3-2 and 6-2 are the only levels
        /// that ship such a pit.
        ///
        /// A tie at the best rank is logged once per level load. Two of the four exit-ambiguous levels stay
        /// tied after this (3-2's two Intermission1 pits sit at the identical position, 2-4's two Level 3-1
        /// pits 1.4 m apart in the same room, so both are benign), and 8-4's EarlyAccessEnd pit is picked by
        /// uniqueness rather than by the successor test -- exactly the shape of level where a future
        /// duplicate would go unnoticed.
        /// </summary>
        private FinalPit ChooseExitByRule()
        {
            var current = MissionNumber(SceneHelper.CurrentScene);
            FinalPit best = null;
            int bestRank = int.MaxValue;
            int tied = 0;
            foreach (var pit in pits)
            {
                if (pit == null) continue;
                var target = pit.targetLevelName;
                bool successor = (target != null && target.StartsWith("Intermission", StringComparison.OrdinalIgnoreCase))
                                 || (current.HasValue && IsSuccessor(current.Value, MissionNumber(target)));
                int rank = (successor ? 0 : 2) + (pit.gameObject.activeInHierarchy ? 0 : 1);
                if (rank > bestRank) continue;
                if (rank == bestRank)
                {
                    tied++; // ties keep FindObjectsOfType order
                    continue;
                }
                best = pit;
                bestRank = rank;
                tied = 0;
            }
            if (tied > 0 && best != null && !exitTieWarned)
            {
                exitTieWarned = true;
                Plugin.Log.LogWarning(
                    $"{tied + 1} FinalPit candidates tie at rank {bestRank} in {SceneHelper.CurrentScene}; "
                    + $"keeping the one to '{best.targetLevelName}' at {CampaignPatches.Key(best.transform.position)}");
            }
            return best;
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
            // Before the player-end snap, deliberately: the two are independent, and on Level 0-1 the
            // player's own 25 m snap fails at the level's spawn point. Sampling the exit end first means
            // exit.ground_pos is still reported there, where the path hint is not.
            UpdateExitGround(exit);
            if (exit == null) return;
            if (!NavMesh.SamplePosition(playerPos, out var from, PlayerSnapDistance, NavMesh.AllAreas)) return;
            if (!exitGroundValid) return;
            var to = exitGround;

            if (navPath == null) navPath = new NavMeshPath();
            if (!NavMesh.CalculatePath(from.position, to, NavMesh.AllAreas, navPath)) return;
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
        /// <summary>
        /// Refreshes <see cref="exitGround"/>: standable ground ABOVE the chosen pit, which is where the
        /// player is when they drop into it. A <c>FinalPit</c>'s own transform sits INSIDE the drop it
        /// triggers, far below anything you can walk on -- measured 61-75 m below the floor on
        /// <c>Level 0-2</c> and 70 m on <c>Level 0-3</c> -- so a target built from <c>exit.pos</c> points the
        /// agent down a killing fall, and a distance to it can never reach zero.
        ///
        /// It deliberately does NOT reuse <see cref="SampleExit"/>, and the difference is the whole point.
        /// That method wants the nearest mesh in ANY direction, which is right for the path hint's endpoint;
        /// measured on <c>Level 0-2</c> it returns <c>(-199, -133.5, 277)</c>, another 47.4 m DOWN the pit
        /// shaft -- a target 47 m worse than the pit itself. The ground the player actually completes from is
        /// 60 m the other way: 0-2's two real completions triggered at y -25 to -27.5.
        ///
        /// So this searches UPWARD, in <see cref="ExitGroundHeights"/> steps from the pit, and accepts only a
        /// hit at or above the pit's own y. Nothing found means <c>ground_pos</c> is null and every consumer
        /// falls back to <c>exit.pos</c>, which is exactly today's behaviour -- so the worst case of this
        /// whole mechanism is a no-op, never a regression. Bounded to
        /// <c>ExitGroundHeights.Length</c> SamplePosition calls on the <see cref="PathEvery"/> cadence.
        /// </summary>
        private void UpdateExitGround(FinalPit exit)
        {
            exitGroundValid = false;
            if (exit == null) return;
            var pit = exit.transform.position;
            foreach (var height in ExitGroundHeights)
            {
                var from = new Vector3(pit.x, pit.y + height, pit.z);
                if (!NavMesh.SamplePosition(from, out var hit, ExitGroundRadius, NavMesh.AllAreas)) continue;
                // Below the pit is the shaft, not the ledge over it. Keep looking.
                if (hit.position.y < pit.y - ExitGroundBelowTolerance) continue;
                exitGround = hit.position;
                exitGroundValid = true;
                return;
            }
        }

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
