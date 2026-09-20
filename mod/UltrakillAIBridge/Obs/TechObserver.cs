using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UltrakillAIBridge.Env;
using UnityEngine;
using UnityEngine.InputSystem.LowLevel;

namespace UltrakillAIBridge.Obs
{
    /// <summary>
    /// The movement / weapon / own-projectile state the speedrun techniques need, as three optional obs
    /// blocks. Every block is OFF by default and switched on individually through `config`, so a client that
    /// knows nothing about mod 0.8 receives exactly the 0.7.2 observation.
    ///
    /// - **A `move_tech`** — the slam family, the wall-jump budget, dash i-frames, the slide state. This is
    ///   the block the agent is most blind without: it is mid-slam every time it lands hard and cannot see it,
    ///   and `wall_available` is the one field the 16 horizontal rays cannot express, because a `WallCheck` is
    ///   a component, not a distance.
    /// - **B `weapon_tech`** — which variant is held and whether it can actually fire. `gun_ready` is the
    ///   field that shows a weapon stuck in its draw animation.
    /// - **C `projectiles`** — the player's OWN live coins, rockets, grenades and cannonballs. Without it
    ///   every coin, core and rocket technique is invisible to the policy at the exact moment it must act.
    ///
    /// Cost: A and B are field reads. C walks three lists the game already maintains and keeps the nearest
    /// few, which is why it is separately switchable.
    /// </summary>
    internal sealed class TechObserver
    {
        internal bool EmitMove;
        internal bool EmitWeapon;
        internal bool EmitProjectiles;
        internal int MaxProjectiles = 3;

        /// <summary>First time each tracked object was seen, so `age` is real rather than guessed.</summary>
        private readonly Dictionary<int, float> firstSeen = new Dictionary<int, float>();
        private readonly HashSet<int> seenThisBuild = new HashSet<int>();
        private readonly List<int> expired = new List<int>();
        private readonly List<(JObject obj, float dist)> nearest = new List<(JObject, float)>();

        // ---- Block A ------------------------------------------------------------------------------------

        internal JObject BuildMove(NewMovement nm)
        {
            var gc = nm.gc;
            var wc = PlayerFields.WallChecks(nm);
            float boostLeft = PlayerFields.BoostLeft(nm);
            var afterSlide = PlayerFields.VelocityAfterSlide(nm);

            return new JObject
            {
                // The slam family. `slam_force` is what turns a landing into a 12.5x bounce (>= 5.5) and what
                // FixedUpdate copies into preSlideSpeed on a heavy-fall landing for a 3x slide.
                ["heavy_fall"] = gc != null && gc.heavyFall,
                ["slam_force"] = nm.slamForce,
                ["slam_storage"] = PlayerFields.SlamStorage(nm),
                ["slam_cooldown"] = nm.slamCooldown,
                // Any of the three chances positive means Jump() takes its bounce branch instead of the plain
                // 2.6x one. ~406 ms wide, i.e. six decisions: expressible today, and measured anti-learned.
                ["bounce_window"] = gc != null && (gc.superJumpChance > 0f || gc.bounceChance > 0f || gc.extraJumpChance > 0f),
                ["super_jump_chance"] = gc != null ? gc.superJumpChance : 0f,
                ["bounce_chance"] = gc != null ? gc.bounceChance : 0f,
                ["extra_jump_chance"] = gc != null ? gc.extraJumpChance : 0f,

                ["coyote"] = gc != null ? (float)gc.sinceLastGrounded : 999f,
                ["can_jump"] = gc != null && gc.canJump,
                ["wall_jumps"] = nm.currentWallJumps,
                ["wall_available"] = wc != null && wc.TryGetActiveInstance(out _),
                ["wall_touching"] = wc != null && wc.OnWall(),
                ["enemy_cols"] = wc != null && wc.CheckForEnemyCols(),

                ["boost"] = nm.boost,
                ["boost_left"] = boostLeft,          // dash i-frames left; > 0 also means layer 15
                ["dash_storage"] = PlayerFields.DashStorage(nm),
                ["invincible"] = nm.gameObject.layer == 15 && boostLeft > 0f,

                ["pre_slide_speed"] = nm.preSlideSpeed,
                ["jump_cooldown"] = PlayerFields.JumpCooldown(nm),
                ["jumping"] = nm.jumping,
                ["falling"] = nm.falling,
                // Real seconds since the last slide release, on the clock TrySSJ and WallJump compare against.
                ["slide_since"] = InputState.currentTime - nm.slideTimestamp,
                ["slide_timestamp"] = nm.slideTimestamp,
                ["jump_timestamp"] = nm.jumpTimestamp,
                ["velocity_after_slide"] = ObservationBuilder.Vec(afterSlide),
                ["riding_rocket"] = RidingRocket(),
                ["ssj"] = MovementPatches.BuildCounters(),
            };
        }

        private static bool RidingRocket()
        {
            var tracker = MonoSingleton<ObjectTracker>.Instance;
            if (tracker == null) return false;
            var list = tracker.grenadeList;
            for (int i = 0; i < list.Count; i++)
            {
                var g = list[i];
                if (g != null && g.playerRiding) return true;
            }
            return false;
        }

        // ---- Block B ------------------------------------------------------------------------------------

        internal JObject BuildWeapon()
        {
            var gun = MonoSingleton<GunControl>.Instance;
            var charges = MonoSingleton<WeaponCharges>.Instance;
            var fists = MonoSingleton<FistControl>.Instance;
            var hook = MonoSingleton<HookArm>.Instance;

            var obj = new JObject
            {
                ["slot"] = gun != null ? gun.currentSlotIndex : -1,
                ["variation"] = gun != null ? gun.currentVariationIndex : -1,
                ["variations_in_slot"] = VariationsInSlot(gun),
                // Revolver.Update gates firing on this, and only the ReadyGun animation event sets it after a
                // draw. A slot press on the already-equipped slot re-draws the weapon (WeaponRedrawBehaviour
                // defaults to "cycle variation"), so this is how a policy suppressing its own fire shows up.
                ["gun_ready"] = gun != null && PlayerFields.GunReady(gun.currentWeapon),
            };

            if (charges != null)
            {
                obj["pierce_charge"] = charges.rev0charge;      // of 100
                obj["coin_charge"] = charges.rev1charge;        // of 400
                obj["sharp_charge"] = charges.rev2charge;       // of 300
                obj["core_charge"] = charges.shoAltNadeCharge;  // of 1, the Core Eject grenade
                obj["saw_charge"] = charges.shoSawCharge;
                obj["rai_charge"] = charges.raicharge;          // of 5
                obj["rocket_charge"] = charges.rocketcharge;
                obj["rocket_frozen"] = charges.rocketFrozen;
                obj["rocket_freeze_time"] = charges.rocketFreezeTime;
            }
            if (fists != null)
            {
                obj["fist_cooldown"] = fists.fistCooldown;
                obj["holding_item"] = fists.heldObject != null;
            }
            if (hook != null)
            {
                obj["hook_equipped"] = hook.equipped;
                obj["hook_state"] = (int)hook.state;
                obj["hook_state_name"] = hook.state.ToString();
                obj["being_pulled"] = hook.beingPulled;
            }
            return obj;
        }

        private static int VariationsInSlot(GunControl gun)
        {
            if (gun == null || gun.slots == null) return 0;
            int i = gun.currentSlotIndex - 1;
            return i >= 0 && i < gun.slots.Count && gun.slots[i] != null ? gun.slots[i].Count : 0;
        }

        // ---- Block C ------------------------------------------------------------------------------------

        /// <summary>
        /// The player's own live projectiles, nearest first. Enemy-owned objects are excluded: a coin the
        /// player did not throw and a rocket an enemy fired are hazards, not tools, and they already show up
        /// through the enemy block.
        /// </summary>
        internal JArray BuildProjectiles(NewMovement nm)
        {
            var arr = new JArray();
            nearest.Clear();
            seenThisBuild.Clear();
            float now = Time.time;
            var origin = nm.transform.position;

            var coins = MonoSingleton<CoinTracker>.Instance;
            if (coins != null)
            {
                var list = coins.revolverCoinsList;
                for (int i = 0; i < list.Count; i++)
                {
                    var c = list[i];
                    if (c == null || c.shotByEnemy) continue;
                    Add(nm, origin, now, c.gameObject, c.GetInstanceID(), "coin", c.transform.position,
                        c.rb != null ? c.rb.velocity : Vector3.zero, c.shot, false, false);
                }
            }

            var tracker = MonoSingleton<ObjectTracker>.Instance;
            if (tracker != null)
            {
                var grenades = tracker.grenadeList;
                for (int i = 0; i < grenades.Count; i++)
                {
                    var g = grenades[i];
                    if (g == null || g.enemy) continue;
                    Add(nm, origin, now, g.gameObject, g.GetInstanceID(), g.rocket ? "rocket" : "grenade",
                        g.transform.position, g.rb != null ? g.rb.velocity : Vector3.zero,
                        g.playerRiding, g.frozen, g.rideable);
                }
                var balls = tracker.cannonballList;
                for (int i = 0; i < balls.Count; i++)
                {
                    var b = balls[i];
                    if (b == null) continue;
                    Add(nm, origin, now, b.gameObject, b.GetInstanceID(), "cannonball", b.transform.position,
                        Vector3.zero, false, false, false);
                }
            }

            nearest.Sort((a, b) => a.dist.CompareTo(b.dist));
            for (int i = 0; i < nearest.Count && i < MaxProjectiles; i++) arr.Add(nearest[i].obj);

            // Objects gone since the last build stop paying for a dictionary slot. Bounded by the game's own
            // lists, which hold a handful of entries, so this never becomes a scan worth caring about.
            expired.Clear();
            foreach (var kv in firstSeen)
            {
                if (!seenThisBuild.Contains(kv.Key)) expired.Add(kv.Key);
            }
            for (int i = 0; i < expired.Count; i++) firstSeen.Remove(expired[i]);
            return arr;
        }

        private void Add(NewMovement nm, Vector3 origin, float now, GameObject go, int id, string kind,
                         Vector3 pos, Vector3 vel, bool riding, bool frozen, bool rideable)
        {
            if (go == null || !go.activeInHierarchy) return;
            seenThisBuild.Add(id);
            if (!firstSeen.TryGetValue(id, out var born))
            {
                born = now;
                firstSeen[id] = born;
            }
            float dist = Vector3.Distance(origin, pos);
            nearest.Add((new JObject
            {
                ["id"] = id,
                ["kind"] = kind,
                ["pos"] = ObservationBuilder.Vec(pos),
                ["rel"] = ObservationBuilder.Vec(nm.transform.InverseTransformPoint(pos)),
                ["vel"] = ObservationBuilder.Vec(vel),
                ["dist"] = dist,
                ["age"] = now - born,
                ["riding"] = riding,
                ["frozen"] = frozen,
                ["rideable"] = rideable,
            }, dist));
        }
    }
}
