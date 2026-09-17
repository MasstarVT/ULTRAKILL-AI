using System.Collections.Generic;
using HarmonyLib;
using UnityEngine;

namespace UltrakillAIBridge.Env
{
    /// <summary>
    /// Optional training speed-ups, active only while the AI has control and enabled through config.
    ///
    /// Soft death: a hit that would kill the player heals them instead and increments a counter. Python
    /// treats that as a terminal death (same penalty), but the next episode continues in the same scene,
    /// so parallel games aren't stalled by scene reloads. Every death goes through NewMovement.GetHurt,
    /// including pits (DeathZone) and instakills.
    ///
    /// No rendering: cameras are disabled because the agent never sees pixels. Enemy Animators are forced
    /// to AlwaysAnimate so animation-driven attacks don't freeze when nothing is rendered, and so are the
    /// player's own fist and camera Animators (see ForcePlayerAnimators).
    /// </summary>
    [HarmonyPatch]
    internal static class TrainingSpeed
    {
        internal static bool SoftDeathEnabled;
        internal static bool RenderingDisabled;

        internal static int SoftDeaths { get; private set; }
        internal static bool LastSoftDeathInstakill { get; private set; }

        private static readonly List<Camera> disabledCameras = new List<Camera>();

        /// <summary>
        /// Player Animators forced to AlwaysAnimate, with the culling mode each had, so RestoreRendering puts
        /// them back exactly as disabledCameras does for the cameras.
        /// </summary>
        private static readonly List<Animator> forcedAnimators = new List<Animator>();
        private static readonly List<AnimatorCullingMode> forcedAnimatorModes = new List<AnimatorCullingMode>();

        /// <summary>
        /// Identity of the Animator set that was last walked: the FistControl/CameraController instance ids
        /// folded together with the instance id of every DIRECT child of each, so the walk itself runs once
        /// per player and once per arm change rather than once per step.
        ///
        /// Child instance ids, not a child count. FistControl.ResetFists destroys every spawned arm and
        /// immediately instantiates the replacements as children of the same FistControl
        /// (decompiled/FistControl.cs:217-256; the Instantiate calls are at :231, :283, :300, :307). Prefs do
        /// not change mid-level, so the arm set is normally rebuilt one for one: the count goes N -> N over an
        /// entirely new set of Animators and a count comparison never invalidates. The new arm's Punch
        /// Animator would then keep its authored culling mode, Unity would cull it because every camera is
        /// disabled, the "ActiveStart" AnimationEvent would stop firing and Punch.AltHit would never run --
        /// i.e. picking up or placing an item becomes silently impossible for the rest of the level load.
        ///
        /// ResetFists runs mid-level, not just at spawn: WeaponPickUp.cs:104 (walking over an arm pickup),
        /// PlayerLoadout.SetLoadout, PlayerLoadoutTarget.CommitLoadout, VariationInfo.cs:199,247 and
        /// FistControl.TutorialCheckForArmThatCanPunch (a UnityEvent target with no C# caller). The loadout
        /// paths can also swap which arm prefabs exist without changing the count.
        /// </summary>
        private static int forcedAnimatorsId;

        /// <summary>
        /// Whether forcedAnimatorsId describes a walk that has actually happened. A separate flag rather than
        /// "id == 0", because the hash of a live pair can legitimately be 0 and that must not read as "already
        /// walked" after a restore.
        /// </summary>
        private static bool forcedAnimatorsValid;

        /// <summary>
        /// The two singleton instance ids alone, so a re-walk can tell "a new player" (every level load) from
        /// "the same player's arms were rebuilt" (ResetFists). Only the second is worth a log line.
        /// </summary>
        private static int forcedPlayerId;

        /// <summary>
        /// Re-walks caused by an arm rebuild rather than a new player, counted once per run. A diagnostic, not
        /// sent in obs: the whole point of this class's fist handling is that its failure is silent, so the
        /// first few rebuilds are logged and the counter makes the rest visible if anything ever needs it.
        /// </summary>
        internal static int ArmRebuilds { get; private set; }

        [HarmonyPrefix]
        [HarmonyPatch(typeof(NewMovement), nameof(NewMovement.GetHurt))]
        private static bool PreventLethalHit(NewMovement __instance, int damage, bool invincible, bool instablack, bool ignoreInvincibility)
        {
            if (!EpisodeController.InControl || !SoftDeathEnabled) return true;

            var nm = __instance;
            // Same early-outs and damage scaling as GetHurt; anything non-lethal runs the original.
            if (nm.dead || nm.levelOver || damage <= 0) return true;
            if (invincible && nm.gameObject.layer == 15 && !ignoreInvincibility) return true;
            var assist = MonoSingleton<AssistController>.Instance;
            int scaled = assist != null && assist.majorEnabled ? Mathf.RoundToInt(damage * assist.damageTaken) : damage;
            if (ULTRAKILL.Cheats.Invincibility.Enabled) scaled = 0;
            if (nm.hp - scaled > 0) return true;

            SoftDeaths++;
            LastSoftDeathInstakill = instablack || damage >= 999;
            Heal(nm);
            return false;
        }

        internal static void Heal(NewMovement nm)
        {
            nm.ResetHardDamage();
            nm.exploded = false;
            nm.GetHealth(999, silent: true);
            nm.FullStamina();
            MonoSingleton<WeaponCharges>.Instance?.MaxCharges();
        }

        /// <summary>Called every step and after resets; cheap when there is nothing to change.</summary>
        internal static void ApplyRendering()
        {
            if (!EpisodeController.InControl || !RenderingDisabled) return;
            ForcePlayerAnimators();
            if (Camera.allCamerasCount == 0) return;
            foreach (var cam in Camera.allCameras)
            {
                cam.enabled = false;
                disabledCameras.Add(cam);
            }
        }

        /// <summary>
        /// Keeps the player's fist and camera Animators running while every camera is disabled.
        ///
        /// Punch.ActiveFrame is the only caller of Punch.AltHit, which is the only way an item is picked up
        /// or placed on an altar, and it only runs while Punch.activeFrames is positive -- which nothing but
        /// the "ActiveStart" AnimationEvent sets (grepped: no C# caller exists for ActiveStart or ActiveEnd).
        /// Unity culls an Animator whose renderers are never drawn, so with rendering off the fist animation
        /// would not advance, the event would never fire and skull carrying would be silently impossible.
        /// Enemy Animators are already handled the same way in OnEnemyAdded.
        ///
        /// The hierarchy walk runs once per player, and again whenever the set of arms changes: both singletons
        /// are NoAutoInstance, so reading Instance is a static field read that returns null (and never creates
        /// anything) before the player exists, and a hash over their instance ids and their direct children,
        /// plus a scan of the forced list for destroyed entries, is all this compares per step. The walk itself
        /// only runs when one of those says the set has changed.
        /// </summary>
        private static void ForcePlayerAnimators()
        {
            var fists = MonoSingleton<FistControl>.Instance;
            var camera = MonoSingleton<CameraController>.Instance;
            if (fists == null && camera == null) return;
            int id = AnimatorSetId(fists, camera);
            if (forcedAnimatorsValid && id == forcedAnimatorsId && !AnyForcedAnimatorDestroyed()) return;

            int playerId = PlayerId(fists, camera);
            if (forcedAnimatorsValid && playerId == forcedPlayerId)
            {
                ArmRebuilds++;
                if (ArmRebuilds <= 4)
                {
                    Plugin.Log.LogInfo(
                        $"Fist Animators rebuilt mid-level (ResetFists), re-forcing AlwaysAnimate (#{ArmRebuilds})");
                }
            }
            forcedAnimatorsValid = true;
            forcedAnimatorsId = id;
            forcedPlayerId = playerId;
            // The previous player's Animators are a different (usually destroyed) set: put them back and drop
            // them, so a run that loads thousands of levels doesn't accumulate a list of dead references.
            RestoreForcedAnimators();
            ForceAnimators(fists);
            ForceAnimators(camera);
        }

        /// <summary>
        /// Hash of the Animator set's identity: the two singletons and every direct child of each, so a child
        /// added, removed, or destroyed and replaced one for one all change it (see forcedAnimatorsId). Order
        /// dependent and allocation free; the two transforms have a handful of children each, and
        /// ApplyRendering already reads Camera.allCameras every step.
        /// </summary>
        private static int AnimatorSetId(Component fists, Component camera)
        {
            unchecked
            {
                return FoldChildren(FoldChildren(17, fists), camera);
            }
        }

        private static int PlayerId(Component fists, Component camera)
        {
            unchecked
            {
                return (fists != null ? fists.GetInstanceID() : 0) * 31
                       + (camera != null ? camera.GetInstanceID() : 0);
            }
        }

        private static int FoldChildren(int hash, Component root)
        {
            unchecked
            {
                if (root == null) return hash * 31;
                hash = hash * 31 + root.GetInstanceID();
                var t = root.transform;
                int count = t.childCount;
                for (int i = 0; i < count; i++)
                {
                    hash = hash * 31 + t.GetChild(i).GetInstanceID();
                }
                return hash;
            }
        }

        /// <summary>
        /// True when any Animator this last forced has since been destroyed. UnityEngine.Object's == reports a
        /// destroyed object as null, so a destroy-and-respawn is caught here whatever the ids hash to, and at
        /// any depth rather than only among direct children. It also cleans up after the frame where Unity has
        /// instantiated the new arms but not yet processed the Destroy of the old ones: those doomed Animators
        /// get forced, and are dropped on the next step.
        ///
        /// Only Animators that were NOT already AlwaysAnimate are in the list (ForceAnimators skips the rest),
        /// which is why the id hash is kept as well: a loadout swap whose old arms were all authored
        /// AlwaysAnimate would destroy nothing this scan can see.
        /// </summary>
        private static bool AnyForcedAnimatorDestroyed()
        {
            for (int i = 0; i < forcedAnimators.Count; i++)
            {
                if (forcedAnimators[i] == null) return true;
            }
            return false;
        }

        private static void ForceAnimators(Component root)
        {
            if (root == null) return;
            foreach (var animator in root.GetComponentsInChildren<Animator>(true))
            {
                if (animator == null || animator.cullingMode == AnimatorCullingMode.AlwaysAnimate) continue;
                forcedAnimators.Add(animator);
                forcedAnimatorModes.Add(animator.cullingMode);
                animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
            }
        }

        internal static void RestoreRendering()
        {
            foreach (var cam in disabledCameras)
            {
                if (cam != null) cam.enabled = true;
            }
            disabledCameras.Clear();
            RestoreForcedAnimators();
            forcedAnimatorsId = 0;
            forcedPlayerId = 0;
            forcedAnimatorsValid = false;
        }

        private static void RestoreForcedAnimators()
        {
            for (int i = 0; i < forcedAnimators.Count; i++)
            {
                var animator = forcedAnimators[i];
                if (animator != null) animator.cullingMode = forcedAnimatorModes[i];
            }
            forcedAnimators.Clear();
            forcedAnimatorModes.Clear();
        }

        internal static void OnEnemyAdded(EnemyIdentifier eid)
        {
            if (!EpisodeController.InControl || !RenderingDisabled || eid == null) return;
            foreach (var animator in eid.GetComponentsInChildren<Animator>(true))
            {
                animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
            }
        }
    }

    /// <summary>
    /// Un-wedge: breaks the absorbing airborne slowMode state, active only while the AI has control.
    ///
    /// A slide that ends while the player is airborne, or a jump out of a slide, reaches
    /// NewMovement.HandleSlideState's stand-up test with the collider still 1.25 m tall. When the ceiling
    /// check above the player fails it sets crouching and slowMode and returns (NewMovement.cs:921-928).
    /// From there the player never lands: GroundCheck keeps onGround from OnTriggerEnter over a list of
    /// colliders, and a collider the ground-check capsule is ALREADY overlapping never fires the callback
    /// again, so the list stays empty. Meanwhile slowMode blocks dash and stamina regen and cuts walk speed
    /// to 1.25x walkSpeed, and NewMovement.Update re-applies the ground-slam velocity every frame while
    /// gc.heavyFall is set. Measured in 0-1's slide vent, at the spawn-room wall and after checkpoint 3:
    /// runs of 672 to 2976 consecutive decisions with no way out.
    ///
    /// The postfix runs after the game's own stand-up test, so it only ever sees the state the game left,
    /// and it re-clears slowMode on every frame that test keeps failing -- which is intended: the player
    /// stays legitimately crouched under a low ceiling, but dash and stamina come back and the ground check
    /// is re-evaluated. It stops by itself the moment the game's own success branch runs.
    ///
    /// It waits for the signature to hold for UnwedgePatch.HoldFrames consecutive frames first, because a
    /// LEGITIMATE ground slam wears the same signature for its first frames. TryStartSlam calls StopSlide,
    /// which does not restore the collider (NewMovement.cs:2448-2474), so a slam started out of a slide
    /// reaches HandleSlideState with height 1.25 still set; under a vent roof, a doorway lintel or any
    /// overhang the stand-up test fails and the game sets crouching and slowMode on the very frame the slam
    /// begins. Acting on frame 1 would clear heavyFall and the -100 velocity there, so the slam would never
    /// land: no LandingImpact, no ground-slam enemy damage and no Breakable.Break(2f) from Update's heavyFall
    /// block (:708-716). A slam that can make progress lands or leaves the low ceiling within a frame or two
    /// and the counter resets; a real wedge holds the signature for hundreds of frames (672-2976 decisions
    /// measured), so the hold separates the two cleanly.
    ///
    /// It never moves the player and never damages anything: heavyFall is cleared BEFORE the ground check
    /// is forced, because GroundCheck.OnTriggerEnter's heavyFall branch deals 5000x damage to every
    /// overlapping enemy and can call Bounce(), which teleports.
    /// </summary>
    [HarmonyPatch]
    internal static class UnwedgePatch
    {
        /// <summary>Config key "unwedge"; a kill switch, on by default.</summary>
        internal static bool Enabled = true;

        /// <summary>
        /// Config key "unwedge_frames": consecutive frames the signature must hold before the state is
        /// broken (see the class comment; clamped to at least 1, where 1 acts on the first frame). 10 frames
        /// is 0.33 s of game time at fixed_fps 30 and 0.17 s at 60, both well inside the 1 s the design
        /// allows, and two orders of magnitude below the shortest measured real wedge.
        /// </summary>
        internal static int HoldFrames = 10;

        private static AccessTools.FieldRef<NewMovement, bool> crouchingRef;
        private static AccessTools.FieldRef<NewMovement, Vector3> groundCheckPosRef;
        private static AccessTools.FieldRef<GroundCheckGroup, List<GroundCheck>> instancesRef;
        private static bool resolved;

        /// <summary>Consecutive frames the signature has held; reset on any frame it does not.</summary>
        private static int heldFrames;

        /// <summary>How many states have been broken this session, counted once per run (a diagnostic, not sent in obs).</summary>
        internal static int Recoveries { get; private set; }

        [HarmonyPostfix]
        [HarmonyPatch(typeof(NewMovement), "HandleSlideState")]
        private static void ClearAirborneSlowMode(NewMovement __instance)
        {
            var nm = __instance;
            // A grounded crouch under a low ceiling is the same state and is legitimate: leave it alone
            // (it keeps reporting slow_mode true in the obs). The counter resets here too, so releasing
            // control or turning the fix off cannot leave it armed for the next frame that matches.
            if (!Enabled || !EpisodeController.InControl || nm == null || nm.gc == null
                || nm.gc.onGround || !nm.slowMode)
            {
                heldFrames = 0;
                return;
            }

            // A ground slam out of a slide wears the same signature while it is still falling: wait for the
            // state to prove it is absorbing before breaking it (see the class comment). The counter
            // saturates one past the threshold, so a wedge that lasts for hours cannot overflow it.
            if (heldFrames <= HoldFrames) heldFrames++;
            if (heldFrames < HoldFrames) return;

            Resolve();
            if (heldFrames == HoldFrames) Recoveries++;

            // 1. Stop the slam before anything touches the ground check (see the class comment).
            nm.gc.heavyFall = false;

            // 2. The velocity NewMovement.Update re-applies every frame while heavyFall is set.
            if (nm.rb != null && nm.rb.velocity.y <= -99f)
            {
                var v = nm.rb.velocity;
                nm.rb.velocity = new Vector3(v.x, 0f, v.z);
            }

            // 3. Defensive and idempotent. StartSlide and the forceCrouch path already leave the collider,
            //    the transform and the ground-check offset mutually consistent, so this normally writes
            //    nothing; it is here so an unforeseen entry path cannot leave them inconsistent.
            if (nm.playerCollider != null && nm.playerCollider.height != 1.25f) nm.playerCollider.height = 1.25f;
            if (groundCheckPosRef != null)
            {
                var target = groundCheckPosRef(nm) + Vector3.up * 1.125f;
                if (nm.gc.transform.localPosition != target) nm.gc.SetLocalPosition(target);
            }
            if (crouchingRef != null && !crouchingRef(nm)) crouchingRef(nm) = true;

            // 4. Give dash, stamina regen and full walk speed back.
            nm.slowMode = false;

            // 5. Re-evaluate the ground: the callbacks that maintain onGround cannot fire for a collider
            //    the capsule is already inside, which is why the state never ends by itself.
            ForceGroundChecks(nm.gc);
        }

        private static void ForceGroundChecks(GroundCheckGroup group)
        {
            var instances = instancesRef?.Invoke(group);
            if (instances != null)
            {
                foreach (var gc in instances)
                {
                    if (gc != null && gc.isActiveAndEnabled) gc.ForceGroundCheck();
                }
                return;
            }
            foreach (var gc in group.GetComponentsInChildren<GroundCheck>(true))
            {
                if (gc != null && gc.isActiveAndEnabled) gc.ForceGroundCheck();
            }
        }

        private static void Resolve()
        {
            if (resolved) return;
            resolved = true;
            try
            {
                crouchingRef = AccessTools.FieldRefAccess<NewMovement, bool>("crouching");
                groundCheckPosRef = AccessTools.FieldRefAccess<NewMovement, Vector3>("groundCheckPos");
            }
            catch (System.Exception e)
            {
                Plugin.Log.LogWarning($"Un-wedge: NewMovement fields not found, skipping the collider fixups: {e.Message}");
            }
            try
            {
                instancesRef = AccessTools.FieldRefAccess<GroundCheckGroup, List<GroundCheck>>("instances");
            }
            catch (System.Exception e)
            {
                // GroundCheckGroup has no ForceGroundCheck of its own; that method is on GroundCheck.
                Plugin.Log.LogWarning($"Un-wedge: GroundCheckGroup.instances not found, using GetComponentsInChildren: {e.Message}");
            }
        }
    }
}
