using HarmonyLib;
using UnityEngine;

namespace UltrakillAIBridge
{
    /// <summary>
    /// Lazily-resolved readers for the private game fields the 0.8 technique work needs.
    ///
    /// Every one of these degrades to a neutral default with a single logged warning if a game update renames
    /// it, exactly as <see cref="Obs.ObservationBuilder"/> already does for NewMovement.crouching: losing a
    /// field must cost that field, never the bridge. Nothing here writes -- these are reads only, so a rename
    /// can make the agent blind but can never make it act on a wrong value.
    ///
    /// Checked against the decompiled build on 2026-09-20; see docs/game-internals.md for the list.
    /// </summary>
    internal static class PlayerFields
    {
        private static bool resolved;

        private static AccessTools.FieldRef<NewMovement, bool> jumpCooldownRef;
        private static AccessTools.FieldRef<NewMovement, WallCheckGroup> wcGroupRef;
        private static AccessTools.FieldRef<NewMovement, float> boostLeftRef;
        private static AccessTools.FieldRef<NewMovement, float> dashStorageRef;
        private static AccessTools.FieldRef<NewMovement, bool> slamStorageRef;
        private static AccessTools.FieldRef<NewMovement, Vector3> velocityAfterSlideRef;
        private static AccessTools.FieldRef<NewMovement, float> ssjMaxFramesRef;
        private static AccessTools.FieldRef<Revolver, bool> gunReadyRef;

        /// <summary>One SSJ frame bucket, 0.008 s wide. TrySSJ divides by this exact literal.</summary>
        internal const double SsjFrame = 0.00800000037997961;

        private static AccessTools.FieldRef<T, F> Ref<T, F>(string name)
        {
            try
            {
                return AccessTools.FieldRefAccess<T, F>(name);
            }
            catch (System.Exception e)
            {
                Plugin.Log.LogWarning($"{typeof(T).Name}.{name} not found, reported as its default: {e.Message}");
                return null;
            }
        }

        private static void Resolve()
        {
            if (resolved) return;
            resolved = true;
            jumpCooldownRef = Ref<NewMovement, bool>("jumpCooldown");
            wcGroupRef = Ref<NewMovement, WallCheckGroup>("wcGroup");
            boostLeftRef = Ref<NewMovement, float>("boostLeft");
            dashStorageRef = Ref<NewMovement, float>("dashStorage");
            slamStorageRef = Ref<NewMovement, bool>("slamStorage");
            velocityAfterSlideRef = Ref<NewMovement, Vector3>("velocityAfterSlide");
            ssjMaxFramesRef = Ref<NewMovement, float>("ssjMaxFrames");
            gunReadyRef = Ref<Revolver, bool>("gunReady");
        }

        /// <summary>NewMovement.jumpCooldown. Reported false when unavailable, so a macro is never refused for a field we lost.</summary>
        internal static bool JumpCooldown(NewMovement nm)
        {
            Resolve();
            return nm != null && jumpCooldownRef != null && jumpCooldownRef.Invoke(nm);
        }

        /// <summary>NewMovement.wcGroup, the component HandleInputs tests before calling WallJump. Null when unavailable.</summary>
        internal static WallCheckGroup WallChecks(NewMovement nm)
        {
            Resolve();
            return nm != null && wcGroupRef != null ? wcGroupRef.Invoke(nm) : null;
        }

        /// <summary>Dash i-frames left: NewMovement.boostLeft, which also gates layer 15 invincibility.</summary>
        internal static float BoostLeft(NewMovement nm)
        {
            Resolve();
            return nm != null && boostLeftRef != null ? boostLeftRef.Invoke(nm) : 0f;
        }

        internal static float DashStorage(NewMovement nm)
        {
            Resolve();
            return nm != null && dashStorageRef != null ? dashStorageRef.Invoke(nm) : 0f;
        }

        internal static bool SlamStorage(NewMovement nm)
        {
            Resolve();
            return nm != null && slamStorageRef != null && slamStorageRef.Invoke(nm);
        }

        /// <summary>The vector TrySSJ overwrites rb.velocity with. Written only by StopSlide.</summary>
        internal static Vector3 VelocityAfterSlide(NewMovement nm)
        {
            Resolve();
            return nm != null && velocityAfterSlideRef != null ? velocityAfterSlideRef.Invoke(nm) : Vector3.zero;
        }

        /// <summary>NewMovement.ssjMaxFrames, serialized, 4 in this build. Falls back to 4 when unavailable.</summary>
        internal static float SsjMaxFrames(NewMovement nm)
        {
            Resolve();
            if (nm == null || ssjMaxFramesRef == null) return 4f;
            var v = ssjMaxFramesRef.Invoke(nm);
            return v > 0f ? v : 4f;
        }

        /// <summary>
        /// Revolver.gunReady. Only the ReadyGun animation event clears it after a draw, and Revolver.Update
        /// gates firing on it, so this is the field that shows a redraw suppressing the agent's own fire.
        /// Reported true for anything that is not a Revolver, i.e. "nothing is blocking the trigger".
        /// </summary>
        internal static bool GunReady(GameObject weapon)
        {
            Resolve();
            if (weapon == null || gunReadyRef == null) return true;
            var rev = weapon.GetComponent<Revolver>();
            return rev == null || gunReadyRef.Invoke(rev);
        }
    }
}
