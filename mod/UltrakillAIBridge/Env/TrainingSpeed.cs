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
    /// to AlwaysAnimate so animation-driven attacks don't freeze when nothing is rendered.
    /// </summary>
    [HarmonyPatch]
    internal static class TrainingSpeed
    {
        internal static bool SoftDeathEnabled;
        internal static bool RenderingDisabled;

        internal static int SoftDeaths { get; private set; }
        internal static bool LastSoftDeathInstakill { get; private set; }

        private static readonly List<Camera> disabledCameras = new List<Camera>();

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
            if (Camera.allCamerasCount == 0) return;
            foreach (var cam in Camera.allCameras)
            {
                cam.enabled = false;
                disabledCameras.Add(cam);
            }
        }

        internal static void RestoreRendering()
        {
            foreach (var cam in disabledCameras)
            {
                if (cam != null) cam.enabled = true;
            }
            disabledCameras.Clear();
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
}
