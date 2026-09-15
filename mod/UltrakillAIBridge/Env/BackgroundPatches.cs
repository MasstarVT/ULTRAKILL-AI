using HarmonyLib;
using UnityEngine;

namespace UltrakillAIBridge.Env
{
    /// <summary>
    /// Lets the game run quietly in the background while the AI has control: the game's own state
    /// logic re-locks the cursor and restores the volume (on every scene load), so both are overridden
    /// right after the game sets them. <see cref="EpisodeController"/> also enforces them every frame.
    /// </summary>
    [HarmonyPatch]
    internal static class BackgroundPatches
    {
        [HarmonyPostfix]
        [HarmonyPatch(typeof(GameStateManager), "EvaluateState")]
        private static void FreeCursor()
        {
            if (EpisodeController.InControl) EpisodeController.ApplyCursorAndAudio();
        }

        [HarmonyPostfix]
        [HarmonyPatch(typeof(GameStateManager), "IntroCheck")]
        private static void KeepMutedOnSceneLoad()
        {
            if (EpisodeController.InControl) EpisodeController.ApplyCursorAndAudio();
        }

        [HarmonyPostfix]
        [HarmonyPatch(typeof(AudioMixerController), "OnPrefChanged")]
        private static void KeepMutedOnPrefChange()
        {
            if (EpisodeController.InControl) EpisodeController.ApplyCursorAndAudio();
        }
    }
}
