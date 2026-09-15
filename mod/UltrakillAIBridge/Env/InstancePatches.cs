using System;
using System.IO;
using HarmonyLib;

namespace UltrakillAIBridge.Env
{
    /// <summary>
    /// Makes it safe to run several copies of the game for parallel training.
    ///
    /// Training instances (launched with -aibridge-port) open the preferences files read-only and never
    /// write preferences or save data, so they can't corrupt each other's or the player's files. The
    /// game normally holds Prefs.json open with exclusive write access, which would crash a second copy.
    ///
    /// Save data is also never written while the AI has control in any instance, so AI runs don't change
    /// the player's progress.
    /// </summary>
    [HarmonyPatch]
    internal static class InstancePatches
    {
        private static readonly AccessTools.FieldRef<PrefsManager, FileStream> PrefsStream =
            AccessTools.FieldRefAccess<PrefsManager, FileStream>("prefsStream");

        private static readonly AccessTools.FieldRef<PrefsManager, FileStream> LocalPrefsStream =
            AccessTools.FieldRefAccess<PrefsManager, FileStream>("localPrefsStream");

        [HarmonyPrefix]
        [HarmonyPatch(typeof(PrefsManager), "Initialize")]
        private static void OpenPrefsReadOnly(PrefsManager __instance)
        {
            if (!Plugin.IsTrainingInstance) return;
            try
            {
                if (PrefsStream(__instance) == null) PrefsStream(__instance) = OpenShared("Prefs.json");
                if (LocalPrefsStream(__instance) == null) LocalPrefsStream(__instance) = OpenShared("LocalPrefs.json");
            }
            catch (Exception e)
            {
                Plugin.Log.LogWarning($"Could not open preferences read-only, falling back to the game's default: {e.Message}");
            }
        }

        private static FileStream OpenShared(string file)
        {
            var path = Path.Combine(PrefsManager.PrefsPath, file);
            return File.Exists(path) ? new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete) : null;
        }

        [HarmonyPrefix]
        [HarmonyPatch(typeof(PrefsManager), "CommitPrefs")]
        private static bool SkipPrefsWrite() => !Plugin.IsTrainingInstance;

        [HarmonyPrefix]
        [HarmonyPatch(typeof(GameProgressSaver), "WriteFile")]
        private static bool SkipSaveWrite() => !Plugin.IsTrainingInstance && !EpisodeController.InControl;
    }
}
