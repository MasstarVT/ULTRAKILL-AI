using HarmonyLib;

namespace UltrakillAIBridge.Env
{
    /// <summary>
    /// Keeps AI runs off the Steam leaderboards. Always active while the mod is installed.
    /// </summary>
    [HarmonyPatch]
    internal static class SafetyPatches
    {
        [HarmonyPrefix]
        [HarmonyPatch(typeof(LeaderboardController), nameof(LeaderboardController.SubmitCyberGrindScore))]
        private static bool BlockCyberGrindScore()
        {
            Plugin.Log.LogInfo("Blocked Cyber Grind leaderboard submission");
            return false;
        }

        [HarmonyPrefix]
        [HarmonyPatch(typeof(LeaderboardController), nameof(LeaderboardController.SubmitLevelScore))]
        private static bool BlockLevelScore()
        {
            Plugin.Log.LogInfo("Blocked level leaderboard submission");
            return false;
        }

        [HarmonyPrefix]
        [HarmonyPatch(typeof(LeaderboardController), nameof(LeaderboardController.SubmitFishSize))]
        private static bool BlockFishSize()
        {
            return false;
        }
    }
}
