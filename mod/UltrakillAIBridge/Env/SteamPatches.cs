using System;
using HarmonyLib;
using Steamworks;

namespace UltrakillAIBridge.Env
{
    /// <summary>
    /// Hides an instance from Steam, so a training copy accrues no playtime and never shows as "running".
    /// Opt-in per process through the <c>-aibridge-nosteam</c> command-line flag (scripts/games.py adds it to
    /// every training launch by default; pass <c>--steam</c> there to leave an instance visible).
    ///
    /// **Why one patch is enough.** Registering with Steam happens in exactly one place: Facepunch's
    /// <c>SteamClient.Init(uint, bool)</c> calls <c>SteamAPI.Init()</c>, which is what loads steam_api64 and
    /// tells a running Steam client that app 1229490 is live. The game calls it once, from
    /// <c>SteamController.Awake</c> (decompiled/SteamController.cs:54), inside a try/catch that already treats
    /// "Steam isn't there" as normal. Nothing in the game calls <c>SteamClient.RestartAppIfNecessary</c>, and
    /// the game ships no <c>steam_appid.txt</c>. Every other Steam consumer -- rich presence, user stats,
    /// UGC playtime tracking, the leaderboards -- is gated on <c>SteamClient.IsValid</c>, which is the private
    /// <c>initialized</c> flag Init sets, so skipping Init leaves all of them switched off by their own code
    /// rather than by another patch. (The two ungated call sites, <c>LeaderboardController</c>'s
    /// <c>SteamClient.SteamId</c> reads, sit behind <c>LeaderboardsSupported</c> and behind
    /// <see cref="SafetyPatches"/>'s fish-size block respectively.)
    ///
    /// Skipping Init also means the <c>SteamAppId</c>/<c>SteamGameId</c> environment variables Init would set
    /// are never set by the game. games.py sets them itself on the launch, which is harmless: an environment
    /// variable tells Steam nothing on its own, only SteamAPI_Init does.
    /// </summary>
    [HarmonyPatch]
    internal static class SteamPatches
    {
        /// <summary>The command-line flag that asks for this. Absent = the game talks to Steam as it always has.</summary>
        public const string Flag = "-aibridge-nosteam";

        /// <summary>True once the skip is actually in place, i.e. reported in the handshake, not merely requested.</summary>
        internal static bool Hidden { get; private set; }

        private static bool skipLogged;

        /// <summary>Whether <see cref="Flag"/> is on this process's command line.</summary>
        public static bool Requested()
        {
            foreach (var arg in Environment.GetCommandLineArgs())
            {
                if (string.Equals(arg, Flag, StringComparison.OrdinalIgnoreCase)) return true;
            }
            return false;
        }

        /// <summary>
        /// Applies the skip when the flag asks for it. Called from Plugin.Awake, which BepInEx's chainloader
        /// runs before the first scene's Awakes, i.e. before <c>SteamController.Awake</c>. If Steam is somehow
        /// already initialised by then the patch would be pointless -- Steam has seen us -- so it is not
        /// applied at all and the situation is logged instead of being silently reported as hidden.
        /// </summary>
        public static void Apply(Harmony harmony)
        {
            if (!Requested()) return;
            if (SteamClient.IsValid)
            {
                Plugin.Log.LogWarning(
                    $"{Flag} was asked for but Steam is already initialised; this instance IS visible to Steam");
                return;
            }
            harmony.PatchAll(typeof(SteamPatches));
            Hidden = true;
            Plugin.Log.LogInfo($"{Flag}: this instance will not register with Steam (no playtime, not shown as running)");
        }

        /// <summary>
        /// Skips <c>SteamClient.Init</c>. Returning false leaves the private <c>initialized</c> flag false, so
        /// <c>SteamClient.IsValid</c> stays false and every consumer takes its own "no Steam" branch. The
        /// caller's try/catch never fires either, because a skipped call is a successful one -- it just does
        /// nothing -- which is why the game logs "Steam initialized!" and then runs entirely without it.
        /// </summary>
        [HarmonyPrefix]
        [HarmonyPatch(typeof(SteamClient), nameof(SteamClient.Init), new[] { typeof(uint), typeof(bool) })]
        private static bool SkipInit(uint appid)
        {
            if (!skipLogged)
            {
                skipLogged = true;
                Plugin.Log.LogWarning($"Skipped SteamClient.Init(appId {appid}): this instance is hidden from Steam");
            }
            return false;
        }
    }
}
