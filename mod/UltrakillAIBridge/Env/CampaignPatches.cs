using System;
using System.Collections.Generic;
using HarmonyLib;
using UnityEngine;

namespace UltrakillAIBridge.Env
{
    /// <summary>
    /// Campaign support, active only while the AI has control.
    ///
    /// Config: <see cref="DifficultyOverride"/> replaces the difficulty the game reads from its prefs, and
    /// <see cref="UnlockAllGear"/> makes every weapon, variant and arm read as owned and switched on. Both live
    /// in memory only (prefs and save writes are skipped while the AI has control, see InstancePatches) and
    /// must be sent before the level loads, because enemies and GunSetter read them at scene start.
    ///
    /// Milestones: an arena's last wave clearing (ActivateNextWave.EndWaves) and a locked door unlocking
    /// (Door.Unlock) are recorded as rounded-position keys, not counters. A checkpoint respawn re-creates the
    /// rooms it owns at the same positions, so an arena cleared again after a death gives the same key and
    /// Python pays each key once per level load. EndWaves is invoked several times per arena (once per door,
    /// then a final call), which a set absorbs. Both sets are cleared when a scene loads.
    ///
    /// Respawns: while the AI has control only the bridge's own checkpoint reset may call StatsManager.Restart.
    /// Outside Cyber Grind, StatsManager.Update restarts by itself when the player is dead and Fire1 is newly
    /// pressed (or R), and the pause menu can restart too; that would respawn the player, or reload the level
    /// when there is no checkpoint, in the middle of a step without Python ever seeing the death.
    /// </summary>
    [HarmonyPatch]
    internal static class CampaignPatches
    {
        /// <summary>Difficulty the game reads while the AI has control (0 Harmless .. 4 Brutal); -1 leaves the game's setting.</summary>
        internal static int DifficultyOverride = -1;
        internal static bool UnlockAllGear;

        /// <summary>True only while EpisodeController itself calls StatsManager.Restart (reset with checkpoint=true).</summary>
        internal static bool BridgeRestart;

        private static readonly HashSet<string> clearedArenas = new HashSet<string>();
        private static readonly HashSet<string> unlockedDoors = new HashSet<string>();

        internal static IReadOnlyCollection<string> ClearedArenas => clearedArenas;
        internal static IReadOnlyCollection<string> UnlockedDoors => unlockedDoors;

        /// <summary>Position key shared with Python: whole metres, "x,y,z" (also used as checkpoint ids).</summary>
        internal static string Key(Vector3 p) =>
            FormattableString.Invariant($"{Mathf.RoundToInt(p.x)},{Mathf.RoundToInt(p.y)},{Mathf.RoundToInt(p.z)}");

        /// <summary>Called on every single-mode scene load (subscribed in Plugin).</summary>
        internal static void OnSceneLoaded()
        {
            clearedArenas.Clear();
            unlockedDoors.Clear();
        }

        [HarmonyPostfix]
        [HarmonyPatch(typeof(PrefsManager), nameof(PrefsManager.GetInt))]
        private static void OverridePrefInt(string key, ref int __result)
        {
            if (!EpisodeController.InControl || key == null) return;
            if (key == "difficulty" && DifficultyOverride >= 0)
            {
                __result = DifficultyOverride;
            }
            else if (UnlockAllGear && __result == 0 && key.StartsWith("weapon.", StringComparison.Ordinal) && key.IndexOf('.', 7) < 0)
            {
                // "weapon.rev0" style keys: 0 = switched off. 2 (the alternate version) is left alone.
                __result = 1;
            }
        }

        [HarmonyPostfix]
        [HarmonyPatch(typeof(GameProgressSaver), nameof(GameProgressSaver.CheckGear))]
        private static void UnlockGear(ref int __result)
        {
            if (EpisodeController.InControl && UnlockAllGear && __result == 0) __result = 1;
        }

        [HarmonyPrefix]
        [HarmonyPatch(typeof(ActivateNextWave), "EndWaves")]
        private static void RecordArenaClear(ActivateNextWave __instance)
        {
            if (EpisodeController.InControl && __instance != null) clearedArenas.Add(Key(__instance.transform.position));
        }

        [HarmonyPrefix]
        [HarmonyPatch(typeof(Door), nameof(Door.Unlock))]
        private static void RecordDoorUnlock(Door __instance)
        {
            if (EpisodeController.InControl && __instance != null && __instance.locked) unlockedDoors.Add(Key(__instance.transform.position));
        }

        [HarmonyPrefix]
        [HarmonyPatch(typeof(StatsManager), nameof(StatsManager.Restart))]
        private static bool OnlyBridgeRestarts() => !EpisodeController.InControl || BridgeRestart;
    }
}
