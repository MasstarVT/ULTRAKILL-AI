using System;
using System.Collections;
using HarmonyLib;
using UnityEngine;

namespace UltrakillAIBridge.Env
{
    /// <summary>
    /// Hitstop and parry freezes wait with WaitForSecondsRealtime, which is wall-clock time and
    /// ignores Time.captureDeltaTime. During lockstep that would make a freeze last a different number
    /// of frames depending on how fast the machine or policy is. Here they count frames instead.
    /// </summary>
    [HarmonyPatch]
    internal static class TimePatches
    {
        private static readonly Action<TimeController, float, bool> ContinueTime =
            AccessTools.MethodDelegate<Action<TimeController, float, bool>>(AccessTools.Method(typeof(TimeController), "ContinueTime"));

        [HarmonyPrefix]
        [HarmonyPatch(typeof(TimeController), "TimeIsStopped")]
        private static bool FrameBasedStop(TimeController __instance, float length, bool trueStop, ref IEnumerator __result)
        {
            if (Time.captureDeltaTime <= 0f) return true;
            __result = StopForFrames(__instance, length, trueStop);
            return false;
        }

        private static IEnumerator StopForFrames(TimeController tc, float length, bool trueStop)
        {
            float elapsed = 0f;
            while (elapsed < length)
            {
                yield return null;
                elapsed += Time.captureDeltaTime > 0f ? Time.captureDeltaTime : Time.unscaledDeltaTime;
            }
            if (tc != null) ContinueTime(tc, length, trueStop);
        }
    }
}
