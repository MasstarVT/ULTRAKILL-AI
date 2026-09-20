using HarmonyLib;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace UltrakillAIBridge.Env
{
    /// <summary>
    /// The Super Slide Jump instrument, and the game's own SSJ indicator.
    ///
    /// **Why an instrument at all.** `NewMovement.TrySSJ` quantises `jumpTimestamp - slideTimestamp` into
    /// 0.008 s buckets and silently does nothing for bucket 0 or bucket >= `ssjMaxFrames`. From outside, a
    /// missed SSJ and a landed one differ only by a velocity change that other things also cause, so a macro
    /// could not be judged. Reading the bucket is the difference between inferring the mechanism and measuring
    /// it, and it is what `macro.ssj_bucket` reports back to the client -- the value the spec's reward gate is
    /// meant to key on, rather than a speed delta, which is perversely signed near the 100 u/s clamp.
    ///
    /// This patch READS ONLY. The prefix takes a velocity snapshot, the postfix takes another; neither
    /// changes an argument, a return value or any game state, and neither skips the original. Patching also
    /// pins TrySSJ against Mono inlining it into Jump/WallJump, which is a side benefit, not the purpose.
    ///
    /// The whole class is applied inside its own try/catch in Plugin.Awake: TrySSJ is private, so a game
    /// update renaming it must cost the instrument and nothing else.
    /// </summary>
    [HarmonyPatch]
    internal static class MovementPatches
    {
        /// <summary>False when the TrySSJ patch failed to apply; the obs then reports the SSJ block as null.</summary>
        internal static bool InstrumentAvailable;

        /// <summary>
        /// Makes PrefsManager report the game's own `ssjIndicator` preference as on while the AI has control,
        /// which is what makes `TrySSJ` print its bucket bar and `+{n}u/s` as a subtitle. Purely a human-visible
        /// readout for the private verification game -- the machine-readable answer is <see cref="Last"/>.
        /// Off by default; the player's stored preference is never written (InstancePatches makes prefs
        /// read-only in a training instance anyway).
        /// </summary>
        internal static bool SsjIndicator;

        /// <summary>One SSJ attempt: what TrySSJ was handed and what it did with it.</summary>
        internal struct SsjEvent
        {
            public int Frame;          // Time.frameCount when TrySSJ ran
            public double Dt;          // jumpTimestamp - slideTimestamp, seconds of REAL time
            public int Bucket;         // (int)(Dt / 0.008); -1 when Dt <= 0, i.e. no slide release preceded the jump
            public bool Landed;        // Dt > 0 && Bucket != 0 && Bucket < ssjMaxFrames -- the game's own accept test
            public bool Wall;          // speedMultiplier 0.75 = WallJump's call site; 0.5 = Jump's
            public float SpeedBefore, SpeedAfter;
            public float HorizontalBefore, HorizontalAfter;
        }

        /// <summary>The most recent attempt, landed or not. `Frame` is 0 until one happens.</summary>
        internal static SsjEvent Last;

        /// <summary>Attempts and landings since the level loaded, and the bucket histogram (index 5 = dt &lt;= 0).</summary>
        internal static int Attempts, Landed;
        internal static readonly int[] Histogram = new int[6];

        internal static void ResetCounters()
        {
            Attempts = 0;
            Landed = 0;
            for (int i = 0; i < Histogram.Length; i++) Histogram[i] = 0;
            Last = default(SsjEvent);
        }

        internal struct Snapshot
        {
            public double Dt;
            public int Bucket;
            public bool Landed;
            public float Speed, Horizontal;
        }

        private static float Horizontal(Vector3 v) => new Vector2(v.x, v.z).magnitude;

        [HarmonyPrefix]
        [HarmonyPatch(typeof(NewMovement), "TrySSJ")]
        private static void SsjPrefix(NewMovement __instance, float speedMultiplier, out Snapshot __state)
        {
            __state = default(Snapshot);
            if (__instance == null || __instance.rb == null) return;

            // Exactly the arithmetic TrySSJ is about to do, on the same two public timestamps.
            double dt = __instance.jumpTimestamp - __instance.slideTimestamp;
            int bucket = dt > 0.0 ? (int)(dt / PlayerFields.SsjFrame) : -1;
            float maxFrames = PlayerFields.SsjMaxFrames(__instance);
            __state.Dt = dt;
            __state.Bucket = bucket;
            __state.Landed = dt > 0.0 && bucket != 0 && bucket < maxFrames;
            var vel = __instance.rb.velocity;
            __state.Speed = vel.magnitude;
            __state.Horizontal = Horizontal(vel);
        }

        [HarmonyPostfix]
        [HarmonyPatch(typeof(NewMovement), "TrySSJ")]
        private static void SsjPostfix(NewMovement __instance, float speedMultiplier, Snapshot __state)
        {
            if (__instance == null || __instance.rb == null) return;
            var vel = __instance.rb.velocity;

            Attempts++;
            if (__state.Landed) Landed++;
            // Bucket -1 (no slide release before the jump) is counted in the last slot, not dropped: on a
            // plain jump it is the overwhelmingly common case and its count is how a macro's rate is judged.
            int slot = __state.Bucket < 0 ? 5 : Mathf.Clamp(__state.Bucket, 0, 4);
            Histogram[slot]++;

            Last = new SsjEvent
            {
                Frame = Time.frameCount,
                Dt = __state.Dt,
                Bucket = __state.Bucket,
                Landed = __state.Landed,
                Wall = speedMultiplier > 0.6f, // Jump passes 0.5, WallJump passes 0.75
                SpeedBefore = __state.Speed,
                SpeedAfter = vel.magnitude,
                HorizontalBefore = __state.Horizontal,
                HorizontalAfter = Horizontal(vel),
            };
        }

        [HarmonyPostfix]
        [HarmonyPatch(typeof(PrefsManager), nameof(PrefsManager.GetBool))]
        private static void OverridePrefBool(string key, ref bool __result)
        {
            if (!SsjIndicator || !EpisodeController.InControl) return;
            if (key == "ssjIndicator") __result = true;
        }

        /// <summary>
        /// The SSJ attempt that happened after <paramref name="sinceFrame"/>, or null. Used to attribute an
        /// SSJ to the step that asked for it rather than to a jump the policy happened to make earlier.
        /// </summary>
        internal static JObject BuildLast(int sinceFrame)
        {
            if (!InstrumentAvailable || Last.Frame <= sinceFrame) return null;
            return new JObject
            {
                ["frame"] = Last.Frame,
                ["dt"] = Last.Dt,
                ["dt_ms"] = Last.Dt * 1000.0,
                ["bucket"] = Last.Bucket,
                ["landed"] = Last.Landed,
                ["wall"] = Last.Wall,
                ["speed_before"] = Last.SpeedBefore,
                ["speed_after"] = Last.SpeedAfter,
                ["gain"] = Last.SpeedAfter - Last.SpeedBefore,
                ["h_speed_before"] = Last.HorizontalBefore,
                ["h_speed_after"] = Last.HorizontalAfter,
                ["h_gain"] = Last.HorizontalAfter - Last.HorizontalBefore,
            };
        }

        /// <summary>Cumulative counters, cheap enough to send every step once block A is on.</summary>
        internal static JObject BuildCounters()
        {
            if (!InstrumentAvailable) return null;
            var hist = new JArray();
            for (int i = 0; i < Histogram.Length; i++) hist.Add(new JValue(Histogram[i]));
            return new JObject
            {
                ["attempts"] = Attempts,
                ["landed"] = Landed,
                ["histogram"] = hist, // index 0..4 = bucket, index 5 = dt <= 0
            };
        }
    }
}
