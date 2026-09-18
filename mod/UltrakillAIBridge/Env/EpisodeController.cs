using System;
using System.Diagnostics;
using Newtonsoft.Json.Linq;
using UltrakillAIBridge.Act;
using UltrakillAIBridge.Net;
using UltrakillAIBridge.Obs;
using UnityEngine;
using UnityEngine.AddressableAssets;

namespace UltrakillAIBridge.Env
{
    /// <summary>
    /// Runs the lockstep protocol at the end of each frame. While the AI has control, the game advances
    /// exactly <see cref="frameskip"/> frames per step, then sends an observation and blocks the main
    /// thread until Python sends the next command. Time.captureDeltaTime fixes how much game time each
    /// frame covers, so a slow policy never costs reaction time and a fast machine trains faster than
    /// real time.
    ///
    /// Frame timeline for a step received at the end of frame N: input for frame N+1 is queued and its
    /// look applied immediately; frames N+1..N+frameskip run with the action; the observation is built at
    /// the end of frame N+frameskip.
    ///
    /// Commands (newline-delimited JSON, see docs/protocol.md):
    ///   hello, config, get_obs  - answered immediately, never take control
    ///   reset                   - takes control, loads/restarts, replies with an obs once ready
    ///   step                    - takes control, runs one step, replies with an obs
    ///   teleport, kill          - take control, reply with an obs straight away (kill is a debug command)
    ///   release                 - gives control back to the human
    /// </summary>
    public sealed class EpisodeController
    {
        private enum State { Idle, AwaitCommand, Stepping, Resetting }

        private readonly BridgeServer server;
        private readonly ActionInjector injector = new ActionInjector();
        private readonly ObservationBuilder observer = new ObservationBuilder();

        private State state = State.Idle;
        private int activeClient = -1;
        private int replyClient = -1; // connection the next reply belongs to
        private int step;
        private int framesRemaining;

        // Config
        private int frameskip = 4;
        private float fixedFps = 60f;
        private bool unlimitedFps = true;
        private bool mute = true;
        private bool blockHumanInput = true;
        private float resetTimeoutSeconds = 120f;
        private int resetSettleFrames = 10;
        private int commandTimeoutMs = 300_000;
        private bool windowed = true;
        private int windowWidth = 640;
        private int windowHeight = 360;

        // Reset bookkeeping
        private string resetScene;
        private readonly Stopwatch resetTimer = new Stopwatch();
        private int readyFrames;
        private bool sceneRequested;
        // The asset sweep that runs once per reset; see SweepAssetsDone. Default ON for training instances:
        // the leak it answers is what took the machine down, and a human's game never reaches it because the
        // sweep is also gated on the AI having control.
        private bool unloadAssetsOnReset = true;
        private AsyncOperation sweep;

        // Settings restored on release
        private int savedVSync, savedTargetFps;
        private float savedVolume;
        private int savedWidth, savedHeight;
        private FullScreenMode savedScreenMode;
        private bool displayChanged;

        public EpisodeController(BridgeServer server)
        {
            this.server = server;
        }

        private bool HasControl => state != State.Idle;

        /// <summary>True while the AI has control. Read by the Harmony patches.</summary>
        internal static bool InControl { get; private set; }
        private static bool muteInControl = true;

        /// <summary>Keeps the cursor free and the game silent while the AI plays in the background.</summary>
        internal static void ApplyCursorAndAudio()
        {
            if (Cursor.lockState != CursorLockMode.None) Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
            if (muteInControl) AudioListener.volume = 0f;
        }

        public void EndOfFrame()
        {
            try
            {
                if (HasControl) ApplyCursorAndAudio();

                if (HasControl && server.CurrentClientId != activeClient)
                {
                    // Controlling client disconnected mid-step or mid-reset; never answer a newer client with its obs.
                    ReleaseControl();
                }

                switch (state)
                {
                    case State.Idle:
                        DrainNonBlocking();
                        break;

                    case State.Stepping:
                        replyClient = activeClient;
                        if (--framesRemaining > 0)
                        {
                            injector.ApplyFrame();
                            break;
                        }
                        Send(observer.Build(++step));
                        state = State.AwaitCommand;
                        BlockForCommand();
                        break;

                    case State.Resetting:
                        replyClient = activeClient;
                        injector.ApplyFrame();
                        TickReset();
                        if (state == State.AwaitCommand) BlockForCommand();
                        break;

                    case State.AwaitCommand:
                        BlockForCommand();
                        break;
                }
            }
            catch (Exception e)
            {
                Plugin.Log.LogError($"Bridge error: {e}");
                if (HasControl)
                {
                    Send(Error(e.Message));
                    state = State.AwaitCommand;
                }
            }
        }

        private void DrainNonBlocking()
        {
            while (state == State.Idle && server.TryReceive(out var msg))
            {
                HandleSafely(msg);
            }
        }

        private void BlockForCommand()
        {
            while (state == State.AwaitCommand)
            {
                // The panic key can't be read while the main thread is blocked, so a client that
                // stops responding without closing the socket is dropped after a timeout instead.
                var incoming = server.WaitReceive(commandTimeoutMs);
                if (incoming == null)
                {
                    Plugin.Log.LogWarning($"No command for {commandTimeoutMs / 1000}s, dropping client");
                    server.DropClient();
                    ReleaseControl();
                    return;
                }
                HandleSafely(incoming);
            }
        }

        private void HandleSafely(BridgeServer.Incoming incoming)
        {
            replyClient = incoming.ClientId;
            try
            {
                Handle(incoming);
            }
            catch (Exception e)
            {
                Plugin.Log.LogError($"Error handling {(string)incoming.Message?["type"]}: {e}");
                Send(Error(e.Message));
                if (HasControl) state = State.AwaitCommand;
            }
        }

        private void Handle(BridgeServer.Incoming incoming)
        {
            if (incoming.Disconnected)
            {
                if (incoming.ClientId == activeClient || !server.Connected) ReleaseControl();
                return;
            }
            if (incoming.ClientId != server.CurrentClientId)
            {
                return; // Stale message from a previous connection.
            }

            var msg = incoming.Message;
            var type = (string)msg["type"];
            switch (type)
            {
                case "hello":
                    Send(new JObject
                    {
                        ["type"] = "hello",
                        ["protocol"] = Plugin.ProtocolVersion,
                        ["mod_version"] = Plugin.Version,
                        ["port"] = Plugin.ListenPort,
                        ["training_instance"] = Plugin.IsTrainingInstance,
                        // Whether the skip is actually in place, not merely asked for: -aibridge-nosteam with
                        // Steam already initialised leaves this false, which is the only in-band way to tell.
                        ["steam_hidden"] = SteamPatches.Hidden,
                        ["scene"] = SceneHelper.CurrentScene,
                    });
                    break;

                case "config":
                    ApplyConfig(msg);
                    Send(new JObject { ["type"] = "ok" });
                    break;

                case "get_obs":
                    Send(observer.Build(step));
                    break;

                case "reset":
                    TakeControl(incoming.ClientId);
                    BeginReset(msg);
                    break;

                case "step":
                    TakeControl(incoming.ClientId);
                    ApplyTimeSettings(); // Scene loads and menus can re-enable vsync or a frame cap.
                    injector.SetAction(msg["action"] as JObject, frameskip);
                    injector.ApplyFrame();
                    framesRemaining = frameskip;
                    state = State.Stepping;
                    break;

                case "teleport":
                    TakeControl(incoming.ClientId);
                    Teleport(msg["pos"] as JArray);
                    Send(observer.Build(step, "teleport"));
                    break;

                case "kill":
                    TakeControl(incoming.ClientId);
                    Kill();
                    Send(observer.Build(step, "kill"));
                    break;

                case "release":
                    ReleaseControl();
                    Send(new JObject { ["type"] = "ok" });
                    break;

                default:
                    Send(Error($"unknown message type '{type}'"));
                    break;
            }
        }

        private void ApplyConfig(JObject msg)
        {
            frameskip = Mathf.Max(1, msg["frameskip"]?.Value<int>() ?? frameskip);
            fixedFps = msg["fixed_fps"]?.Value<float>() ?? fixedFps;
            unlimitedFps = msg["unlimited_fps"]?.Value<bool>() ?? unlimitedFps;
            mute = msg["mute"]?.Value<bool>() ?? mute;
            blockHumanInput = msg["block_human_input"]?.Value<bool>() ?? blockHumanInput;
            resetTimeoutSeconds = msg["reset_timeout_s"]?.Value<float>() ?? resetTimeoutSeconds;
            resetSettleFrames = msg["reset_settle_frames"]?.Value<int>() ?? resetSettleFrames;
            if (msg["command_timeout_s"] != null) commandTimeoutMs = Mathf.Max(1, msg["command_timeout_s"].Value<int>()) * 1000;
            windowed = msg["windowed"]?.Value<bool>() ?? windowed;
            TrainingSpeed.SoftDeathEnabled = msg["soft_death"]?.Value<bool>() ?? TrainingSpeed.SoftDeathEnabled;
            unloadAssetsOnReset = msg["unload_assets_on_reset"]?.Value<bool>() ?? unloadAssetsOnReset;
            UnwedgePatch.Enabled = msg["unwedge"]?.Value<bool>() ?? UnwedgePatch.Enabled;
            UnwedgePatch.HoldFrames = Mathf.Max(1, msg["unwedge_frames"]?.Value<int>() ?? UnwedgePatch.HoldFrames);
            // PrefsManager's own validator clamps a stored difficulty above 4 down to 4, but the Harmony
            // postfix overwrites __result AFTER that validator runs, so an out-of-range override would
            // reach call sites that index arrays directly by the value. Clamp here instead; -1 stays -1.
            CampaignPatches.DifficultyOverride = Mathf.Clamp(msg["difficulty"]?.Value<int>() ?? CampaignPatches.DifficultyOverride, -1, 4);
            CampaignPatches.UnlockAllGear = msg["unlock_all_gear"]?.Value<bool>() ?? CampaignPatches.UnlockAllGear;
            if (msg["render"] != null)
            {
                TrainingSpeed.RenderingDisabled = !msg["render"].Value<bool>();
                if (!TrainingSpeed.RenderingDisabled) TrainingSpeed.RestoreRendering();
            }
            windowWidth = Mathf.Max(160, msg["window_width"]?.Value<int>() ?? windowWidth);
            windowHeight = Mathf.Max(90, msg["window_height"]?.Value<int>() ?? windowHeight);
            observer.Configure(msg);

            if (HasControl) ApplyTimeSettings();
        }

        private void TakeControl(int clientId)
        {
            activeClient = clientId;
            if (HasControl) return;

            savedVSync = QualitySettings.vSyncCount;
            savedTargetFps = Application.targetFrameRate;
            savedVolume = AudioListener.volume;
            savedWidth = Screen.width;
            savedHeight = Screen.height;
            savedScreenMode = Screen.fullScreenMode;
            displayChanged = false;

            injector.Attach(blockHumanInput);
            state = State.AwaitCommand;
            InControl = true;
            ApplyTimeSettings();
            Plugin.Log.LogInfo("AI took control");
        }

        private void ApplyTimeSettings()
        {
            Time.captureDeltaTime = fixedFps > 0f ? 1f / fixedFps : 0f;
            if (unlimitedFps)
            {
                QualitySettings.vSyncCount = 0;
                Application.targetFrameRate = -1;
            }
            muteInControl = mute;
            AudioListener.volume = mute ? 0f : savedVolume;

            if (HasControl && windowed && (Screen.fullScreenMode != FullScreenMode.Windowed || Screen.width != windowWidth || Screen.height != windowHeight))
            {
                // A small window renders faster and doesn't hold the mouse or the screen.
                Screen.SetResolution(windowWidth, windowHeight, FullScreenMode.Windowed);
                displayChanged = true;
            }
            if (HasControl)
            {
                ApplyCursorAndAudio();
                TrainingSpeed.ApplyRendering(); // cameras come back with every scene load
            }
        }

        public void ReleaseControl()
        {
            if (!HasControl) return;

            injector.Detach();
            Time.captureDeltaTime = 0f;
            QualitySettings.vSyncCount = savedVSync;
            Application.targetFrameRate = savedTargetFps;
            AudioListener.volume = savedVolume;
            if (displayChanged)
            {
                Screen.SetResolution(savedWidth, savedHeight, savedScreenMode);
                displayChanged = false;
            }
            TrainingSpeed.RestoreRendering();
            InControl = false;
            state = State.Idle;
            activeClient = -1;
            Plugin.Log.LogInfo("AI released control");
        }

        /// <summary>
        /// reset message: {"type":"reset", "scene":"Endless" | "Level 0-1", "checkpoint":false}
        /// scene omitted or equal to the current scene restarts it. checkpoint=true respawns at the
        /// last checkpoint instead of reloading (campaign only).
        /// </summary>
        private void BeginReset(JObject msg)
        {
            injector.Clear();
            var scene = (string)msg["scene"];
            bool checkpoint = msg["checkpoint"]?.Value<bool>() ?? false;

            resetScene = string.IsNullOrEmpty(scene) ? SceneHelper.CurrentScene : scene;
            if (!SceneExists(resetScene))
            {
                // Loading an unknown scene would leave SceneHelper stuck with a pending load until restart.
                Send(Error($"unknown scene '{resetScene}'"));
                return;
            }

            resetTimer.Restart();
            readyFrames = 0;
            sceneRequested = false;
            // A reset that timed out mid-sweep would otherwise leave a FINISHED operation here, and the next
            // reset would see isDone and skip its own sweep. One reset, one sweep.
            sweep = null;
            ApplyTimeSettings();

            var sm = MonoSingleton<StatsManager>.Instance;
            if (checkpoint && resetScene == SceneHelper.CurrentScene && sm != null && !sm.infoSent)
            {
                UnpauseIfNeeded();
                CampaignPatches.BridgeRestart = true; // CampaignPatches blocks every other Restart while in control
                try
                {
                    sm.Restart();
                }
                finally
                {
                    CampaignPatches.BridgeRestart = false;
                }
                sceneRequested = true;
            }

            state = State.Resetting;
        }

        /// <summary>
        /// Unloads assets the newly loaded level no longer references, once per reset, before the reply.
        ///
        /// **Why this is the mod's business at all.** ULTRAKILL never sweeps: a grep over all 1178 decompiled
        /// files finds `Resources.UnloadUnusedAssets` in exactly one, the sandbox saver, and nothing on the
        /// level-change path. `SceneHelper.LoadSceneCoroutine` also discards its `Addressables.LoadSceneAsync`
        /// handle, and `SetUpFootstepPhysicsScene` rebuilds a parallel physics scene per load, duplicating
        /// every qualifying MeshCollider into native PhysX memory. A human loads 10-20 scenes in a session and
        /// never notices; a training instance loads 40-150 and was measured growing 1.2-1.4 GB an HOUR, from
        /// 1.2 GB at boot to the 5.8-6.1 GB that took this machine down on 2026-09-18.
        ///
        /// Returns false while the sweep is still running, so `TickReset` keeps ticking and the 100 ms - 2 s
        /// stall lands inside a reset the client is already blocked on rather than in the middle of a rollout.
        /// `resetTimeoutSeconds` still bounds the whole reset, so a sweep that somehow never finished degrades
        /// into the ordinary reset timeout rather than a hang.
        ///
        /// Gated on `unloadAssetsOnReset` AND on the AI having control, so a human's game is never touched.
        /// </summary>
        private bool SweepAssetsDone()
        {
            if (!unloadAssetsOnReset || !InControl) return true;
            if (sweep == null)
            {
                sweep = Resources.UnloadUnusedAssets();
                return false;
            }
            if (!sweep.isDone) return false;
            sweep = null;
            // Boehm is non-moving, so this does not compact -- it returns the managed side of what the
            // unload just orphaned, which is what keeps the heap's high-water mark from ratcheting.
            System.GC.Collect();
            return true;
        }

        private void TickReset()
        {
            if (!sceneRequested)
            {
                if (!string.IsNullOrEmpty(SceneHelper.PendingScene)) return; // Wait for an in-progress load.
                UnpauseIfNeeded();
                SceneHelper.LoadScene(resetScene);
                sceneRequested = true;
                return;
            }

            bool ready = SceneHelper.CurrentScene == resetScene && ObservationBuilder.PlayerReady();
            readyFrames = ready ? readyFrames + 1 : 0;

            if (readyFrames >= resetSettleFrames)
            {
                if (!SweepAssetsDone()) return;  // pay the sweep inside the reset Python is already waiting on
                step = 0;
                injector.ResolveBindings();
                ApplyTimeSettings();
                Send(observer.Build(step, "reset"));
                state = State.AwaitCommand;
            }
            else if (resetTimer.Elapsed.TotalSeconds > resetTimeoutSeconds)
            {
                Send(Error($"reset to '{resetScene}' timed out"));
                state = State.AwaitCommand;
            }
        }

        /// <summary>Moves the player to a world position and stops them (used to skip walking into the Cyber Grind arena).</summary>
        private static void Teleport(JArray pos)
        {
            var nm = MonoSingleton<NewMovement>.Instance;
            if (nm == null || pos == null || pos.Count < 3) throw new ArgumentException("teleport needs a player and pos [x,y,z]");
            var target = new Vector3((float)pos[0], (float)pos[1], (float)pos[2]);
            nm.transform.position = target;
            if (nm.rb != null)
            {
                nm.rb.position = target;
                nm.rb.velocity = Vector3.zero;
            }
            Physics.SyncTransforms();
        }

        /// <summary>
        /// Debug command for the in-game death check: a lethal hit through the normal damage path. With
        /// soft_death on, TrainingSpeed heals it instead and counts a soft death. GetHurt ignores it once the
        /// level is over.
        /// </summary>
        private static void Kill()
        {
            var nm = MonoSingleton<NewMovement>.Instance;
            if (nm == null || nm.dead) throw new InvalidOperationException("kill needs a living player");
            nm.GetHurt(999, invincible: false, ignoreInvincibility: true);
        }

        private static void UnpauseIfNeeded()
        {
            var om = MonoSingleton<OptionsManager>.Instance;
            if (om != null && om.paused && !om.mainMenu && MonoSingleton<NewMovement>.Instance != null)
            {
                om.UnPause();
            }
        }

        private static bool SceneExists(string scene)
        {
            if (scene == SceneHelper.CurrentScene) return true;
            foreach (var locator in Addressables.ResourceLocators)
            {
                if (locator.Locate(scene, null, out var locations) && locations.Count > 0) return true;
            }
            return false;
        }

        private void Send(JObject obj) => server.Send(obj, replyClient);

        private static JObject Error(string message) => new JObject { ["type"] = "error", ["message"] = message };
    }
}
