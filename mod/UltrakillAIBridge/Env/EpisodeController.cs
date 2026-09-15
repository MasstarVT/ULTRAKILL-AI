using Newtonsoft.Json.Linq;
using UltrakillAIBridge.Act;
using UltrakillAIBridge.Net;
using UltrakillAIBridge.Obs;
using UnityEngine;

namespace UltrakillAIBridge.Env
{
    /// <summary>
    /// Runs the lockstep protocol. While the AI has control, the game advances exactly
    /// <see cref="frameskip"/> frames per step and then blocks the main thread until Python sends the
    /// next command. Time.captureDeltaTime fixes how much game time each frame covers, so a slow
    /// policy never costs reaction time and a fast machine trains faster than real time.
    ///
    /// Commands (newline-delimited JSON, see docs/protocol.md):
    ///   hello, config, get_obs  - answered immediately, never take control
    ///   reset                   - takes control, loads/restarts, replies with an obs once ready
    ///   step                    - takes control, runs one step, replies with an obs
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
        private int step;
        private int framesRemaining;

        // Config
        private int frameskip = 4;
        private float fixedFps = 60f;
        private bool unlimitedFps = true;
        private bool mute = true;
        private bool blockHumanInput = true;
        private int resetTimeoutFrames = 60 * 60;
        private int resetSettleFrames = 30;
        private int commandTimeoutMs = 300_000;

        // Reset bookkeeping
        private string resetScene;
        private int resetFrames;
        private int readyFrames;
        private bool sceneRequested;

        // Settings restored on release
        private int savedVSync, savedTargetFps;
        private float savedVolume;

        public EpisodeController(BridgeServer server)
        {
            this.server = server;
        }

        private bool HasControl => state != State.Idle;

        public void Tick()
        {
            switch (state)
            {
                case State.Idle:
                    DrainNonBlocking();
                    return;

                case State.Stepping:
                    injector.ApplyFrame();
                    if (--framesRemaining > 0) return;
                    Send(observer.Build(++step));
                    state = State.AwaitCommand;
                    BlockForCommand();
                    return;

                case State.Resetting:
                    injector.ApplyFrame();
                    TickReset();
                    if (state == State.AwaitCommand) BlockForCommand();
                    return;

                case State.AwaitCommand:
                    BlockForCommand();
                    return;
            }
        }

        private void DrainNonBlocking()
        {
            while (state == State.Idle && server.TryReceive(out var msg))
            {
                Handle(msg);
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
                Handle(incoming);
            }
        }

        private void Handle(BridgeServer.Incoming incoming)
        {
            if (incoming.ClientId != server.CurrentClientId && !incoming.Disconnected)
            {
                return; // Stale message from a previous connection.
            }

            if (incoming.Disconnected)
            {
                if (incoming.ClientId == activeClient || !server.Connected) ReleaseControl();
                return;
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
                    injector.SetAction(msg["action"] as JObject, frameskip);
                    framesRemaining = frameskip;
                    state = State.Stepping;
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
            resetTimeoutFrames = msg["reset_timeout_frames"]?.Value<int>() ?? resetTimeoutFrames;
            resetSettleFrames = msg["reset_settle_frames"]?.Value<int>() ?? resetSettleFrames;
            if (msg["command_timeout_s"] != null) commandTimeoutMs = Mathf.Max(1, msg["command_timeout_s"].Value<int>()) * 1000;
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

            injector.Attach(blockHumanInput);
            ApplyTimeSettings();
            state = State.AwaitCommand;
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
            AudioListener.volume = mute ? 0f : savedVolume;
        }

        public void ReleaseControl()
        {
            if (!HasControl) return;

            injector.Detach();
            Time.captureDeltaTime = 0f;
            QualitySettings.vSyncCount = savedVSync;
            Application.targetFrameRate = savedTargetFps;
            AudioListener.volume = savedVolume;
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
            resetFrames = 0;
            readyFrames = 0;
            sceneRequested = false;

            var sm = MonoSingleton<StatsManager>.Instance;
            if (checkpoint && resetScene == SceneHelper.CurrentScene && sm != null && !sm.infoSent)
            {
                MonoSingleton<OptionsManager>.Instance?.UnPause();
                sm.Restart();
                sceneRequested = true;
            }

            state = State.Resetting;
        }

        private void TickReset()
        {
            resetFrames++;

            if (!sceneRequested)
            {
                MonoSingleton<OptionsManager>.Instance?.UnPause();
                SceneHelper.LoadScene(resetScene);
                sceneRequested = true;
                return;
            }

            bool ready = SceneHelper.CurrentScene == resetScene && ObservationBuilder.PlayerReady();
            readyFrames = ready ? readyFrames + 1 : 0;

            if (readyFrames >= resetSettleFrames)
            {
                step = 0;
                injector.ResolveBindings();
                Send(observer.Build(step, "reset"));
                state = State.AwaitCommand;
            }
            else if (resetFrames > resetTimeoutFrames)
            {
                Send(Error($"reset to '{resetScene}' timed out"));
                state = State.AwaitCommand;
            }
        }

        private void Send(JObject obj) => server.Send(obj);

        private static JObject Error(string message) => new JObject { ["type"] = "error", ["message"] = message };
    }
}
