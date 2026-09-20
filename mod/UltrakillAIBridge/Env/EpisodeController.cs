using System;
using System.Diagnostics;
using Newtonsoft.Json.Linq;
using UltrakillAIBridge.Act;
using UltrakillAIBridge.Net;
using UltrakillAIBridge.Obs;
using UnityEngine;
using UnityEngine.AddressableAssets;
using UnityEngine.InputSystem.LowLevel;

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

        // mod 0.8.0. The frame the current step began on, so an SSJ can be attributed to the step that asked
        // for it rather than to a jump the policy happened to make earlier.
        private int stepStartFrame;
        private JObject variantReport;

        // A rolling estimate of real seconds per game frame, on the same clock NewMovement's SSJ windows use.
        // Measured only between frames INSIDE a step, so Python's think time is never counted: the wall macro
        // needs to predict how much real time passes between queueing an event and the frame that reads it.
        private double lastFrameTime = -1.0;
        private double frameGap;

        /// <summary>Whether the `variant` action may switch the held weapon's variation. Refused by default.</summary>
        private bool variantSwitching;

        /// <summary>Adds the `input` block (timestamp cursor vs the live clock) to a step reply. Diagnostic.</summary>
        private bool reportInputClock;

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
            if (lastFrameTime > 0.0 && state == State.Stepping)
            {
                double delta = InputState.currentTime - lastFrameTime;
                if (delta > 0.0 && delta < 0.5)
                {
                    frameGap = frameGap <= 0.0 ? delta : frameGap * 0.9 + delta * 0.1;
                    injector.FrameGapEstimate = frameGap;
                }
            }
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
                        Send(BuildStepObs());
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
            finally
            {
                // Recorded AFTER any BlockForCommand above, so the next frame's delta is pure frame time.
                lastFrameTime = InputState.currentTime;
            }
        }

        /// <summary>
        /// The step reply: the ordinary observation, plus the `macro` and `variant` blocks when the client
        /// asked for either. A client that sends neither gets exactly the 0.7.2 message.
        /// </summary>
        private JObject BuildStepObs()
        {
            var obs = observer.Build(++step);
            var macro = injector.BuildMacroReport(stepStartFrame);
            if (macro != null) obs["macro"] = macro;
            if (variantReport != null) obs["variant"] = variantReport;
            if (reportInputClock)
            {
                // `cursor` ahead of `now` is the monotonic cursor holding a lifted timestamp: exactly the
                // case in which an un-lifted event would have been silently dropped by InputManager.OnUpdate.
                obs["input"] = new JObject
                {
                    ["cursor"] = injector.LastQueuedTime,
                    ["now"] = InputState.currentTime,
                    ["frame_gap"] = frameGap,
                };
            }
            return obs;
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
                        // Capability discovery. The protocol number stays 1 because every 0.8 addition is
                        // optional on both sides; this array is how a client tells a 0.8 DLL from a 0.7 one
                        // WITHOUT parsing the version string, and it is the check the S6 test gates on: if the
                        // Doorstop override were ignored, the private game would load the LIVE plugin and
                        // every result would be a false negative.
                        ["features"] = Features(),
                        ["diag"] = Diag(),
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
                    stepStartFrame = Time.frameCount;
                    var action = msg["action"] as JObject;
                    variantReport = ApplyVariant(action?["variant"]);
                    injector.SetAction(action, frameskip);
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

            // mod 0.8.0. Macros are available but a client only gets one by asking for it in an action, so
            // these defaults leave a 0.7.x client's behaviour untouched. `variant` and the reserved macro
            // values 3..5 are refused by default: that is what makes widening the action head at S7 exactly
            // behaviour-preserving, and enabling them later a config flip rather than a second break.
            injector.MacrosEnabled = msg["macros"]?.Value<bool>() ?? injector.MacrosEnabled;
            injector.AllowReservedMacros = msg["allow_reserved_macros"]?.Value<bool>() ?? injector.AllowReservedMacros;
            injector.AllowSsjWall = msg["macro_ssj_wall"]?.Value<bool>() ?? injector.AllowSsjWall;
            injector.AllowUnsafeWallLead = msg["macro_wall_lead_unsafe"]?.Value<bool>() ?? injector.AllowUnsafeWallLead;
            injector.SsjGapSeconds = msg["ssj_gap_s"]?.Value<double>() ?? injector.SsjGapSeconds;
            injector.WallLeadSeconds = msg["macro_wall_lead_s"]?.Value<double>() ?? injector.WallLeadSeconds;
            injector.WallLeadFrames = msg["macro_wall_lead_frames"]?.Value<double>() ?? injector.WallLeadFrames;
            variantSwitching = msg["variant_switching"]?.Value<bool>() ?? variantSwitching;
            reportInputClock = msg["obs_input_clock"]?.Value<bool>() ?? reportInputClock;
            MovementPatches.SsjIndicator = msg["ssj_indicator"]?.Value<bool>() ?? MovementPatches.SsjIndicator;

            observer.Configure(msg);

            if (HasControl) ApplyTimeSettings();
        }

        /// <summary>
        /// The optional `variant` action field: 0/absent keeps the held variation, 1..3 select variation 0..2
        /// of the CURRENT slot.
        ///
        /// This is the one place the mod may legitimately bypass the virtual-device pipeline. Weapon
        /// selection has no Harmony-inlining hazard -- the reason input goes through virtual devices at all is
        /// that Mono may inline `InputActionState`'s getters, which does not apply to a public method called
        /// directly -- and `SelectVariant1/2/3` are not reliably bound to a key, so there may be no key to
        /// press. `GunControl.SwitchWeapon(slot, variation)` is public and is exactly what the game's own
        /// key handler calls.
        ///
        /// Refused by default (`variant_switching` false). The variants worth holding on 0-1 are all
        /// variation 0, so refusing costs nothing and keeps the S7 migration behaviour-preserving: a live
        /// `variant` head at P(!= keep) = 0.15 would otherwise change the held weapon on 15 % of steps from
        /// the first rollout, an environment change the offline migration test cannot see.
        /// </summary>
        private JObject ApplyVariant(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null) return null;
            int requested = token.Type == JTokenType.Integer ? token.Value<int>() : 0;
            if (requested <= 0) return null;

            var report = new JObject { ["requested"] = requested };
            if (!variantSwitching)
            {
                report["result"] = "disabled";
                report["reason"] = "reserved";
                return report;
            }

            var gun = MonoSingleton<GunControl>.Instance;
            if (gun == null || gun.slots == null || gun.slots.Count == 0)
            {
                report["result"] = "refused";
                report["reason"] = "no_gun_control";
                return report;
            }

            int slotIndex = gun.currentSlotIndex;
            int i = slotIndex - 1;
            int available = i >= 0 && i < gun.slots.Count && gun.slots[i] != null ? gun.slots[i].Count : 0;
            int target = requested - 1;
            report["variations"] = available;
            if (target >= available)
            {
                report["result"] = "refused";
                report["reason"] = "no_such_variation";
                return report;
            }
            if (target == gun.currentVariationIndex)
            {
                // Switching to the variation already held would re-draw the weapon and clear Revolver.gunReady.
                report["result"] = "refused";
                report["reason"] = "already_held";
                return report;
            }

            gun.SwitchWeapon(slotIndex, target);
            report["result"] = "ran";
            report["variation"] = gun.currentVariationIndex;
            return report;
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
            variantReport = null;
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

        /// <summary>
        /// The handful of values the macro design was derived from but could not be READ from the decompiled
        /// C#, because they are serialized in the scene rather than assigned in code: `walkSpeed` and
        /// `fixedDeltaTime` set every u/s figure in the design, and `updateMode` decides whether a
        /// future-dated input event is processed in the current update or time-sliced into a later one.
        /// Reported once, on hello, so a test never has to assume them.
        /// </summary>
        private static JObject Diag()
        {
            var nm = MonoSingleton<NewMovement>.Instance;
            var obj = new JObject
            {
                ["update_mode"] = UnityEngine.InputSystem.InputSystem.settings != null
                    ? UnityEngine.InputSystem.InputSystem.settings.updateMode.ToString()
                    : null,
                ["fixed_delta_time"] = Time.fixedDeltaTime,
                ["capture_delta_time"] = Time.captureDeltaTime,
                ["input_now"] = InputState.currentTime,
                ["unity_time"] = Time.realtimeSinceStartup,
                ["ssj_instrument"] = MovementPatches.InstrumentAvailable,
            };
            if (nm != null)
            {
                obj["walk_speed"] = nm.walkSpeed;
                obj["jump_power"] = nm.jumpPower;
                obj["wall_jump_power"] = nm.wallJumpPower;
                obj["ssj_max_frames"] = PlayerFields.SsjMaxFrames(nm);
                // The bonus TrySSJ adds at bucket 1, in u/s: speedMultiplier * walkSpeed * 2.75 * 3 * fixedDeltaTime.
                obj["ssj_bonus_jump"] = 0.5f * nm.walkSpeed * 2.75f * 3f * Time.fixedDeltaTime;
                obj["ssj_bonus_wall"] = 0.75f * nm.walkSpeed * 2.75f * 3f * Time.fixedDeltaTime;
            }
            return obj;
        }

        /// <summary>What this build can do, for a client that wants to check rather than assume.</summary>
        private static JArray Features()
        {
            var arr = new JArray();
            arr.Add(new JValue("monotonic_input_clock"));
            arr.Add(new JValue("macro.ssj"));
            // Built and enforced, but reserved and refused unless `macro_ssj_wall` is set. Advertised so a
            // client can tell "this DLL has it" from "this DLL is 0.7.x"; whether it will RUN is config.
            arr.Add(new JValue("macro.ssj_wall"));
            arr.Add(new JValue("obs.move_tech"));
            arr.Add(new JValue("obs.weapon_tech"));
            arr.Add(new JValue("obs.projectiles"));
            arr.Add(new JValue("action.variant"));
            if (MovementPatches.InstrumentAvailable) arr.Add(new JValue("ssj_instrument"));
            return arr;
        }
    }
}
