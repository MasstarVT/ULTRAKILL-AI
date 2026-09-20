using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UltrakillAIBridge.Env;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.LowLevel;

namespace UltrakillAIBridge.Act
{
    /// <summary>The macro the client asked for. Values are the spec's `macro` action dimension, 0..5.</summary>
    internal enum MacroKind
    {
        None = 0,
        Ssj = 1,
        SsjWall = 2,
        // Reserved headroom. These are ordinary multi-decision sequences, not sub-decision timing, and the
        // two that must land a hitscan on a moving object could only be macroed by AIMING, which the design
        // forbids. The mod refuses them so the action space can be widened once and enabled later by config.
        CoreNuke = 3,
        RocketDown = 4,
        CoinRocket = 5,
    }

    /// <summary>
    /// Drives the player through a virtual keyboard and mouse registered with the Unity Input System,
    /// so the game's own input pipeline (InputActionState: IsPressed, WasPerformedThisFrame, ReadValue)
    /// behaves exactly as it does for a human. Keys are resolved from the player's current bindings.
    /// Camera look bypasses the mouse and rotates CameraController directly, so look actions are in
    /// degrees and independent of mouse sensitivity.
    ///
    /// ## Timestamped events and the monotonic cursor (mod 0.8.0)
    ///
    /// Every queued event carries a timestamp on the same clock `NewMovement` compares against
    /// (`InputState.currentTime`, sourced from the native wall clock -- `Time.captureDeltaTime` does not touch
    /// it). Two rules:
    ///
    /// - When nothing asks for a particular timestamp, the event is queued exactly as before, with
    ///   `time = -1`, letting the runtime stamp it. That is the legacy call, byte for byte.
    /// - Timestamps are forced **monotonic across both devices**. `InputManager.OnUpdate` silently discards a
    ///   state event whose timestamp precedes the device's last update time, and `Keyboard` has no state
    ///   callbacks, so the drop is unconditional. After a macro has queued an event at `now + 0.012`, the next
    ///   frame's ordinary event can easily carry a LOWER timestamp on a fast machine and be dropped, eating a
    ///   whole frame of the agent's input. The cursor clamps it to `lastQueued + 0.5 ms` instead. Without a
    ///   macro in play the clamp never binds, because the clock is already monotonic.
    ///
    /// ## Macros
    ///
    /// A macro is a short input script the mod plays across the frames of ONE step, so the client can express
    /// a timing a 15 Hz decision loop cannot reach. It never aims and never moves the camera: the client keeps
    /// look, move and every other button for the whole step. A macro step consumes exactly `frameskip` game
    /// frames -- the same as any other step -- so nothing about time accounting changes.
    /// </summary>
    public sealed class ActionInjector
    {
        // Held for the whole step.
        private static readonly string[] HoldButtons = { "fire1", "fire2", "slide", "hook" };
        // Pressed on the first frame of the step only, so repeated presses register as new presses.
        private static readonly string[] TapButtons = { "jump", "dash", "punch", "change_fist" };

        private readonly Dictionary<string, List<ButtonControl>> buttons = new Dictionary<string, List<ButtonControl>>();
        private readonly List<ButtonControl>[] slots = new List<ButtonControl>[7];
        private readonly Dictionary<string, List<ButtonControl>> moveParts = new Dictionary<string, List<ButtonControl>>();
        private readonly List<InputDevice> disabledDevices = new List<InputDevice>();

        private Keyboard keyboard;
        private Mouse mouse;
        private InputSettings.BackgroundBehavior previousBackgroundBehavior;

        private float moveX, moveY, yawPerFrame, pitchPerFrame;
        private readonly HashSet<string> held = new HashSet<string>();
        private readonly HashSet<string> tapped = new HashSet<string>();
        private int slot;
        private bool firstFrame;
        private bool tapsDownLastQueue;

        public bool Attached => keyboard != null;

        // ---- Timestamps -------------------------------------------------------------------------------

        /// <summary>Smallest gap the cursor leaves between two queued events. Well under one SSJ bucket (8 ms).</summary>
        private const double TimeEpsilon = 0.0005;

        private double lastQueuedTime = -1.0;

        /// <summary>The last timestamp actually queued, for reporting. Negative before the first event.</summary>
        internal double LastQueuedTime => lastQueuedTime;

        // ---- Macro configuration ---------------------------------------------------------------------

        /// <summary>Master switch. False makes every macro report `disabled`, which is the pre-0.8 behaviour.</summary>
        internal bool MacrosEnabled = true;

        /// <summary>Macro values 3..5 stay refused until their observation blocks exist. See <see cref="MacroKind"/>.</summary>
        internal bool AllowReservedMacros;

        /// <summary>
        /// Gap queued between the slide release and the jump press. TrySSJ buckets it as
        /// `(int)(gap / 0.008)` and accepts 1..3, so 0.012 is the middle of bucket 1 -- the bucket with the
        /// strongest multiplier for BOTH call sites (Jump's `1/2^(f-1)` and WallJump's `(4-f+1)/4` are each 1.0
        /// at f = 1). Configurable so a test can sweep the buckets.
        /// </summary>
        internal double SsjGapSeconds = 0.012;

        /// <summary>
        /// How far the wall macro's slide release is dated into the future, in seconds. Negative = adaptive.
        ///
        /// `WallJump` only takes its SSJ branch when `InputState.currentTime - slideTimestamp &lt; 0.032` at the
        /// moment it runs, and that is REAL time, not game time. The release is queued at the end of frame N
        /// and the jump it is measured against is not read until frame N+2's Update, so the interval that has
        /// to fit inside 32 ms is about TWO real frames -- tens of milliseconds on a loaded 12-game fleet.
        /// Dating the release forward by that much, minus the bucket gap, puts the frame N+2 read back inside
        /// the grace. **This is the least certain part of the macro design and the reason S6 exists.**
        /// </summary>
        internal double WallLeadSeconds = -1.0;

        /// <summary>Frames of lead the adaptive path assumes; see <see cref="WallLeadSeconds"/>. Sweepable by a test.</summary>
        internal double WallLeadFrames = 2.0;

        /// <summary>Recent real seconds per game frame, measured by EpisodeController, used by the adaptive lead.</summary>
        internal double FrameGapEstimate;

        // ---- Macro state ------------------------------------------------------------------------------

        private MacroKind macroRequested;
        private MacroKind macroKind;          // None unless a script is actually playing
        private bool macroRunning;
        private bool macroSuppressSlide;
        private string macroResult;           // ran | refused | degraded | disabled
        private string macroReason;
        private double macroAnchor;           // the slide-release timestamp the jump is measured from
        private double macroLead;
        private int macroScriptFrames;
        private int frameIndex;               // 1-based ApplyFrame call within the step
        private int stepFrames;

        public void Attach(bool blockHumanInput)
        {
            if (Attached) return;

            previousBackgroundBehavior = InputSystem.settings.backgroundBehavior;
            InputSystem.settings.backgroundBehavior = InputSettings.BackgroundBehavior.IgnoreFocus;
            Application.runInBackground = true;

            if (blockHumanInput)
            {
                foreach (var device in InputSystem.devices)
                {
                    if (device.enabled && (device is Keyboard || device is Mouse || device is Gamepad))
                    {
                        InputSystem.DisableDevice(device);
                        disabledDevices.Add(device);
                    }
                }
            }

            keyboard = InputSystem.AddDevice<Keyboard>("UltrakillAI Keyboard");
            mouse = InputSystem.AddDevice<Mouse>("UltrakillAI Mouse");
            lastQueuedTime = -1.0; // fresh devices, no history to stay ahead of
            ResolveBindings();
            Clear();
        }

        public void Detach()
        {
            if (!Attached) return;

            Clear();
            // ResetDevice fires the cancel callbacks the game's InputActionState relies on. Removing a
            // device with keys still down would leave actions like Fire1 stuck pressed.
            InputSystem.ResetDevice(keyboard);
            InputSystem.ResetDevice(mouse);
            InputSystem.RemoveDevice(keyboard);
            InputSystem.RemoveDevice(mouse);
            keyboard = null;
            mouse = null;
            lastQueuedTime = -1.0;

            foreach (var device in disabledDevices)
            {
                if (device.added) InputSystem.EnableDevice(device);
            }
            disabledDevices.Clear();
            InputSystem.settings.backgroundBehavior = previousBackgroundBehavior;
        }

        /// <summary>Maps game actions to controls on the virtual devices using the current key bindings.</summary>
        public void ResolveBindings()
        {
            if (!Attached) return;
            buttons.Clear();
            moveParts.Clear();

            var input = MonoSingleton<InputManager>.Instance?.InputSource;
            if (input == null)
            {
                Plugin.Log.LogWarning("InputManager not ready, cannot resolve bindings yet");
                return;
            }

            AddButton("fire1", input.Fire1);
            AddButton("fire2", input.Fire2);
            AddButton("slide", input.Slide);
            AddButton("hook", input.Hook);
            AddButton("jump", input.Jump);
            AddButton("dash", input.Dodge);
            AddButton("punch", input.Punch);
            AddButton("change_fist", input.ChangeFist);

            var slotStates = new[] { input.Slot1, input.Slot2, input.Slot3, input.Slot4, input.Slot5, input.Slot6 };
            for (int i = 0; i < slotStates.Length; i++)
            {
                slots[i + 1] = Resolve(slotStates[i].Action);
            }

            foreach (var binding in input.Move.Action.bindings)
            {
                if (!binding.isPartOfComposite) continue;
                var control = FindButton(binding.effectivePath);
                if (control == null) continue;
                var part = binding.name.ToLowerInvariant();
                if (!moveParts.TryGetValue(part, out var list)) moveParts[part] = list = new List<ButtonControl>();
                list.Add(control);
            }

            foreach (var kv in buttons)
            {
                if (kv.Value.Count == 0) Plugin.Log.LogWarning($"No keyboard/mouse binding found for '{kv.Key}'");
            }
            if (!moveParts.ContainsKey("up")) Plugin.Log.LogWarning("No keyboard binding found for movement");
        }

        private void AddButton(string name, InputActionState state)
        {
            buttons[name] = Resolve(state.Action);
        }

        private List<ButtonControl> Resolve(InputAction action)
        {
            var list = new List<ButtonControl>();
            foreach (var binding in action.bindings)
            {
                if (binding.isComposite || binding.isPartOfComposite) continue;
                var control = FindButton(binding.effectivePath);
                if (control != null) list.Add(control);
            }
            return list;
        }

        private ButtonControl FindButton(string path)
        {
            if (string.IsNullOrEmpty(path)) return null;
            return (InputControlPath.TryFindControl(keyboard, path) ?? InputControlPath.TryFindControl(mouse, path)) as ButtonControl;
        }

        /// <summary>
        /// Sets the action for the next step. Expected shape:
        /// {"move":[x,y], "look":[yaw_deg,pitch_deg], "buttons":["fire1","jump",...], "slot":0..6, "macro":0..5}
        /// Look degrees are spread evenly over the step's frames. `macro` is optional and defaults to none,
        /// so a client that never sends it gets exactly the pre-0.8 behaviour.
        /// </summary>
        public void SetAction(JObject action, int frames)
        {
            Clear();
            stepFrames = Mathf.Max(1, frames);
            if (action == null) return;

            if (action["move"] is JArray move && move.Count >= 2)
            {
                moveX = Mathf.Clamp((float)move[0], -1f, 1f);
                moveY = Mathf.Clamp((float)move[1], -1f, 1f);
            }

            if (action["look"] is JArray look && look.Count >= 2)
            {
                int n = Mathf.Max(1, frames);
                yawPerFrame = (float)look[0] / n;
                pitchPerFrame = (float)look[1] / n;
            }

            if (action["buttons"] is JArray pressed)
            {
                foreach (var token in pressed)
                {
                    var name = (string)token;
                    if (System.Array.IndexOf(HoldButtons, name) >= 0) held.Add(name);
                    else if (System.Array.IndexOf(TapButtons, name) >= 0) tapped.Add(name);
                }
            }

            slot = action["slot"]?.Type == JTokenType.Integer ? action["slot"].Value<int>() : 0;
            firstFrame = true;
            BeginMacro(ParseMacro(action["macro"]), stepFrames);
        }

        private static MacroKind ParseMacro(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null) return MacroKind.None;
            if (token.Type == JTokenType.Integer)
            {
                int v = token.Value<int>();
                return v >= 0 && v <= 5 ? (MacroKind)v : MacroKind.None;
            }
            switch ((token.Value<string>() ?? string.Empty).ToLowerInvariant())
            {
                case "": case "none": return MacroKind.None;
                case "ssj": return MacroKind.Ssj;
                case "ssj_wall": return MacroKind.SsjWall;
                case "core_nuke": return MacroKind.CoreNuke;
                case "rocket_down": return MacroKind.RocketDown;
                case "coin_rocket": return MacroKind.CoinRocket;
                default: return MacroKind.None;
            }
        }

        public void Clear()
        {
            moveX = moveY = yawPerFrame = pitchPerFrame = 0f;
            held.Clear();
            tapped.Clear();
            slot = 0;
            firstFrame = false;
            frameIndex = 0;
            macroRequested = MacroKind.None;
            macroKind = MacroKind.None;
            macroRunning = false;
            macroSuppressSlide = false;
            macroResult = null;
            macroReason = null;
            macroAnchor = 0.0;
            macroLead = 0.0;
            macroScriptFrames = 0;
        }

        // ---- Macro preconditions ----------------------------------------------------------------------

        private void BeginMacro(MacroKind kind, int frames)
        {
            macroRequested = kind;
            if (kind == MacroKind.None) return;

            if (!MacrosEnabled)
            {
                macroResult = "disabled";
                macroReason = "macros_disabled";
                return;
            }
            if (kind >= MacroKind.CoreNuke)
            {
                // Reserved headroom: the action row exists so the space is widened once, but the mod refuses
                // it, which is what makes the S7 migration exactly behaviour-preserving.
                macroResult = "disabled";
                macroReason = AllowReservedMacros ? "not_implemented" : "reserved";
                return;
            }

            var why = CheckPreconditions(kind, frames);
            if (why != null)
            {
                // A refusal runs the plain action and costs nothing extra. It is reported, never charged:
                // charging it would teach a policy to avoid macros rather than to learn their preconditions.
                macroResult = "refused";
                macroReason = why;
                return;
            }

            macroKind = kind;
            macroRunning = true;
            macroSuppressSlide = true;
            macroResult = "ran";
            macroScriptFrames = kind == MacroKind.SsjWall ? 2 : 1;
        }

        /// <summary>
        /// The state that must hold for the macro's script to reach TrySSJ at all, read at the moment the
        /// step command arrives. Each returned string is a refusal reason the client can histogram: a macro
        /// refused 99 % of the time is a design bug, and the reason says which one.
        /// </summary>
        private static string CheckPreconditions(MacroKind kind, int frames)
        {
            var nm = MonoSingleton<NewMovement>.Instance;
            if (nm == null || !nm.activated || nm.dead) return "no_player";
            if (nm.gc == null) return "no_ground_check";
            if (PlayerFields.JumpCooldown(nm)) return "jump_cooldown";

            if (kind == MacroKind.Ssj)
            {
                // SlideCancelled only records slideTimestamp `if (sliding)`, and Jump()'s `if (sliding)` branch
                // is what calls StopSlide() and so refreshes the velocityAfterSlide that TrySSJ overwrites
                // velocity WITH. Without an active slide the macro would land on a stale vector.
                if (!nm.sliding) return "not_sliding";
                if (!(nm.gc.onGround || nm.gc.canJump)) return "airborne";
                return null;
            }

            // MacroKind.SsjWall
            if (frames < 2) return "frameskip_lt_2";
            if (nm.gc.onGround) return "grounded";
            // HandleInputs' FIRST jump block wins whenever `!falling`, or whenever the player is airborne with
            // coyote time or an enemy under the feet. It calls Jump(), sets jumpCooldown, and the WallJump
            // block below it is then skipped entirely -- so these are refusals, not degradations.
            if (!nm.falling) return "not_falling";
            if (nm.gc.canJump) return "coyote";
            if (nm.currentWallJumps >= 3) return "wall_jumps_spent";
            var wc = PlayerFields.WallChecks(nm);
            if (wc == null) return "no_wall_check_group";
            if (wc.CheckForEnemyCols()) return "enemy_step";
            if (!wc.TryGetActiveInstance(out _)) return "no_wall";
            return null;
        }

        /// <summary>Re-checked on the wall macro's second frame; the world moved for two frames since the request.</summary>
        private static string WallStillValid()
        {
            var nm = MonoSingleton<NewMovement>.Instance;
            if (nm == null || !nm.activated || nm.dead) return "no_player";
            if (nm.gc == null) return "no_ground_check";
            if (nm.gc.onGround) return "grounded";
            if (PlayerFields.JumpCooldown(nm)) return "jump_cooldown";
            if (nm.currentWallJumps >= 3) return "wall_jumps_spent";
            var wc = PlayerFields.WallChecks(nm);
            if (wc == null) return "no_wall_check_group";
            if (wc.CheckForEnemyCols()) return "enemy_step";
            if (!wc.TryGetActiveInstance(out _)) return "no_wall";
            return null;
        }

        private double WallLead()
        {
            if (WallLeadSeconds >= 0.0) return WallLeadSeconds;
            double gap = FrameGapEstimate;
            if (gap <= 0.0 || gap > 0.25) return 0.0;
            double lead = WallLeadFrames * gap - SsjGapSeconds;
            if (lead <= 0.0) return 0.0;
            return lead > 0.5 ? 0.5 : lead;
        }

        // ---- Per-frame input ---------------------------------------------------------------------------

        /// <summary>
        /// Queues input for the next frame and applies that frame's look. Call at the end of each frame
        /// of a step, starting with the frame on which the step command arrives.
        /// </summary>
        public void ApplyFrame()
        {
            if (!Attached) return;
            frameIndex++;

            if (macroRunning && macroKind == MacroKind.Ssj && frameIndex == 1)
            {
                // M1, one frame, two events. Event A releases the slide (and every tap, which doubles as the
                // release the ordinary path would have queued). Event B presses jump 12 ms later.
                //
                // Both land in one Update, so HandleInputs runs with `sliding` still true -- HandleSlideState
                // comes later in Update() and SlideCancelled only records a timestamp, it does not stop the
                // slide. Jump() therefore takes its `if (sliding)` branch, calls StopSlide() (refreshing
                // velocityAfterSlide) and only then calls TrySSJ. If the runtime instead defers the future-dated
                // event B by a frame, frame 1's HandleSlideState has refreshed velocityAfterSlide by then and
                // the SSJ still lands: M1 is correct either way.
                QueueAt(QueueOpts.SuppressSlide, -1.0);
                macroAnchor = lastQueuedTime;
                QueueAt(QueueOpts.IncludeTaps | QueueOpts.SuppressSlide | QueueOpts.ForceJump, macroAnchor + SsjGapSeconds);
                ApplyLook();
                firstFrame = false;
                return;
            }

            if (macroRunning && macroKind == MacroKind.SsjWall && frameIndex == 1)
            {
                // M2 frame 1: the slide release ALONE, so this frame's HandleSlideState runs StopSlide().
                // It must be two frames. TrySSJ does not add to velocity, it overwrites with
                // `velocityAfterSlide + direction * bonus`, and WallJump never calls StopSlide -- it reaches
                // TrySSJ through the `sliding ||` disjunct. A one-frame wall SSJ would therefore land on the
                // PREVIOUS slide's vector, in that old direction: a speed loss disguised as a technique.
                macroLead = WallLead();
                QueueAt(QueueOpts.SuppressSlide, InputState.currentTime + macroLead);
                macroAnchor = lastQueuedTime;
                ApplyLook();
                firstFrame = false;
                return;
            }

            if (macroRunning && macroKind == MacroKind.SsjWall && frameIndex == 2)
            {
                var why = WallStillValid();
                if (why == null)
                {
                    QueueAt(QueueOpts.IncludeTaps | QueueOpts.SuppressSlide | QueueOpts.ForceJump, macroAnchor + SsjGapSeconds);
                    macroRunning = false;
                    ApplyLook();
                    firstFrame = false;
                    return;
                }
                // The wall, the ground or the jump cooldown changed under us between the two frames. Fall
                // through to the plain action for the rest of the step and say so.
                macroRunning = false;
                macroResult = "degraded";
                macroReason = why;
            }

            var opts = macroSuppressSlide ? QueueOpts.SuppressSlide : QueueOpts.None;
            if (firstFrame && tapsDownLastQueue)
            {
                // The same tap on consecutive steps needs a release first, or it never re-triggers.
                // Both events are processed in the same input update, in order.
                QueueAt(opts, -1.0);
            }
            QueueAt(firstFrame ? opts | QueueOpts.IncludeTaps : opts, -1.0);
            ApplyLook();
            firstFrame = false;
        }

        [System.Flags]
        private enum QueueOpts
        {
            None = 0,
            IncludeTaps = 1,
            /// <summary>Drop `slide` from this event even when the client is holding it (macros only).</summary>
            SuppressSlide = 2,
            /// <summary>Press `jump` whether or not the client asked for it (macros only).</summary>
            ForceJump = 4,
        }

        /// <summary>
        /// Builds one keyboard and one mouse state event and queues both at the same timestamp.
        ///
        /// <paramref name="requestedTime"/> negative means "no opinion": the event goes out with `time = -1`,
        /// the pre-0.8 call, unless the monotonic cursor has to lift it. A non-negative value is a macro
        /// asking for a specific point on `InputState.currentTime`'s clock, and is still lifted to stay
        /// strictly ahead of the previous event.
        /// </summary>
        private void QueueAt(QueueOpts opts, double requestedTime)
        {
            var keyboardState = new KeyboardState();
            var mouseState = new MouseState { position = new Vector2(Screen.width * 0.5f, Screen.height * 0.5f) };

            void Press(List<ButtonControl> controls)
            {
                if (controls == null) return;
                foreach (var control in controls)
                {
                    if (control is KeyControl key) keyboardState.Set(key.keyCode, true);
                    else if (control.device == mouse) mouseState = mouseState.WithButton(ToMouseButton(control), true);
                }
            }

            bool suppressSlide = (opts & QueueOpts.SuppressSlide) != 0;
            bool includeTaps = (opts & QueueOpts.IncludeTaps) != 0;
            bool forceJump = (opts & QueueOpts.ForceJump) != 0;

            foreach (var name in held)
            {
                if (suppressSlide && name == "slide") continue;
                if (buttons.TryGetValue(name, out var c)) Press(c);
            }

            bool taps = includeTaps && (tapped.Count > 0 || (slot >= 1 && slot < slots.Length));
            if (taps)
            {
                foreach (var name in tapped) if (buttons.TryGetValue(name, out var c)) Press(c);
                if (slot >= 1 && slot < slots.Length) Press(slots[slot]);
            }
            if (forceJump && buttons.TryGetValue("jump", out var jumpControls)) Press(jumpControls);
            tapsDownLastQueue = taps || forceJump;

            const float deadzone = 0.33f;
            if (moveY > deadzone) Press(Part("up"));
            if (moveY < -deadzone) Press(Part("down"));
            if (moveX > deadzone) Press(Part("right"));
            if (moveX < -deadzone) Press(Part("left"));

            double now = InputState.currentTime;
            double stamp;
            bool explicitStamp;
            if (requestedTime < 0.0)
            {
                if (now > lastQueuedTime)
                {
                    // The ordinary case, and the pre-0.8 call exactly: let the runtime stamp it.
                    stamp = now;
                    explicitStamp = false;
                }
                else
                {
                    stamp = lastQueuedTime + TimeEpsilon;
                    explicitStamp = true;
                }
            }
            else
            {
                stamp = requestedTime < lastQueuedTime + TimeEpsilon ? lastQueuedTime + TimeEpsilon : requestedTime;
                explicitStamp = true;
            }
            lastQueuedTime = stamp;

            if (explicitStamp)
            {
                InputSystem.QueueStateEvent(keyboard, keyboardState, stamp);
                InputSystem.QueueStateEvent(mouse, mouseState, stamp);
            }
            else
            {
                InputSystem.QueueStateEvent(keyboard, keyboardState);
                InputSystem.QueueStateEvent(mouse, mouseState);
            }
        }

        private List<ButtonControl> Part(string name) => moveParts.TryGetValue(name, out var list) ? list : null;

        private void ApplyLook()
        {
            if (yawPerFrame == 0f && pitchPerFrame == 0f) return;

            var cc = MonoSingleton<CameraController>.Instance;
            if (cc == null || !cc.activated || cc.platformerCamera) return;
            if (GameStateManager.Instance != null && GameStateManager.Instance.CameraLocked) return;
            if (MonoSingleton<OptionsManager>.Instance != null && MonoSingleton<OptionsManager>.Instance.paused) return;

            cc.rotationY += yawPerFrame;
            cc.rotationX = Mathf.Clamp(cc.rotationX + pitchPerFrame, -90f, 90f);
        }

        private static MouseButton ToMouseButton(ButtonControl control)
        {
            switch (control.name)
            {
                case "rightButton": return MouseButton.Right;
                case "middleButton": return MouseButton.Middle;
                case "forwardButton": return MouseButton.Forward;
                case "backButton": return MouseButton.Back;
                default: return MouseButton.Left;
            }
        }

        internal static string MacroName(MacroKind kind)
        {
            switch (kind)
            {
                case MacroKind.Ssj: return "ssj";
                case MacroKind.SsjWall: return "ssj_wall";
                case MacroKind.CoreNuke: return "core_nuke";
                case MacroKind.RocketDown: return "rocket_down";
                case MacroKind.CoinRocket: return "coin_rocket";
                default: return "none";
            }
        }

        /// <summary>
        /// The macro block for this step's obs, or null when the client asked for no macro -- so an obs for a
        /// client that never sends `macro` is byte-identical to a 0.7.2 one.
        ///
        /// `frames` is how many game frames of the step the macro script occupied and `step_frames` how many
        /// the step consumed. They are reported separately on purpose, but in this build every step is
        /// `frameskip` frames long whether or not a macro ran, so the env needs no time correction.
        /// </summary>
        internal JObject BuildMacroReport(int stepStartFrame)
        {
            if (macroRequested == MacroKind.None) return null;
            var obj = new JObject
            {
                ["requested"] = MacroName(macroRequested),
                ["result"] = macroResult ?? "refused",
                ["reason"] = macroReason,
                ["frames"] = macroKind == MacroKind.None ? 0 : macroScriptFrames,
                ["step_frames"] = stepFrames,
                ["gap_s"] = SsjGapSeconds,
                ["lead_s"] = macroLead,
                ["anchor"] = macroAnchor,
                ["frame_gap"] = FrameGapEstimate,
            };
            var ssj = MovementPatches.BuildLast(stepStartFrame);
            obj["ssj"] = ssj; // null when TrySSJ did not run during this step
            obj["ssj_bucket"] = ssj != null ? ssj["bucket"] : new JValue(-1);
            obj["ssj_landed"] = ssj != null && ssj["landed"].Value<bool>();
            return obj;
        }
    }
}
