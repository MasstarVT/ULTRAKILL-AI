using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.LowLevel;

namespace UltrakillAIBridge.Act
{
    /// <summary>
    /// Drives the player through a virtual keyboard and mouse registered with the Unity Input System,
    /// so the game's own input pipeline (InputActionState: IsPressed, WasPerformedThisFrame, ReadValue)
    /// behaves exactly as it does for a human. Keys are resolved from the player's current bindings.
    /// Camera look bypasses the mouse and rotates CameraController directly, so look actions are in
    /// degrees and independent of mouse sensitivity.
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

        public bool Attached => keyboard != null;

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
        /// {"move":[x,y], "look":[yaw_deg,pitch_deg], "buttons":["fire1","jump",...], "slot":0..6}
        /// Look degrees are spread evenly over the step's frames.
        /// </summary>
        public void SetAction(JObject action, int frames)
        {
            Clear();
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
        }

        public void Clear()
        {
            moveX = moveY = yawPerFrame = pitchPerFrame = 0f;
            held.Clear();
            tapped.Clear();
            slot = 0;
            firstFrame = false;
        }

        /// <summary>
        /// Queues input for the next frame and applies that frame's look. Call at the end of each frame
        /// of a step, starting with the frame on which the step command arrives.
        /// </summary>
        public void ApplyFrame()
        {
            if (!Attached) return;
            if (firstFrame && tapsDownLastQueue)
            {
                // The same tap on consecutive steps needs a release first, or it never re-triggers.
                // Both events are processed in the same input update, in order.
                Queue(includeTaps: false);
            }
            Queue(includeTaps: firstFrame);
            ApplyLook();
            firstFrame = false;
        }

        private bool tapsDownLastQueue;

        private void Queue(bool includeTaps)
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

            foreach (var name in held) if (buttons.TryGetValue(name, out var c)) Press(c);
            bool taps = includeTaps && (tapped.Count > 0 || (slot >= 1 && slot < slots.Length));
            if (taps)
            {
                foreach (var name in tapped) if (buttons.TryGetValue(name, out var c)) Press(c);
                if (slot >= 1 && slot < slots.Length) Press(slots[slot]);
            }
            tapsDownLastQueue = taps;

            const float deadzone = 0.33f;
            if (moveY > deadzone) Press(Part("up"));
            if (moveY < -deadzone) Press(Part("down"));
            if (moveX > deadzone) Press(Part("right"));
            if (moveX < -deadzone) Press(Part("left"));

            InputSystem.QueueStateEvent(keyboard, keyboardState);
            InputSystem.QueueStateEvent(mouse, mouseState);
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
    }
}
