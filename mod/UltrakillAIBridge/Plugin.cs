using System.Collections;
using BepInEx;
using BepInEx.Configuration;
using BepInEx.Logging;
using HarmonyLib;
using UltrakillAIBridge.Env;
using UltrakillAIBridge.Net;
using UnityEngine;

namespace UltrakillAIBridge
{
    [BepInPlugin(Guid, Name, Version)]
    public class Plugin : BaseUnityPlugin
    {
        public const string Guid = "masstarvt.ultrakill.aibridge";
        public const string Name = "ULTRAKILL AI Bridge";
        public const string Version = "0.2.0";
        public const int ProtocolVersion = 1;

        internal static ManualLogSource Log;

        internal static ConfigEntry<int> Port;
        internal static ConfigEntry<KeyCode> PanicKey;

        private static Harmony harmony;

        private void Awake()
        {
            Log = Logger;

            Port = Config.Bind("Bridge", "Port", 47800, "TCP port (localhost only) the Python side connects to.");
            PanicKey = Config.Bind("Bridge", "PanicKey", KeyCode.F8, "Drops the Python client and gives control back to you.");

            // The game ships with runInBackground off, which stops Update (and therefore the bridge)
            // whenever the window loses focus, e.g. while starting a script from a terminal.
            Application.runInBackground = true;

            harmony = new Harmony(Guid);
            harmony.PatchAll(typeof(SafetyPatches));
            harmony.PatchAll(typeof(TimePatches));

            // ULTRAKILL destroys BepInEx's manager GameObject during startup, which would take this
            // component with it. The bridge runs on its own hidden, persistent object instead.
            var host = new GameObject("UltrakillAIBridge") { hideFlags = HideFlags.HideAndDontSave };
            DontDestroyOnLoad(host);
            host.AddComponent<BridgeRunner>();

            Log.LogInfo($"{Name} {Version} loaded");
        }
    }

    internal sealed class BridgeRunner : MonoBehaviour
    {
        private static readonly WaitForEndOfFrame EndOfFrame = new WaitForEndOfFrame();

        private BridgeServer server;
        private EpisodeController controller;

        private void Awake()
        {
            server = new BridgeServer(Plugin.Port.Value);
            server.Start();
            controller = new EpisodeController(server);
            StartCoroutine(EndOfFrameLoop());
            Plugin.Log.LogInfo($"Listening on 127.0.0.1:{Plugin.Port.Value}");
        }

        private void Update()
        {
            if (Input.GetKeyDown(Plugin.PanicKey.Value) && server.Connected)
            {
                Plugin.Log.LogWarning("Panic key pressed, disconnecting client");
                server.DropClient();
            }
        }

        /// <summary>
        /// The lockstep runs at the end of the frame, after every script and the camera have updated,
        /// so observations are consistent and queued input lands in the very next frame.
        /// </summary>
        private IEnumerator EndOfFrameLoop()
        {
            while (true)
            {
                yield return EndOfFrame;
                controller.EndOfFrame();
            }
        }

        private void OnDestroy()
        {
            Plugin.Log.LogWarning("Bridge runner destroyed, shutting down server");
            controller?.ReleaseControl();
            server?.Dispose();
        }
    }
}
