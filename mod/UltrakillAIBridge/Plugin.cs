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
        public const string Version = "0.1.0";
        public const int ProtocolVersion = 1;

        internal static ManualLogSource Log;

        internal static ConfigEntry<int> Port;
        internal static ConfigEntry<KeyCode> PanicKey;

        private Harmony harmony;
        private BridgeServer server;
        private EpisodeController controller;

        private void Awake()
        {
            Log = Logger;

            Port = Config.Bind("Bridge", "Port", 47800, "TCP port (localhost only) the Python side connects to.");
            PanicKey = Config.Bind("Bridge", "PanicKey", KeyCode.F8, "Drops the Python client and gives control back to you.");

            harmony = new Harmony(Guid);
            harmony.PatchAll(typeof(SafetyPatches));

            server = new BridgeServer(Port.Value);
            server.Start();

            controller = new EpisodeController(server);

            Log.LogInfo($"{Name} {Version} loaded, listening on 127.0.0.1:{Port.Value}");
        }

        private void Update()
        {
            if (Input.GetKeyDown(PanicKey.Value) && server.Connected)
            {
                Log.LogWarning("Panic key pressed, disconnecting client");
                server.DropClient();
            }

            controller.Tick();
        }

        private void OnDestroy()
        {
            controller?.ReleaseControl();
            server?.Dispose();
            harmony?.UnpatchSelf();
        }
    }
}
