using System.Collections;
using BepInEx;
using BepInEx.Configuration;
using BepInEx.Logging;
using HarmonyLib;
using UltrakillAIBridge.Env;
using UltrakillAIBridge.Net;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace UltrakillAIBridge
{
    [BepInPlugin(Guid, Name, Version)]
    public class Plugin : BaseUnityPlugin
    {
        public const string Guid = "masstarvt.ultrakill.aibridge";
        public const string Name = "ULTRAKILL AI Bridge";
        public const string Version = "0.7.0";
        public const int ProtocolVersion = 1;

        internal static ManualLogSource Log;

        internal static ConfigEntry<int> Port;
        internal static ConfigEntry<KeyCode> PanicKey;

        /// <summary>Launched by scripts/games.py for parallel training (has -aibridge-port on the command line).</summary>
        internal static bool IsTrainingInstance { get; private set; }

        /// <summary>Port from -aibridge-port or the config; updated to the port actually bound.</summary>
        internal static int ListenPort { get; set; }

        private static Harmony harmony;

        private void Awake()
        {
            Log = Logger;

            Port = Config.Bind("Bridge", "Port", 47800, "TCP port (localhost only) the Python side connects to.");
            PanicKey = Config.Bind("Bridge", "PanicKey", KeyCode.F8, "Drops the Python client and gives control back to you.");

            ListenPort = Port.Value;
            var args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "-aibridge-port" && int.TryParse(args[i + 1], out var port))
                {
                    ListenPort = port;
                    IsTrainingInstance = true;
                }
            }

            // The game ships with runInBackground off, which stops Update (and therefore the bridge)
            // whenever the window loses focus, e.g. while starting a script from a terminal.
            Application.runInBackground = true;

            harmony = new Harmony(Guid);
            harmony.PatchAll(typeof(SafetyPatches));
            harmony.PatchAll(typeof(TimePatches));
            harmony.PatchAll(typeof(BackgroundPatches));
            harmony.PatchAll(typeof(InstancePatches));
            harmony.PatchAll(typeof(TrainingSpeed));
            try
            {
                // Binds the private NewMovement.HandleSlideState, so a game update renaming it would throw.
                // Isolated so the un-wedge failing can't take soft death and the rest of TrainingSpeed down.
                harmony.PatchAll(typeof(UnwedgePatch));
            }
            catch (System.Exception e)
            {
                Log.LogError($"Un-wedge patch failed to apply, the slowMode state will not be broken: {e}");
            }
            try
            {
                // CampaignPatches binds a private method (ActivateNextWave.EndWaves) among its five targets;
                // a game update renaming or restructuring any of them would throw here. Isolate that failure
                // so campaign support degrades instead of taking the whole bridge down with it.
                harmony.PatchAll(typeof(CampaignPatches));
            }
            catch (System.Exception e)
            {
                Log.LogError($"Campaign patches failed to apply, campaign support is disabled: {e}");
            }
            EnemyTracker.onEnemyAdded += TrainingSpeed.OnEnemyAdded;

            // Arena and door keys belong to one level load. Checkpoint respawns don't load a scene, so the
            // keys survive them, which is what lets Python ignore an arena cleared again after a death.
            SceneManager.sceneLoaded += (scene, mode) =>
            {
                if (mode == LoadSceneMode.Single) CampaignPatches.OnSceneLoaded();
            };

            // ULTRAKILL destroys BepInEx's manager GameObject during startup, which would take this
            // component with it. The bridge runs on its own hidden, persistent object instead.
            var host = new GameObject("UltrakillAIBridge") { hideFlags = HideFlags.HideAndDontSave };
            DontDestroyOnLoad(host);
            host.AddComponent<BridgeRunner>();

            Log.LogInfo($"{Name} {Version} loaded{(IsTrainingInstance ? " as a training instance" : "")}");
        }
    }

    internal sealed class BridgeRunner : MonoBehaviour
    {
        private static readonly WaitForEndOfFrame EndOfFrame = new WaitForEndOfFrame();

        private BridgeServer server;
        private EpisodeController controller;

        private void Awake()
        {
            // Training instances need their exact port. A normally launched game moves to the next free
            // port if the configured one is taken (e.g. by training instances already running).
            int attempts = Plugin.IsTrainingInstance ? 1 : 16;
            int basePort = Plugin.ListenPort;
            for (int i = 0; i < attempts; i++)
            {
                int port = basePort + i;
                try
                {
                    server = new BridgeServer(port);
                    server.Start();
                    Plugin.ListenPort = port;
                    Plugin.Log.LogInfo($"Listening on 127.0.0.1:{port}");
                    break;
                }
                catch (System.Net.Sockets.SocketException e)
                {
                    Plugin.Log.LogWarning($"Port {port} unavailable: {e.Message}");
                    server = null;
                }
            }
            if (server == null)
            {
                Plugin.Log.LogError("Could not start the bridge server on any port");
                enabled = false;
                return;
            }
            controller = new EpisodeController(server);
            StartCoroutine(EndOfFrameLoop());
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

        private void OnApplicationQuit()
        {
            // Restore the player's display settings before Unity saves them on exit.
            controller?.ReleaseControl();
        }

        private void OnDestroy()
        {
            Plugin.Log.LogWarning("Bridge runner destroyed, shutting down server");
            controller?.ReleaseControl();
            server?.Dispose();
        }
    }
}
