using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json.Linq;

namespace UltrakillAIBridge.Net
{
    /// <summary>
    /// Single-client TCP server speaking newline-delimited JSON.
    /// Socket I/O runs on background threads; the Unity main thread consumes
    /// messages through <see cref="TryReceive"/> / <see cref="WaitReceive"/>.
    /// </summary>
    public sealed class BridgeServer : IDisposable
    {
        public sealed class Incoming
        {
            public JObject Message;
            public bool Disconnected;
            public int ClientId;
        }

        private readonly int port;
        private readonly BlockingCollection<Incoming> inbox = new BlockingCollection<Incoming>();
        private readonly object clientLock = new object();

        private TcpListener listener;
        private Thread acceptThread;
        private TcpClient client;
        private StreamWriter writer;
        private int clientId;
        private volatile bool running;

        public BridgeServer(int port)
        {
            this.port = port;
        }

        public bool Connected
        {
            get { lock (clientLock) return client != null; }
        }

        public int CurrentClientId
        {
            get { lock (clientLock) return client != null ? clientId : -1; }
        }

        public void Start()
        {
            running = true;
            listener = new TcpListener(IPAddress.Loopback, port);
            listener.Start();
            acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "UltrakillAI-Accept" };
            acceptThread.Start();
        }

        private void AcceptLoop()
        {
            while (running)
            {
                TcpClient incoming;
                try
                {
                    incoming = listener.AcceptTcpClient();
                }
                catch (Exception)
                {
                    if (!running) return;
                    continue;
                }

                incoming.NoDelay = true;
                // Sends happen on the main thread; never let a client that stopped reading freeze the game.
                incoming.Client.SendTimeout = 5000;
                int id;
                lock (clientLock)
                {
                    CloseClientLocked();
                    client = incoming;
                    id = ++clientId;
                    writer = new StreamWriter(incoming.GetStream(), new UTF8Encoding(false)) { AutoFlush = true, NewLine = "\n" };
                }
                Plugin.Log.LogInfo($"Client {id} connected");

                var readThread = new Thread(() => ReadLoop(incoming, id)) { IsBackground = true, Name = "UltrakillAI-Read" };
                readThread.Start();
            }
        }

        private void ReadLoop(TcpClient tcp, int id)
        {
            try
            {
                using (var reader = new StreamReader(tcp.GetStream(), new UTF8Encoding(false)))
                {
                    string line;
                    while (running && (line = reader.ReadLine()) != null)
                    {
                        if (line.Length == 0) continue;
                        JObject msg;
                        try
                        {
                            msg = JObject.Parse(line);
                        }
                        catch (Exception e)
                        {
                            Plugin.Log.LogWarning($"Bad message from client {id}: {e.Message}");
                            continue;
                        }
                        inbox.Add(new Incoming { Message = msg, ClientId = id });
                    }
                }
            }
            catch (Exception)
            {
                // Connection closed or reset.
            }

            lock (clientLock)
            {
                if (client == tcp) CloseClientLocked();
            }
            inbox.Add(new Incoming { Disconnected = true, ClientId = id });
            Plugin.Log.LogInfo($"Client {id} disconnected");
        }

        public bool TryReceive(out Incoming msg) => inbox.TryTake(out msg);

        /// <summary>Blocks the calling thread until a message or disconnect arrives. Returns null on timeout.</summary>
        public Incoming WaitReceive(int timeoutMs) => inbox.TryTake(out var msg, timeoutMs) ? msg : null;

        public void Send(string json)
        {
            lock (clientLock)
            {
                if (writer == null) return;
                try
                {
                    writer.WriteLine(json);
                }
                catch (Exception e)
                {
                    Plugin.Log.LogWarning($"Send failed: {e.Message}");
                    CloseClientLocked();
                }
            }
        }

        public void Send(JObject obj) => Send(obj.ToString(Newtonsoft.Json.Formatting.None));

        public void DropClient()
        {
            lock (clientLock) CloseClientLocked();
        }

        private void CloseClientLocked()
        {
            try { writer?.Dispose(); } catch (Exception) { }
            try { client?.Close(); } catch (Exception) { }
            writer = null;
            client = null;
        }

        public void Dispose()
        {
            running = false;
            try { listener?.Stop(); } catch (Exception) { }
            DropClient();
        }
    }
}
