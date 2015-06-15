using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

namespace PythonComm {

    public class BinaryContext {
        public Dictionary<string, MemoryStream> data = new Dictionary<string, MemoryStream>();
    }

    public class BinaryArgument {
        public string __id { get; set; }
        public BinaryArgument() {
            this.__id = Guid.NewGuid().ToString();
        }
    }

    public class BasePythonCall {
        public string method { get; set; }
        public BasePythonCall(string method) {
            this.method = method;
        }
        public virtual string Serialize() {
            return JsonConvert.SerializeObject(this);
        }
    }
    public class PythonCallWithArguments : BasePythonCall {
        public string arguments { get; set; }
        public PythonCallWithArguments(string method, string arguments)
            : base(method) {
            this.arguments = arguments;
        }
    }
    public class StaticPythonCall : PythonCallWithArguments {
        public string module { get; set; }
        public StaticPythonCall(string module, string method, string arguments)
            : base(method, arguments) {
            this.module = module;
        }
    }
    public class StaticPythonDispatcher {
        public string op { get; set; }
        public string arguments { get; set; }
        public StaticPythonDispatcher(string op, string arguments) {
            this.op = op;
            this.arguments = arguments;
        }
        public virtual string Serialize() {
            return JsonConvert.SerializeObject(this);
        }

        public static StaticPythonDispatcher Create(StaticPythonCall ctor) {
            return new StaticPythonDispatcher("create", ctor.Serialize());
        }
        public static StaticPythonDispatcher Call(StaticPythonCall function) {
            return new StaticPythonDispatcher("call", function.Serialize());
        }
    }
    public class InstancePythonDispatcher : StaticPythonDispatcher {
        public string contextId { get; set; }
        public InstancePythonDispatcher(string instance, string op, string arguments)
            : base(op, arguments) {
            this.contextId = instance;
        }
        public static InstancePythonDispatcher Call(string instance, PythonCallWithArguments method) {
            return new InstancePythonDispatcher(instance, "call", method.Serialize());
        }
        public static InstancePythonDispatcher Delete(string instance) {
            return new InstancePythonDispatcher(instance, "delete", null);
        }
    }


    public class PythonCommunicator : IDisposable {
        private string host;
        private int port;
        private Socket socket;

        public PythonCommunicator(string host, int port) {
            this.host = host;
            this.port = port;
            this.socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            this.socket.Connect(this.host, this.port);

        }


        public string Call(string request, BinaryContext binaryData) {
            this.WriteString(request);

            this.WriteBinaryData(binaryData);

            return this.ReadString();
        }

        private void WriteBinaryData(BinaryContext context) {
            int items = 0;
            if (context != null) {
                items = context.data.Count;
            }
            this.WriteInt(items);
            if (items > 0) {
                foreach (var kvPair in context.data) {
                    this.WriteString(kvPair.Key);
                    var stream = kvPair.Value;
                    byte[] data = new byte[stream.Length];
                    stream.Position = 0;
                    stream.Read(data, 0, data.Length);
                    this.WriteBytes(data);
                }
            }
        }


        public void WriteInt(int i) {
            var intBuffer = BitConverter.GetBytes(i);
            this.WriteWithRetry(intBuffer, 0, intBuffer.Length);
        }

        public void WriteString(string s) {
            var stringBuffer = System.Text.ASCIIEncoding.UTF8.GetBytes(s);
            this.WriteBytes(stringBuffer);
        }

        public void WriteBytes(byte[] data) {
            var lengthBuffer = BitConverter.GetBytes(data.Length);
            this.WriteWithRetry(lengthBuffer, 0, lengthBuffer.Length);
            this.WriteWithRetry(data, 0, data.Length);
        }

        public int WriteWithRetry(byte[] data, int offset, int length) {
            return this.SocketOperation((o, l) => this.socket.Send(data, o, l, SocketFlags.None),
                offset, length);
        }

        public string ReadString() {
            var stringBuffer = this.ReadBytes();
            return System.Text.ASCIIEncoding.UTF8.GetString(stringBuffer);
        }

        public byte[] ReadBytes() {
            byte[] lengthBuffer = new byte[sizeof(int)];
            this.ReadWithRetry(lengthBuffer, 0, lengthBuffer.Length);
            int length = BitConverter.ToInt32(lengthBuffer, 0);
            byte[] recvBuffer = new byte[length];
            this.ReadWithRetry(recvBuffer, 0, recvBuffer.Length);
            return recvBuffer;
        }

        public int ReadWithRetry(byte[] buffer, int offset, int length) {
            return this.SocketOperation((o, l) => this.socket.Receive(buffer, o, l, SocketFlags.None),
                offset, length);
        }

        private int SocketOperation(Func<int, int, int> socketOp, int offset, int length) {
            int totalProcessed = 0;
            int leftToProcess = length;
            while (leftToProcess > 0) {
                if (!this.socket.Connected) {
                    break;
                }
                int processed = socketOp(offset, leftToProcess);
                offset += processed;
                leftToProcess -= processed;
                totalProcessed += processed;
            }
            if (totalProcessed != length) {
                throw new ApplicationException(String.Format("Failed to process socket. {0} of {1} bytes successfull.", totalProcessed, length));
            }
            return totalProcessed;
        }

        #region Dispose
        ~PythonCommunicator() {
            this.Dispose(false);
        }
        public void Dispose() {
            this.Dispose(true);
            GC.SuppressFinalize(this);
        }
        protected void Dispose(bool disposing) {
            if (disposing) {
                if (this.socket != null) {
                    this.socket.Shutdown(SocketShutdown.Both);
                    this.socket.Close();
                    this.socket.Dispose();
                    this.socket = null;
                }
            }
        }
        #endregion


        private static PythonCommunicator sInstance = null;
        public static PythonCommunicator Instance {
            get {
                if (sInstance == null) {
                    sInstance = new PythonCommunicator("localhost", 7888);
                }
                return sInstance;
            }
        }
    }

    public class ContextResponse {
        public string contextId { get; set; }
        public static ContextResponse Deserialize(string response) {
            return JsonConvert.DeserializeObject<ContextResponse>(response);
        }
    }

    public class BasePythonContext : IDisposable {
        protected string contextId;
        protected string module;
        protected string constructor;
        public BasePythonContext(string module, string constructor) {
            this.module = module;
            this.constructor = constructor;
        }

        public void Construct(params object[] arguments) {
            this.InternalConstruct(SerializeArguments(arguments));
        }

        private void InternalConstruct(Tuple<string, BinaryContext> argument) {
            var dispatcher = StaticPythonDispatcher.Create(new StaticPythonCall(this.module, this.constructor, argument.Item1));
            ContextResponse result = this.InternalCall<ContextResponse>(dispatcher, argument.Item2);
            this.contextId = result.contextId;
        }


        public void StaticCall(string method) {
            this.StaticCall<object>(method);
        }

        public U StaticCall<U>(string method, params object[] arguments) {
            return this.InternalStaticCall<U>(method, SerializeArguments(arguments));
        }
        private U InternalStaticCall<U>(string method, Tuple<string, BinaryContext> arguments) {
            var dispatcher = StaticPythonDispatcher.Call(new StaticPythonCall(this.module, method, arguments.Item1));
            return this.InternalCall<U>(dispatcher, arguments.Item2);
        }


        public void InstanceCall(string method) {
            this.InstanceCall<object>(method);
        }

        public U InstanceCall<U>(string method, params object[] arguments) {
            return this.InternalInstanceCall<U>(method, SerializeArguments(arguments));
        }
        private U InternalInstanceCall<U>(string method, Tuple<string, BinaryContext> arguments) {
            var dispatcher = InstancePythonDispatcher.Call(this.contextId, new PythonCallWithArguments(method, arguments.Item1));
            return this.InternalCall<U>(dispatcher, arguments.Item2);
        }


        private U InternalCall<U>(StaticPythonDispatcher dispatcher, BinaryContext context) {
            var response = PythonCommunicator.Instance.Call(dispatcher.Serialize(), context);
            return JsonConvert.DeserializeObject<U>(response);
        }

        public void Dispose() {
            if (this.contextId != null) {
                var dispatcher = InstancePythonDispatcher.Delete(this.contextId);
                var result = this.InternalCall<ContextResponse>(dispatcher, null);
                this.contextId = null;
            }
        }


        private Tuple<string, BinaryContext> SerializeArguments(params object[] arguments) {
            List<string> serializedArguments = null;
            BinaryContext binaryContext = null;
            if (arguments.Length > 0) {
                serializedArguments = new List<string>(arguments.Length);
                foreach (var arg in arguments) {
                    object argToSerialize = arg;
                    // Check for binary data
                    var memStream = arg as MemoryStream;
                    if (memStream != null) {
                        if (binaryContext == null) {
                            binaryContext = new BinaryContext();
                        }
                        var binArg = new BinaryArgument();
                        binaryContext.data.Add(binArg.__id, memStream);
                        argToSerialize = binArg;
                    }
                    serializedArguments.Add(JsonConvert.SerializeObject(argToSerialize));
                }
            }
            var plainArguments = JsonConvert.SerializeObject(serializedArguments);
            return Tuple.Create(plainArguments, binaryContext);
        }
    }
}
