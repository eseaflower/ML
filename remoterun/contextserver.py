import sys
import socket
import wireutil
import json
import importlib
import argparse


class Dispatcher(object):
    def __init__(self):
        self.contexts = dict()
        # Define the ops supported.
        self.ops = { "create": self.createContext,
                    "delete": self.deleteContext,
                    "call": self.callContext}
        self.nextId = 0

    def handleCall(self, jsonArguments, binaryContext):
        arguments = json.loads(jsonArguments) # Load one level of arguments.
        contextId = arguments.get("contextId")        
        operation = arguments["op"]
        optionalArgs = arguments.get("arguments")        
        dispatcherArgs = [binaryContext, contextId]
        if optionalArgs:
            dispatcherArgs.append(optionalArgs)
        return self.ops[operation](*dispatcherArgs)

    def dynamicCall(self, binaryContext, target, arguments):
        callable = getattr(target, arguments["method"])        
        innerJson = arguments.get("arguments")
        callArgs = []
        if innerJson:
            jsonArgList = json.loads(innerJson)
            if jsonArgList:
                for jsonArg in jsonArgList:
                    deserializedArg = json.loads(jsonArg)
                    # Check for binary arguments.
                    binaryId = None
                    if hasattr(deserializedArg, "get"):
                        binaryId = deserializedArg.get("__id")
                    if binaryId and binaryContext:
                        deserializedArg = binaryContext.get(binaryId)
                    callArgs.append(deserializedArg)            
        return callable(*callArgs)
    
    def moduleCall(self,binaryContext, jsonArguments):
        arguments = json.loads(jsonArguments)
        module = importlib.import_module(arguments["module"])        
        return self.dynamicCall(binaryContext, module, arguments)
    
    def createContext(self, binaryContext, id, jsonArguments):
        if not id:
            id = str(self.nextId)
            self.nextId += 1
        self.contexts[id] = self.moduleCall(binaryContext, jsonArguments)
        return json.dumps({"contextId": id})

    def deleteContext(self, binaryContext, id):
        del self.contexts[id]
        return json.dumps({"contextId": id})
    def callContext(self, binaryContext, id, jsonArguments):
        result = None
        if not id:
            result = self.moduleCall(binaryContext, jsonArguments)
        else:                        
            arguments = json.loads(jsonArguments)
            result = self.dynamicCall(binaryContext, self.contexts[id], arguments) 
        return json.dumps(result)

class ClientSession(object):
    def __init__(self, client):
        self.client = client
        self.dispatcher = Dispatcher()
    
    def shutdown(self):
        if self.client:
            self.client.close()
            self.client = None

    def handleSession(self):
        try:
            while(True):
                jsonStr = wireutil.read_packed_string(self.client)
                binaryContext = wireutil.read_binary_data(self.client)
                result = self.dispatcher.handleCall(jsonStr, binaryContext)
                wireutil.write_packed_string(self.client, result)
                print("Call completed")
        except:            
            print("Session ended: {0}".format(sys.exc_info()))


def runServer():
        
    p = argparse.ArgumentParser()
    p.add_argument("port", type=int, nargs='?', default=7888)
    args = p.parse_args()
    port = args.port


    host = '' # Listen on local interface
    #port = 7888

    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.bind((host, port))
    listen_socket.listen(1)
    
    #Create Dispatcher object
    dispatcher = Dispatcher()

    while True:    
        print("Waiting for connect...")
        client_socket, addr = listen_socket.accept()
        print("Got connect from ", addr)
        session = ClientSession(client_socket)
        try:            
            session.handleSession()
        finally:
            session.shutdown()
        

        #run(client_socket)          
        #runMethod(client_socket)

    listen_socket.close()

if __name__ == "__main__":    
    sys.path.append(r'G:\MyWork\PythonTest\PythonCode\theano2')    
    runServer()
