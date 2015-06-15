import sys
import socket
import struct
import wireutil
import subutil
import json
import importlib


class SocketRedirect(object):
    def __init__(self, s):
        self.client = s
    def send(self, line):
        wireutil.write_packed_string(self.client, line)


def run(client):
    try:
        message = wireutil.read_packed_string(client)
        print("Got message", message)

        redirector = SocketRedirect(client)
        subutil.exec_child(message, redirector.send)
        print("Closing client")
        client.close()
    except:
        print("Unexpected error:", sys.exc_info()[0])


def runMethod(client):
    jsonStr = wireutil.read_packed_string(client)
    jsonObject = json.loads(jsonStr)
    mn = jsonObject["module"]
    print(mn)
    module = importlib.import_module(mn)
    method = getattr(module, jsonObject["method"])
    jsonArguments = jsonObject.get("arguments")
    args = None
    if jsonArguments:
        args = json.loads(jsonArguments)
    result = method(*args)
    jsonResult = json.dumps(result)
    wireutil.write_packed_string(client, jsonResult)
    print("done runMethod")
    client.close()


host = '' # Listen on local interface
port = 7888

listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.bind((host, port))
listen_socket.listen(1)

while True:    
    print("Waiting for connect...")
    client_socket, addr = listen_socket.accept()
    print("Got connect from ", addr)
    #run(client_socket)          
    runMethod(client_socket)

listen_socket.close()
