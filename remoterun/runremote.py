import socket
import wireutil
import glob
import shutil
import os
import json

def run_remote(program, host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    # Launch remote
    wireutil.write_packed_string(s, program)
    while True:
        try:
            msg = wireutil.read_packed_string(s)
            print(msg)
        except EOFError:
            print("Host closed down")
            break
    s.close()

def copy_files_to_remote(target_dir):
    py_files = glob.glob("*.py")
    for file in py_files:
        print("Copying {0} to {1}".format(file, target_dir))
        shutil.copy(file, target_dir)

def exec_remote(host="antabus2", port=7888, remote_dir="//antabus2/execstorage/", program="TheanoTests.py"):
    copy_files_to_remote(remote_dir)
    exec_path=os.path.join(remote_dir, program)
    run_remote(exec_path, host, port)


def exec_method(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    callObject = dict()
    callObject["module"] = "methodTest"
    callObject["method"] = "test"
    callObject["arguments"] = ["Hej","hopp"]
    jsonStr = json.dumps(callObject)
    wireutil.write_packed_string(s, jsonStr)
    resultStr = wireutil.read_packed_string(s)
    resultObject = json.loads(resultStr)
    print(resultObject)
    s.close()


if __name__ == "__main__":
    #exec_remote("antabus2", 7888, "//antabus2/execstorage/", "TheanoTests.py")
    input("alsdkalkd")
    
    exec_method("localhost", 7888)
    print("Done")