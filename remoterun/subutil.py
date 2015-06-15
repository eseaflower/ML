import os
import subprocess
import select


class Redirector(object):
    def __init__(self, produce_fn, consume_fn):
        self.produce_fn = produce_fn
        self.consume_fn = consume_fn
    def tick(self):
        line = str.rstrip(self.produce_fn())
        if line:
            self.consume_fn(line)

def exec_child(program, redir_target_fn):
    exec_args = ["python","-u",program]
    work_dir = os.path.split(program)[0]
    child = subprocess.Popen(exec_args, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    redir = Redirector(child.stdout.readline, redir_target_fn)
    while child.poll() is None:
        redir.tick()       

    return child.wait()

def test_exec_child():    
    def m_print(l):
        print("Test: " + l)
    exec_child("TheanoTests.py", m_print)
    print("Child done!")

if __name__ == "__main__":
    test_exec_child()