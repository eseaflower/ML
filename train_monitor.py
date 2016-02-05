import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import time


class FileChangeMonitor(object):
    def __init__(self, filename, onChange):
        self.filename = filename
        self.onChange = onChange
    def watch(self):
        change_time = 0
        while True:
            try:
                fstat = os.stat(filename)
                if change_time != fstat.st_mtime:
                    change_time = fstat.st_mtime
                    self.onChange(self.filename)                    
            except (FileNotFoundError, EOFError):
                change_time = 0
                print("File deleted/reset")
            time.sleep(1)


def handleChange(filename):    
    with open(filename, "rb") as f:
        data = pickle.load(f)
    train = [item["training_outputs"][1] for item in data]
    validation = [item["validation_outputs"][1] for item in data]
    train_cost = [item["training_outputs"][0] for item in data]
    validation_cost = [item["validation_outputs"][0] for item in data]
    f, axarr = plt.subplots(2)   
    axarr[0].plot(train, 'r')    
    axarr[0].plot(validation, 'g')
    axarr[1].plot(train_cost, 'r')    
    axarr[1].plot(validation_cost, 'g')    
    plt.show()

filename = r".\training_stats.pkl"
watcher = FileChangeMonitor(filename, handleChange)
watcher.watch()
