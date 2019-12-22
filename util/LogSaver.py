import numpy as np


def save_log(file_path=None, stat=""):
    fp = open(file_path, mode="a+")
    fp.write(stat + "\n")
    fp.close()
