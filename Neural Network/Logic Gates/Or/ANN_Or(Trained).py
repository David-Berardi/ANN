# @author David Berardi

import matplotlib.pyplot as plt
import numpy as np

# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

unknown = [0, 1]

# load state from trained
file = open("state.txt", "r")
txt = file.readlines()
w1 = np.float(txt[0].replace("\n", ""))
w2 = np.float(txt[1].replace("\n", ""))
b  = np.float(txt[2])

# data parameters: length, width, expected value
length = unknown[0]
width = unknown[1]

# feed forward
h = (w1 * length) + (w2 * width) + b;
output = sigmoid(h)

print(f"length: {length}\twidth: {width}\toutput: {'1' if output >= .9 else '0'}\t{output * 100}%")
