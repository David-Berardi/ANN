# @author David Berardi

import matplotlib.pyplot as plt
import numpy as np
import time

# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of the activation function
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# training data
training_data = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
]

# random initial values for the weights and bias
w1 = np.random.randn()
w2 = np.random.randn()
b  = np.random.randn()

# step for gradiant descent
eta = 0.2

# list of the costs for graphing the cost history
costs = []

# list of the expected outputs
expect = []

# list of the normalized predicted outputs
predicted = []

# training cycle
start = time.time()
for sample in range(10000000):
    # select training data
    index = np.random.randint(len(training_data))
    data = training_data[index]

    # data parameters: x1, x2, expected value
    x1 = data[0]
    x2 = data[1]
    expected = data[2]
    if sample % 100000 == 0: expect.append(expected)

    # feed forward
    h = (w1 * x1) + (w2 * x2) + b;
    output = sigmoid(h)
    if sample % 100000 == 0: predicted.append(output)

    # QC, quadratic cost
    cost = np.square(output - expected)
    costs.append(cost)

    # derivatives
    dw1 = x1    # w1 derivate
    dw2 = x2     # w2 derivative
    db = 1          # b derivative
    dsigmoid = d_sigmoid(h) # sigmoid derivative with h as a parameter
    dcost = 2 * (output - expected) * dsigmoid

    dcost_dw1 = dcost * dw1 # derivative of the cost with respect of the first weight
    dcost_dw2 = dcost * dw2 # derivative of the cost with respect of the second weight
    dcost_db = dcost * db   # derivative of the cost with respect of the bias

    # backpropagation, update the values of the weights and bias
    w1 = w1 - (eta * dcost_dw1)
    w2 = w2 - (eta * dcost_dw2)
    b = b - (eta * dcost_db)

    if sample % 1000000 == 0: print(f"index: {sample}\tx1: {x1}\tx2: {x2}\ty: {output}\texpected: {expected}\tcost: {cost}")

end = time.time();
print(f"The training took: {end - start}s")
print(f"w1: {w1}\tw2: {w2}\tb: {b}")

# save the state of the machine
save_state = open("state.txt", "w")
save_state.write(str(w1))
save_state.write("\n")
save_state.write(str(w2))
save_state.write("\n")
save_state.write(str(b))
save_state.close()

plt.title("Gradiant Descent")
plt.xlabel("Samples")
plt.ylabel("Error")
plt.subplot(2, 1, 1)
plt.plot(costs)
plt.grid(True)

plt.title("Predicted Outputs")
plt.xlabel("Samples")
plt.ylabel("Expected/Predicted")
plt.subplot(2, 1, 2)
plt.plot(expect, label="expected")
plt.plot(predicted, label="predicted", color="r")
plt.legend(loc="upper right")

plt.show()
