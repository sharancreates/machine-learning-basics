import numpy as np

def sigmoid(sop):
    return 1.0 / (1 + np.exp(-sop))

def error(pred, target):
    return np.power(pred - target, 2)

def error_predicted_deriv(predicted, target):
    return 2 * (predicted - target) # 2*(pred-tar) or 2x

def activation_sop_deriv(sop):
    return sigmoid(sop) * (1.0 - sigmoid(sop)) # actual function

def sop_w_deriv(x):
    return x

def update_w(w, grad, learning_rate):
    return w - learning_rate * grad # Update the weight

x=0.1
target = 0.3
learning_rate = 0.01
w = np.random.rand()
print ("Initial W : ", w)

for k in range (1000000) :
    # Forward Pass
    y = w * x
    predicted = sigmoid(y)
    err = error (predicted, target)

    # Backward Pass
    g1 = error_predicted_deriv (predicted, target)
    g2 = activation_sop_deriv (predicted)
    g3 = sop_w_deriv (x)

    grad = g3 * g2 * g1
    print (predicted)

    W = update_w(w, grad, learning_rate)