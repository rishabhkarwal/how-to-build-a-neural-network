import numpy as np

# activation functions
def _sigmoid(x, derivative = False):
    if derivative: # d/dx[σ(x)] = σ(x) ⋅ (1 − σ(x))
        s = _sigmoid(x) 
        return s * (1 - s)
    return 1 / (1 + np.exp(-x))

def _relu(x, derivative = False):
    if derivative:
        return (x > 0).astype(float)
    return x * (x > 0)

def _tanh(x, derivative = False):
    if derivative:
        return 1 - np.tanh(x) ** 2
    return np.tanh(x)


#e error/loss function
def _mean_squared_error(x, y): #better for regression
    return np.mean((x - y) ** 2)

def _cross_entropy_loss(x, y):
    x = np.clip(x, 1e-12, 1.0) # safety in case x = 0 
    return -np.mean(np.sum(y * np.log(x), axis=0))