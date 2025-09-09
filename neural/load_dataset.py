import numpy as np
def load_dataset(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)
    
#Dataset Source: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz