import numpy as np
from functions import _cross_entropy_loss, _sigmoid
from random import shuffle
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images, self.labels = Dataset.adjust(images, labels)

    @staticmethod
    def adjust(images, labels):
        return images / 255, np.eye(10)[labels]
    
class Layer:
    def __init__(self, in_nodes: int, out_nodes: int, activation_function):
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.weights = np.random.uniform(-0.5, 0.5, (out_nodes, in_nodes))
        self.biases = np.zeros((out_nodes, 1))
        self.activation = activation_function

from ipywidgets import interact
class Neural:
    def __init__(self, epochs, learn_rate, nodes, dataset_path):
        self.epochs, self.learn_rate = epochs, learn_rate
        #define layers
        input_layer_nodes, hidden_layer_nodes, output_layer_nodes = nodes
        self.input_hidden = Layer(input_layer_nodes, hidden_layer_nodes, _sigmoid)
        self.hidden_output = Layer(hidden_layer_nodes, output_layer_nodes, _sigmoid)

        self.training_data, self.test_data = Neural.load_data(dataset_path)

    @staticmethod
    def load_data(path):
        with np.load(path) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
            return Dataset(x_train, y_train), Dataset(x_test, y_test)
        
    def train(self, cost = _cross_entropy_loss):
        dataset = self.training_data
        n_samples = dataset.images.shape[0]

        for epoch in range(self.epochs):
            correct = 0
            total_error = 0
            dataset_list = list(zip(dataset.images, dataset.labels))
            shuffle(dataset_list)

            for image, label in dataset_list:
                image = image.reshape(784, 1)
                label = label.reshape(10, 1)
                
                output, hidden = self.forward(image)

                error = cost(output, label)
                total_error += error

                correct += int(np.argmax(output) == np.argmax(label))

                self.back(output - label, hidden, image)

            avg_error = total_error / n_samples
            accuracy = correct / n_samples * 100

            print(f"Epoch {epoch+1}/{self.epochs}, Accuracy: {accuracy:.2f}%, Error: {avg_error:.6f}")

    def forward(self, image):
        w_i_h, w_h_o, b_i_h, b_h_o = self.input_hidden.weights, self.hidden_output.weights, self.input_hidden.biases, self.hidden_output.biases
        # Forward propagation input -> hidden
        hidden = self.input_hidden.activation(b_i_h + w_i_h @ image)
        # Forward propagation hidden -> output
        output = self.hidden_output.activation(b_h_o + w_h_o @ hidden)
        return output, hidden

    def back(self, difference, hidden, image): #difference = delta_output
        w_i_h, w_h_o, b_i_h, b_h_o = self.input_hidden.weights, self.hidden_output.weights, self.input_hidden.biases, self.hidden_output.biases

        delta_h = w_h_o.T @ difference * self.input_hidden.activation(b_i_h + w_i_h @ image, True)
        # Backpropagation output -> hidden (cost function derivative)
        w_h_o += -self.learn_rate * difference @ hidden.T
        b_h_o += -self.learn_rate * difference
        # Backpropagation hidden -> input (activation function derivative)
        w_i_h += -self.learn_rate * delta_h @ image.T
        b_i_h += -self.learn_rate * delta_h
        

    def test(self):
        count = 0
        dataset = self.test_data
        for idx, (image, label) in enumerate(zip(dataset.images, dataset.labels)):
            image, label = image.reshape(784, 1), np.argmax(label)
            output, _ = self.forward(image)
            pred_label = int(np.argmax(output.flatten()))

            if pred_label == label: count += 1
        print(f"{count}/{idx + 1}\t{count/(idx + 1) : .2f}%")
        

net = Neural(epochs=3, learn_rate=0.015, nodes=[784, 20, 10], dataset_path='./dataset/mnist.npz')

net.train()
net.test()