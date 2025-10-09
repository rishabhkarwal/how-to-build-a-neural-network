import numpy as np
from functions import _cross_entropy_loss, _sigmoid
from random import shuffle
from rich.console import Console
from rich.table import Table
from rich.progress import track
import os
from time import perf_counter

console = Console()

class Dataset:
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images, self.labels = Dataset.adjust(images, labels)

    @staticmethod
    def adjust(images, labels):
        n_classes = np.max(labels) + 1
        return images / 255, np.eye(n_classes)[labels]
    
class Layer:
    def __init__(self, in_nodes: int, out_nodes: int, activation_function):
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

        self.weights = np.random.uniform(-0.5, 0.5, (out_nodes, in_nodes))
        self.biases = np.zeros((out_nodes, 1))
        self.activation = activation_function

class Neural:
    def __init__(self, epochs, learn_rate, nodes, dataset_path):
        self.epochs, self.learn_rate = epochs, learn_rate
        input_layer_nodes, hidden_layer_nodes, output_layer_nodes = nodes

        self.input_hidden = Layer(input_layer_nodes, hidden_layer_nodes, _sigmoid)
        self.hidden_output = Layer(hidden_layer_nodes, output_layer_nodes, _sigmoid)

        self.training_data, self.test_data = Neural.load_data(dataset_path)

    @staticmethod
    def load_data(path):
        with np.load(path) as f:
            return Dataset(f['x_train'], f['y_train']), Dataset(f['x_test'], f['y_test'])
        
    def train(self, cost=_cross_entropy_loss):
        dataset = self.training_data
        n_samples = dataset.images.shape[0]

        table = Table(title='Training Statistics')
        table.add_column('Epoch', justify='right', style='pale_turquoise4', no_wrap=True)
        table.add_column('Accuracy (%)', justify='right', style='pale_turquoise4')
        table.add_column('Avg Error', justify='right', style='pale_turquoise4')
        table.add_column('Time (s)', justify='right', style='pale_turquoise4')

        for epoch in range(self.epochs):
            correct, total_error = 0, 0
            dataset_list = list(zip(dataset.images, dataset.labels))
            shuffle(dataset_list)
            start_time = perf_counter()
            for image, label in track(dataset_list, description=f'[steel_blue]Epoch {epoch + 1}/{self.epochs}', complete_style='cyan', finished_style='red', style='white'):
                image, label = image.reshape(-1, 1), label.reshape(-1, 1)

                output, hidden = self.forward(image)

                error = cost(output, label)
                total_error += error
                correct += int(np.argmax(output) == np.argmax(label))

                self.back(output - label, hidden, image)

            end_time, avg_error, accuracy = perf_counter(), total_error / n_samples, correct / n_samples * 100
            table.add_row(str(epoch + 1), f'{accuracy: .2f}', f'{avg_error: .6f}', f'{end_time - start_time : .4f}')
            os.system('clear || cls')
            console.print(table)

    def forward(self, image):
        hidden = self.input_hidden.activation(self.input_hidden.biases + self.input_hidden.weights @ image)
        output = self.hidden_output.activation(self.hidden_output.biases + self.hidden_output.weights @ hidden)
        return output, hidden

    def back(self, difference, hidden, image):
        delta_h = self.hidden_output.weights.T @ difference * self.input_hidden.activation(self.input_hidden.biases + self.input_hidden.weights @ image, True)
        self.hidden_output.weights -= self.learn_rate * difference @ hidden.T
        self.hidden_output.biases -= self.learn_rate * difference
        self.input_hidden.weights -= self.learn_rate * delta_h @ image.T
        self.input_hidden.biases -= self.learn_rate * delta_h
        
    def test(self):
        count, dataset = 0, self.test_data
        for i, (image, label) in enumerate(zip(dataset.images, dataset.labels)):
            output, _ = self.forward(image.reshape(-1, 1))
            count += int(np.argmax(output) == np.argmax(label))
        console.print(f'[bold]Test Accuracy:[/bold] [steel_blue]{count}/{i + 1} = [uu]{count/(i + 1): .2%}[/uu][/steel_blue]')
        console.print(f'\nAccuracy: {count/(i + 1): .2%} | Epochs: {self.epochs} | Learning Rate: {self.learn_rate} | Hidden Layer: {self.input_hidden.out_nodes} nodes')
        
net = Neural(epochs=1, learn_rate=0.018, nodes=[784, 30, 10], dataset_path='./dataset/mnist.npz')
net.train()
net.test()