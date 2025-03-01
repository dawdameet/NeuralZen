#network.py

from layer import Layer
from random import randint

class Network:
    def __init__(self, input_layer, output_layer):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.hidden_layers = []

    def create_hidden_layers(self, num_layers, input_size):
        """
        Create hidden layers with random numbers of neurons.
        """
        self.hidden_layers = [
            Layer(randint(10, 20), input_size=input_size) for _ in range(num_layers)
        ]

    def forward(self, inputs):
        """
        Forward pass through the entire network.
        """
        output = self.input_layer.forward(inputs)
        for layer in self.hidden_layers:
            output = layer.forward(output)
        return self.output_layer.forward(output)