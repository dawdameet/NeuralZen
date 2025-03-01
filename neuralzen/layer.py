#layer.py
from random import uniform
from neuron import Neuron

class Layer:
    def __init__(self, num_neurons, input_size):
        """
        Initialize a layer with `num_neurons` neurons.
        Each neuron has `input_size` weights and a bias.
        """
        self.neurons = [
            Neuron(weight=uniform(-0.1, 0.1), bias=uniform(-0.1, 0.1))
            for _ in range(num_neurons)
        ]

    def forward(self, inputs):
        """
        Forward pass for the layer.
        Each neuron processes all inputs and produces one output.
        """
        return [neuron.forward(inputs) for neuron in self.neurons]