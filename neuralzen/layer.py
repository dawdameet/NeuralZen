#layer.py
from random import random, randint
import neuron
class Layer:
    def __init__(self, num_neurons):
        self.neurons=[neuron.Neuron(random(),randint(1, 10)) for _ in range(num_neurons)]
    def forward(self, inputs):
        return [neuron.forward(inputs[i]) for i, neuron in enumerate(self.neurons)]
    