#neuron.py
from activations import Activations

class Neuron:
    def __init__(self, weight, bias, inp=0, func=0):
        """0-ReLu::1-Sigmoid"""
        self.weight = weight
        self.bias = bias
        self.inp = inp
        self.function = func

    def forward(self, inp):
        z = self.weight * inp + self.bias
        if self.function == 0:
            return Activations.ReLu(z)
        elif self.function == 1:
            return Activations.sigmoid(z)
        else:
            return z