# neuron.py
import activations
class Neuron:
    def __init__(self, weight, bias, inp=0, func=0):
        """0-ReLu::1-Sigmoid::2-XOR"""
        self.weight=weight
        self.bias=bias
        self.inp=inp
        self.function=func


    def forward(self, inp):
        z=self.weight * inp + self.bias 
        if self.function==0:
            return activations.ReLu(z)
        elif self.function==1:
            return activations.sigmoid(z)
        else:
            return z
        