#network.py
from neuron import Neuron
from layer import Layer
from random import random, randint
class Network:
    def __init__(self, input_layer, output_layer):
        self.input_layer=input_layer
        self.output_layer=output_layer
        self.hiddenLayers=[]  
    def create_hiddenlayer(self, num_layers):
        self.hiddenLayers =  [Layer(randint(10,20)) for _ in range(num_layers)]
    def forward(self, inputs):
        output=self.input_layer.forward(inputs)
        for layer in self.hiddenLayers:
            out=layer.forward(out)
        return self.output_layer.forward(out)