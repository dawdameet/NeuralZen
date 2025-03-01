from neuralzen.network import Network
from neuralzen.layer import Layer
input_layer = Layer(num_neurons=3, input_size=1)
output_layer = Layer(num_neurons=1, input_size=3)
net = Network(input_layer, output_layer)
net.create_hidden_layers(num_layers=2, input_size=3)
sample_input = [0.5, -1.2, 0.8]
output = net.forward(sample_input)
print("Network Output:", output)