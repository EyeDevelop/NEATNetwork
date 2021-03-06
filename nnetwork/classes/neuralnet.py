import pickle

from nnetwork.util.neuralnet import activation
from nnetwork.util import rng


class Neuron:
    def __init__(self, bias: float = None, activation_function: str = "sigmoid"):
        # Keep track of connections to other neurons and their weights.
        # Structure should be (node: Neuron, weight: float).
        self.connections = []

        # Keep track of the inputs (* weights) of other neurons.
        self.inputs = []
        self.value = 0.0

        # Keep track of nnetwork settings.
        self.activation_function = activation.activation_functions[activation_function]

        # Set the bias of the neuron.
        if bias is None:
            # It's between -1 and 1.
            self.bias = rng.random_number() * 2 - 1
        else:
            self.bias = bias

    # Return the value if requested.
    def get_value(self) -> float:
        return self.value

    # Set the value when requested.
    def set_value(self, value: float):
        self.value = value

    # Add to the current input list.
    def add_input(self, value: float):
        self.inputs.append(value)

    # Add a function to clear neuron inputs.
    def clear_inputs(self):
        self.inputs = []

    # Feed forward the data.
    def feed_forward(self):
        # Run the activation function before feed-forward.
        # But only if it's not an input neuron. Then the value is already set.
        self.value = self.activation_function(self.inputs)

        # Only feed-forward if the bias has been met.
        if self.value > self.bias:
            # Set the values of the neurons in the next layer.
            for connection in self.connections:
                next_neuron, weight = connection

                next_neuron.inputs.append(self.value * weight)

    # Create connection between other neurons.
    def connect(self, neuron_connector, weight=None):
        # Initialise weight if it's not specified.
        if weight is None:
            # Weight is between -1 and 1
            weight = rng.random_number() * 2 - 1

        # Add the actual connection.
        self.connections.append([neuron_connector, weight])


class Network:
    def __init__(self, hidden_layer_count: int, network_structure: list, activation_function: str = "sigmoid"):
        # Make the general nnetwork structure.
        self.layers: list = []
        self.make_layers(hidden_layer_count, network_structure, activation_function)
        self.connect_neurons()

    def make_layers(self, hidden_layer_count: int, network_structure: list, activation_function: str):
        for layer_index in range(hidden_layer_count + 2):  # The (+ 2) is for the input and output layer.
            # Create an intermediary list to keep neurons in.
            layer = []

            # Get the amount of neurons that need to be created for this layer.
            neuron_count = network_structure[layer_index]

            for neuron_index in range(neuron_count):
                # Add a neuron with random biases to the list.
                layer.append(Neuron(activation_function=activation_function))

            # Add the layer to the central layers list.
            self.layers.append(layer)

    def connect_neurons(self):
        # We connect the neurons from the output layer back.
        # So the range goes from len(self.layers) - 1 to 0.
        for layer_index in range(len(self.layers) - 1, 0, -1):
            # The connection is made from the layer before layer_index to the one at layer_index.
            for neuron in self.layers[layer_index - 1]:
                for last_layer_neuron in self.layers[layer_index]:
                    neuron.connect(last_layer_neuron)

    def make_prediction(self, input_values: list) -> list:
        # First reset all neuron values.
        for layer in self.layers:
            for neuron in layer:
                neuron.clear_inputs()

        # Set the values of the input neurons (first layer).
        for neuron_index in range(len(self.layers[0])):
            self.layers[0][neuron_index].add_input(input_values[neuron_index])

        # Feed forward the data
        for layer in self.layers:
            for neuron in layer:
                neuron.feed_forward()

        # The answer is now in the last layer, so get those values.
        values = [neuron.get_value() for neuron in self.layers[-1]]

        return values

    # Function to retrieve weights and biases.
    def get_weights_and_biases(self) -> tuple:
        # Make a list of biases and weights.
        weights = []
        biases = []

        # Get the actual weights and biases.
        for layer in self.layers:
            layer_weights = []
            layer_biases = []

            for neuron in layer:
                connection_weights = []
                layer_biases.append(neuron.bias)

                for connection in neuron.connections:
                    connection_weights.append(connection[1])
                layer_weights.append(connection_weights)

            weights.append(layer_weights)
            biases.append(layer_biases)

        return weights, biases

    # Save the nnetwork to a file.
    def save_network(self, filename: str = "nnetwork.pickle"):
        with open(filename, "wb") as fp:
            # Dump as pickle file.
            pickle.dump(self, fp)
