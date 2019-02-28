# A replica of biological genetic crossover applied to neural networks.
from network.classes.network import Network
from network.util import rng


def crossover(neat_object, network1: Network, network2: Network):
    # Make a list of weights of the first parent network.
    weights1 = []

    # Do the same for the second parent.
    weights2 = []

    # Get the weights of the first parent.
    for layer in network1.layers:
        layer_weights = []

        for neuron in layer:
            connection_weights = []

            for connection in neuron.connections:
                connection_weights.append(connection[1])
            layer_weights.append(connection_weights)
        weights1.append(layer_weights)

    # Get the weights of the second parent.
    for layer in network2.layers:
        layer_weights = []

        for neuron in layer:
            connection_weights = []

            for connection in neuron.connections:
                connection_weights.append(connection[1])
            layer_weights.append(connection_weights)
        weights2.append(layer_weights)

    # Decide on a split point
    split_point_1 = rng.randint(0, sum(neat_object.neuron_counts) // 2)
    split_point_2 = rng.randint(split_point_1, sum(neat_object.neuron_counts) - 1)

    # Keep track of whether the split has been surpassed.
    split_point = False

    # Also keep track of how many neurons have been passed.
    neurons_passed = 0

    # Create a child network.
    child = Network(neat_object.layer_count, neat_object.neuron_counts, activation_function=neat_object.activation_function)

    # Perform the actual crossover.
    for layer_index in range(len(weights1)):
        for neuron_index in range(len(weights1[layer_index])):
            for connection_index in range(len(weights1[layer_index][neuron_index])):
                # If the crossover point hasn't been reached yet, we use weights1 as parent.
                if not split_point:
                    parent = weights1
                else:
                    parent = weights2

                # Select the parent weight.
                parent_weight = parent[layer_index][neuron_index][connection_index]

                # Set the child neuron weights.
                child.layers[layer_index][neuron_index].connections[connection_index][1] = parent_weight

            neurons_passed += 1
            if neurons_passed >= split_point_1 and not split_point and not neurons_passed >= split_point_2:
                split_point = True
            elif neurons_passed >= split_point_2 and split_point:
                split_point = False

    return child


# Do fully random choice of weights.
def random_gene_copy(self, network1: Network, network2: Network):
    # Make a list of weights of the first parent network.
    weights1 = []

    # Do the same for the second parent.
    weights2 = []

    # Get the weights of the first parent.
    for layer in network1.layers:
        layer_weights = []

        for neuron in layer:
            connection_weights = []

            for connection in neuron.connections:
                connection_weights.append(connection[1])
            layer_weights.append(connection_weights)
        weights1.append(layer_weights)

    # Get the weights of the second parent.
    for layer in network2.layers:
        layer_weights = []

        for neuron in layer:
            connection_weights = []

            for connection in neuron.connections:
                connection_weights.append(connection[1])
            layer_weights.append(connection_weights)
        weights2.append(layer_weights)

    # Generate a child.
    child = Network(self.layer_count, self.neuron_counts, activation_function=self.activation_function)

    # Update the child weights.
    for layer_index in range(len(weights1)):
        for neuron_index in range(len(weights1[layer_index])):
            for connection_index in range(len(weights1[layer_index][neuron_index])):
                # Pick the weight randomly
                parent_weight = rng.choice([
                    weights1[layer_index][neuron_index][connection_index],
                    weights2[layer_index][neuron_index][connection_index],
                ])

                # Set the child weight
                child.layers[layer_index][neuron_index].connections[connection_index][1] = parent_weight

    return child
