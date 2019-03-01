# A replica of biological genetic crossover applied to neural networks.
from network.classes.network import Network
from network.util import rng


def crossover(neat_object, network1: Network, network2: Network):
    # Make a list of biases and weights of the first parent network.
    weights1 = []
    biases1 = []

    # Do the same for the second parent.
    weights2 = []
    biases2 = []

    # Get the weights of the first parent.
    for layer in network1.layers:
        layer_weights = []
        layer_biases = []

        for neuron in layer:
            connection_weights = []
            layer_biases.append(neuron.bias)

            for connection in neuron.connections:
                connection_weights.append(connection[1])
            layer_weights.append(connection_weights)

        weights1.append(layer_weights)
        biases1.append(layer_biases)

    # Get the weights of the second parent.
    for layer in network2.layers:
        layer_weights = []
        layer_biases = []

        for neuron in layer:
            connection_weights = []
            layer_biases.append(neuron.bias)

            for connection in neuron.connections:
                connection_weights.append(connection[1])
            layer_weights.append(connection_weights)

        weights2.append(layer_weights)
        biases2.append(layer_biases)

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
            # If the crossover point hasn't been reached yet, we use the first parent's biases.
            # Otherwise use the second parent's biases.
            if not split_point:
                parent_biases = biases1
            else:
                parent_biases = biases2

            for connection_index in range(len(weights1[layer_index][neuron_index])):
                # Do the same as above.
                if not split_point:
                    parent_weights = weights1
                else:
                    parent_weights = weights2

                # Select the parent weight.
                parent_weight = parent_weights[layer_index][neuron_index][connection_index]

                # Set the child neuron weights.
                child.layers[layer_index][neuron_index].connections[connection_index][1] = parent_weight

            # Get the parents neuron
            parent_bias = parent_biases[layer_index][neuron_index]

            # Set the child neuron bias.
            child.layers[layer_index][neuron_index].bias = parent_bias

            neurons_passed += 1
            if neurons_passed >= split_point_1 and not split_point and not neurons_passed >= split_point_2:
                split_point = True
            elif neurons_passed >= split_point_2 and split_point:
                split_point = False

    return child


# Do fully random choice of weights.
def random_gene_copy(neat_object, network1: Network, network2: Network):
    # Make a list of biases and weights of the first parent network.
    weights1 = []
    biases1 = []

    # Do the same for the second parent.
    weights2 = []
    biases2 = []

    # Get the weights of the first parent.
    for layer in network1.layers:
        layer_weights = []
        layer_biases = []

        for neuron in layer:
            connection_weights = []
            layer_biases.append(neuron.bias)

            for connection in neuron.connections:
                connection_weights.append(connection[1])
            layer_weights.append(connection_weights)

        weights1.append(layer_weights)
        biases1.append(layer_biases)

    # Get the weights of the second parent.
    for layer in network2.layers:
        layer_weights = []
        layer_biases = []

        for neuron in layer:
            connection_weights = []
            layer_biases.append(neuron.bias)

            for connection in neuron.connections:
                connection_weights.append(connection[1])
            layer_weights.append(connection_weights)

        weights2.append(layer_weights)
        biases2.append(layer_biases)

    # Generate a child.
    child = Network(neat_object.layer_count, neat_object.neuron_counts, activation_function=neat_object.activation_function)

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

            # Set the child neuron's bias
            child.layers[layer_index][neuron_index].bias = rng.choice([
                biases1[layer_index][neuron_index],
                biases2[layer_index][neuron_index],
            ])

    return child


# Random breeding function
def mutation_variants(neat_object, network1: Network, network2: Network):
    return neat_object.mutate(network1)
