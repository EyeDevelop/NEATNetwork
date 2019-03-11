from nnetwork.classes.neuralnet import Network
from nnetwork.util import rng


def crossover_wb(weights_and_biases1, weights_and_biases2, network_structure1, network_structure2):
    # Split the weights and biases.
    weights1, biases1 = weights_and_biases1
    weights2, biases2 = weights_and_biases2

    # Decide on a split point
    split_point_1 = rng.randint(0, sum(network_structure1) // 2)
    split_point_2 = rng.randint(split_point_1, sum(network_structure2) - 1)

    # Keep track of whether the split has been surpassed.
    split_point = False

    # Also keep track of how many neurons have been passed.
    neurons_passed = 0

    # Perform the actual crossover.
    weights = []
    biases = []

    for layer_index in range(network_structure1[0]):
        for neuron_index in range(network_structure2[0]):
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
                weights[layer_index][neuron_index].connections[connection_index][1] = parent_weight

            neurons_passed += 1
            if neurons_passed >= split_point_1 and not split_point and not neurons_passed >= split_point_2:
                split_point = True
            elif neurons_passed >= split_point_2 and split_point:
                split_point = False

    return weights, biases


def crossover(neat_object, network1: Network, network2: Network):
    # Get the parameters of the first nnetwork.
    parameters1 = network1.structure

    # Do the same for the second parent.
    parameters2 = network2.structure

    # Choose the nnetwork structure
    hidden_layer_count = rng.choice([
        parameters1[0],
        parameters2[0]
    ])

    neuron_counts = rng.choice([
        parameters1[1],
        parameters2[1]
    ])

    # Crosover the weights and biases.
    weights, biases = crossover_wb(weights_and_biases1, weights_and_biases2)

    return child
