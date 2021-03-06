from nnetwork.classes.neuralnet import Network
from nnetwork.util import rng


def crossover(genn_object, network1: Network, network2: Network):
    # Get the weights and biases of the first parent.
    weights1, biases1 = network1.get_weights_and_biases()

    # Do the same for the second parent.
    weights2, biases2 = network2.get_weights_and_biases()

    # Decide on a split point
    split_point_1 = rng.randint(0, sum(genn_object.network_structure) // 2)
    split_point_2 = rng.randint(split_point_1, sum(genn_object.network_structure) - 1)

    # Keep track of whether the split has been surpassed.
    split_point = False

    # Also keep track of how many neurons have been passed.
    neurons_passed = 0

    # Create a child nnetwork.
    child = Network(genn_object.hidden_layer_count, genn_object.network_structure, activation_function=genn_object.activation_function)

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
