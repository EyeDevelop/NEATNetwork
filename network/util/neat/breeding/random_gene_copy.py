# Do fully random choice of weights.
from network.classes.neuralnet import Network
from network.util import rng


def random_gene_copy(genn_object, network1: Network, network2: Network):
    # Get the weights and biases of the first parent.
    weights1, biases1 = network1.get_weights_and_biases()

    # Do the same for the second parent.
    weights2, biases2 = network2.get_weights_and_biases()

    # Generate a child.
    child = Network(genn_object.layer_count, genn_object.neuron_counts, activation_function=genn_object.activation_function)

    # Update the child weights.
    for layer_index in range(len(weights1)):
        for neuron_index in range(len(weights1[layer_index])):
            parent_choice = rng.randint(0, 1)

            for connection_index in range(len(weights1[layer_index][neuron_index])):
                # Pick the weight randomly
                parent_weight = [
                    weights1[layer_index][neuron_index][connection_index],
                    weights2[layer_index][neuron_index][connection_index],
                ][parent_choice]

                # Set the child weight
                child.layers[layer_index][neuron_index].connections[connection_index][1] = parent_weight

            # Set the child neuron's bias
            child.layers[layer_index][neuron_index].bias = [
                biases1[layer_index][neuron_index],
                biases2[layer_index][neuron_index],
            ][parent_choice]

    return child
