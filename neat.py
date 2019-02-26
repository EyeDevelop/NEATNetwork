import random
from network import Network


class NEAT:
    def __init__(self, layer_count: int, neuron_counts: list, population_size: int = 50, mutation_chance: int = 20, mutation_severity: int = 3):
        # Keep track of the generation being trained and the previous score.
        self.generation = 0
        self.previous_generation_score = 0

        # Store the network structure.
        self.layer_count = layer_count
        self.neuron_counts = neuron_counts

        # Store the mutation settings.
        self.mutation_chance = mutation_chance
        self.mutation_severity = mutation_severity

        # Store the population size
        self.population_size = population_size

        # Keep track of the current network being assessed.
        self.current_specimen = 0

        # Make a list of networks in the current generation.
        self.specimen = []
        self.specimen_fitness = {}
        self.reset_generation()

    # This prepares generation 0.
    def reset_generation(self):
        # Generate the random networks and store them in the specimen list.
        for _ in range(self.population_size):
            self.specimen.append(Network(self.layer_count, self.neuron_counts))

    # The function to determine the fitness of a network is different each time,
    # so this function needs to be abstract.
    def fitness(self, inputs: list, outputs: list):
        raise NotImplementedError()

    # A basic redirection function that allows the fitness function to be written more easily.
    def train(self, inputs: list):
        # Validate the network against the fitness function.
        network_output = self.specimen[self.current_specimen].make_prediction(inputs)
        fitness = self.fitness(inputs, network_output)

        # Store the fitness of the current network.
        self.specimen_fitness[self.specimen[self.current_specimen]] = fitness

        # Go on to the next specimen.
        self.next_specimen()

    # A helper function to shift to the next specimen.
    def next_specimen(self):
        # Check if the entire generation has been ran.
        # If it has been, breed the networks to generate a new generation.
        if self.current_specimen >= self.population_size - 1:
            # Store the sum of fitness as a generation fitness score.
            self.previous_generation_score = sum(self.specimen_fitness.values())

            # Make a new generation.
            self.breed()

            # Reset the counter.
            self.current_specimen = 0
        else:
            self.current_specimen += 1

    # Breed to networks with crossover.
    def breed(self):
        # Sort the networks to get the two best.
        specimen_sorted = sorted(self.specimen_fitness, key=lambda x: self.specimen_fitness[x])

        # Reset the specimen list.
        self.specimen = []

        # Start generating population_size children based on the two best in the previous generation.
        for _ in range(self.population_size):
            child = self.crossover(specimen_sorted[0], specimen_sorted[1])
            self.specimen.append(child)

        # Add 1 to the generation counter.
        self.generation += 1

    # A replica of biological genetic crossover applied to neural networks.
    def crossover(self, network1: Network, network2: Network):
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
        split_point = random.randint(0, sum(self.neuron_counts) - 1)

        # Keep track of whether the split has been surpassed.
        split_point_passed = False

        # Also keep track of how many neurons have been passed.
        neurons_passed = 0

        # Create a child network.
        child = Network(self.layer_count, self.neuron_counts)

        # Perform the actual crossover.
        for layer_index in range(len(weights1)):
            for neuron_index in range(len(weights1[layer_index])):
                for connection_index in range(len(weights1[layer_index][neuron_index])):
                    # If the crossover point hasn't been reached yet, we use weights1 as parent.
                    if not split_point_passed:
                        parent = weights1
                    else:
                        parent = weights2

                    # Select the parent weight.
                    parent_weight = parent[layer_index][neuron_index][connection_index]

                    # Set the child neuron weights.
                    child.layers[layer_index][neuron_index].connections[connection_index][1] = parent_weight

                neurons_passed += 1
                if neurons_passed >= split_point and not split_point_passed:
                    split_point_passed = True

        # Mutate the child.
        child = self.mutate(child)

        return child

    # A function to apply random mutation to networks to provide the genetic variation.
    def mutate(self, network):
        # Only mutate if the chance is met.
        if random.randint(0, self.mutation_chance) == 0:

            # Mutate self.mutation_severity times
            for _ in range(self.mutation_severity):
                # Get the neuron that is going to be mutated.
                mutation_layer = random.randint(0, len(network.layers) - 1)
                mutation_neuron = random.randint(0, len(network.layers[mutation_layer]) - 1)

                # Mutate the bias or mutate the weight?
                mutate_bias = random.randint(0, 1) == 1

                if mutate_bias:
                    # Mutate the bias to a new random value.
                    network.layers[mutation_layer][mutation_neuron].bias = random.random()
                else:
                    # Cannot mutate a connection that doesn't exist (for example a neuron in the output layer). Perform that check first.
                    if len(network.layers[mutation_layer][mutation_neuron].connections) <= 0:
                        continue

                    mutation_connection = random.randint(0, len(network.layers[mutation_layer][mutation_neuron].connections) - 1)

                    # Add the mutation to the current value, to make a small change.
                    network.layers[mutation_layer][mutation_neuron].connections[mutation_connection][1] *= (random.random() * 2 - 1)

        return network
