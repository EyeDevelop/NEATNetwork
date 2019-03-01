import pickle

from network.classes.network import Network
from network.util import rng, breed


class NEAT:
    def __init__(self, layer_count: int, neuron_counts: list, population_size: int = 50, mutation_chance: float = 0.02, mutation_severity: int = 3, retention_rate=5, activation_function="tanh", breeding_function="crossover"):
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
        self.retention_rate = retention_rate

        # Keep track of the current network being assessed.
        self.current_specimen = 0

        # Update network settings.
        self.activation_function = activation_function

        self.breeding_function = {
            "crossover": breed.crossover,
            "random_gene_copy": breed.random_gene_copy
        }[breeding_function]

        # Make a list of networks in the current generation.
        self.specimen = []
        self.specimen_fitness = {}
        self.reset_generation()

    # This prepares generation 0.
    def reset_generation(self):
        # Generate the random networks and store them in the specimen list.
        for _ in range(self.population_size):
            self.specimen.append(Network(self.layer_count, self.neuron_counts, activation_function=self.activation_function))

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
        self.specimen_fitness[self.current_specimen] = fitness

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
        specimen_sorted = sorted(self.specimen_fitness.keys(), key=lambda x: self.specimen_fitness[x], reverse=True)

        # Get the two best networks.
        parent1 = self.specimen[specimen_sorted[0]]
        parent2 = self.specimen[specimen_sorted[1]]

        # Make a placeholder list to keep the best networks in.
        retention_specimen = []

        # The best retention_rate networks will be put in the next generation.
        for key in specimen_sorted[:self.retention_rate]:
            retention_specimen.append(self.specimen[key])

        # Reset the specimen list.
        self.specimen = retention_specimen

        # Mutate the networks copied over.
        for specimen_index in range(len(self.specimen)):
            # Mutate the network.
            mutated_network = self.mutate(self.specimen[specimen_index])

            # Replace the un-mutated network.
            self.specimen[specimen_index] = mutated_network

        # Let 50% of the next generation be created by the two best.
        # The other 50% by the other networks in the retention_specimen list.
        half_index = (self.population_size - self.retention_rate) // 2

        # Start generating population_size children based on the two best in the previous generation.
        for i in range(self.population_size - self.retention_rate):
            if i >= half_index:
                parent1 = rng.choice(retention_specimen)
                parent2 = rng.choice(retention_specimen)

            # Make a child based on the parents.
            child = self.breeding_function(self, parent1, parent2)

            # Mutate the child.
            child = self.mutate(child)

            # Add it to the specimen list.
            self.specimen.append(child)

        # Add 1 to the generation counter.
        self.generation += 1

        # Reset the fitness dictionary.
        self.specimen_fitness = {}

    # A function to apply random mutation to networks to provide the genetic variation.
    def mutate(self, network):
        # Only mutate if the chance is met.
        if rng.random_number() <= self.mutation_chance:

            # Mutate self.mutation_severity times
            for _ in range(self.mutation_severity):
                # Get the neuron that is going to be mutated.
                mutation_layer = rng.randint(0, len(network.layers) - 1)
                mutation_neuron = rng.randint(0, len(network.layers[mutation_layer]) - 1)

                # Mutate the bias or mutate the weight?
                mutate_bias = rng.randint(0, 1) == 1

                if mutate_bias:
                    # Mutate the bias to a new random value.
                    network.layers[mutation_layer][mutation_neuron].bias = rng.random_number()
                else:
                    # Cannot mutate a connection that doesn't exist (for example a neuron in the output layer). Perform that check first.
                    if len(network.layers[mutation_layer][mutation_neuron].connections) <= 0:
                        continue

                    mutation_connection = rng.randint(0, len(network.layers[mutation_layer][mutation_neuron].connections) - 1)

                    # Add the mutation to the current value, to make a small change.
                    network.layers[mutation_layer][mutation_neuron].connections[mutation_connection][1] *= (rng.random_number() * 2 - 1)

        return network

    # Save the network to a file.
    def save_network(self, filename: str = "neat.pickle"):
        # Open the file.
        with open(filename, "wb") as fp:
            # Store it as pickle object.
            pickle.dump(self, fp)
