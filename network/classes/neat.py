import math
import pickle

from network.classes.network import Network
from network.util import breeding
from network.util import rng


class NEAT:
    def __init__(self, layer_count: int, neuron_counts: list, population_size: int = 50, breed_using=0.35, mutation_chance: float = 0.02, mutation_severity: int = 3, activation_function="tanh", breeding_function="crossover"):
        # Keep track of the generation being trained and some scoring of the previous generation.
        self.generation = 0
        self.previous_generation_score = 0
        self.best_of_previous = 0

        # Store the network structure.
        self.layer_count = layer_count
        self.neuron_counts = neuron_counts

        # Store the mutation settings.
        self.mutation_chance = mutation_chance
        self.mutation_severity = mutation_severity
        self.breed_using = breed_using

        # Store the population size
        self.population_size = population_size

        # Keep track of the current network being assessed.
        self.current_specimen = 0

        # Update network settings.
        self.activation_function = activation_function

        self.breeding_function = breeding.breeding_functions[breeding_function]

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
        # Sort the networks to get the top networks.
        specimen_sorted = sorted(self.specimen_fitness.keys(), key=lambda x: self.specimen_fitness[x], reverse=True)

        # Store the score of the best network of the previous generation.
        self.best_of_previous = self.specimen_fitness[specimen_sorted[0]]

        # The best of the generation is copied over without crossover or mutation.
        # Make a list of the new generation.
        best_network = self.specimen[specimen_sorted[0]]
        new_generation = [best_network]

        # Pick from the best breed_using of the population.
        half_index = math.floor(len(specimen_sorted) * self.breed_using)
        specimen_sorted = specimen_sorted[:half_index]

        # For memory efficiency, remove the networks that didn't pass the fitness barrier.
        self.specimen = [self.specimen[x] for x in specimen_sorted]

        # Start generating population_size children based on the previous generation
        for _ in range(self.population_size - 1):
            parent1 = rng.choice(self.specimen)
            parent2 = rng.choice(self.specimen)

            # Make a child based on the parents.
            child = self.breeding_function(self, parent1, parent2)

            # Mutate the child.
            child = self.mutate(child)

            # Add it to the specimen list.
            new_generation.append(child)

        # Set the specimen list.
        self.specimen = new_generation

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
                    # Mutate the bias to a new random value between -1 and 1.
                    network.layers[mutation_layer][mutation_neuron].bias = rng.random_number() * 2 - 1
                else:
                    # Cannot mutate a connection that doesn't exist (for example a neuron in the output layer). Perform that check first.
                    if len(network.layers[mutation_layer][mutation_neuron].connections) <= 0:
                        continue

                    mutation_connection = rng.randint(0, len(network.layers[mutation_layer][mutation_neuron].connections) - 1)

                    # Add the mutation to the current value, to make a small change.
                    # Have a maximum value of 1 and a minimum of -1.
                    current_weight = network.layers[mutation_layer][mutation_neuron].connections[mutation_connection][1]
                    weight_delta = max(-1, min(current_weight + (rng.random_number() * 2 - 1), 1))
                    network.layers[mutation_layer][mutation_neuron].connections[mutation_connection][1] += weight_delta

        return network

    # Save the network to a file.
    def save_network(self, filename: str = "neat.pickle"):
        # Open the file.
        with open(filename, "wb") as fp:
            # Store it as pickle object.
            pickle.dump(self, fp)
