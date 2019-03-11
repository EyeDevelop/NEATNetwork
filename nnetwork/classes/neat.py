import logging
import math
import pickle

from nnetwork.classes.neuralnet import Network
from nnetwork.util.neat import breeding
from nnetwork.util import rng


class NEAT:
    def __init__(self, input_size, output_size, population_size: int = 5000, mutation_chance: float = 0.02, mutation_severity: int = None, activation_function="tanh", breeding_function="crossover", console_log_level=logging.INFO, file_log_level=None):
        # Make a logger if requested.
        if console_log_level is not None or file_log_level is not None:
            self.logger = logging.getLogger("NEAT")
            self.logger.setLevel(logging.DEBUG)

            if console_log_level is not None:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(console_log_level)

            if file_log_level is not None:
                file_handler = logging.FileHandler("NEAT.log", mode='w')
                file_handler.setLevel(file_log_level)

            log_format = logging.Formatter("[%(name)s] %(asctime)s: %(levelname)s - %(message)s")

            if console_log_level is not None:
                console_handler.setFormatter(log_format)
                self.logger.addHandler(console_handler)

            if file_log_level is not None:
                file_handler.setFormatter(log_format)
                self.logger.addHandler(file_handler)

        # Keep track of the generation being trained and some scoring of the previous generation.
        self.generation = 0
        self.previous_generation_score = 0
        self.best_of_previous = 0

        # Store the input/output size.
        self.input_size = input_size
        self.output_size = output_size

        # Store the mutation settings.
        self.mutation_chance = mutation_chance

        # Choose a mutation function.
        if mutation_severity is not None:
            self.mutation_severity = mutation_severity
            self.mutation_func = self.mutate
        else:
            self.mutation_func = self.mutate_all

        # Store the population size
        self.population_size = population_size

        # Keep track of the current nnetwork being assessed.
        self.current_specimen = 0

        # Update nnetwork settings.
        self.activation_function = activation_function

        self.breeding_function = breeding.breeding_functions[breeding_function]

        # Make a list of networks in the current generation.
        self.specimen = []
        self.specimen_fitness = {}
        self.reset_generation()

        self.log(f"Setting up population with: Size: {self.population_size}, Mutation: {self.mutation_chance * 100}%")

    # A helper function to make logging easier.
    def log(self, msg, level=logging.INFO):
        self.logger.log(level, msg)

    # This prepares generation 0.
    def reset_generation(self):
        self.log("Preparing population for first use...")

        # Generate the random networks and store them in the specimen list.
        for _ in range(self.population_size):
            # First generate a nnetwork structure randomly.
            # Rounding happens because there can be no float count of layers.
            # The max happens because there can be no negative counts.
            hidden_layer_count = round(max(0, rng.random_gaussian(mean=0, std_deviation=2)))

            # Make a random amount of neurons per layer.
            network_structure = [self.input_size]
            for _ in range(hidden_layer_count):
                neurons = round(max(0, rng.random_gaussian(mean=5, std_deviation=5)))
                network_structure.append(neurons)
            network_structure.append(self.output_size)

            network = Network(hidden_layer_count, network_structure, activation_function=self.activation_function)
            network.structure = (hidden_layer_count, network_structure, network.get_weights_and_biases())

            self.specimen.append(network)

        self.log("Population generated.")

    # The function to determine the fitness of a nnetwork is different each time,
    # so this function needs to be abstract.
    def fitness(self, inputs: list, outputs: list):
        raise NotImplementedError()

    # A basic redirection function that allows the fitness function to be written more easily.
    def train(self, inputs: list):
        # Validate the nnetwork against the fitness function.
        network_output = self.specimen[self.current_specimen].make_prediction(inputs)
        fitness = self.fitness(inputs, network_output)

        # Store the fitness of the current nnetwork.
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
        self.log("Starting breeding process...")
        self.log(f"Generation average: {sum(self.specimen_fitness.values()) / self.population_size}. Best: {sorted(self.specimen_fitness.values(), reverse=True)[0]}")

        # Sort the networks to get the top networks.
        specimen_sorted = sorted(self.specimen_fitness.keys(), key=lambda x: self.specimen_fitness[x], reverse=True)

        # Store the score of the best nnetwork of the previous generation.
        self.best_of_previous = self.specimen_fitness[specimen_sorted[0]]

        # The best of the generation is copied over without crossover or mutation.
        # Make a list of the new generation.
        best_network = self.specimen[specimen_sorted[0]]
        new_generation = [best_network]

        # Start generating population_size children based on the previous generation
        for _ in range(self.population_size - 1):
            parent1 = self.specimen[self.choose_parent()]
            parent2 = self.specimen[self.choose_parent()]

            # Make a child based on the parents.
            child = self.breeding_function(self, parent1, parent2)

            # Mutate the child.
            child = self.mutation_func(child)

            # Add it to the specimen list.
            new_generation.append(child)

        # Set the specimen list.
        self.specimen = new_generation

        # Add 1 to the generation counter.
        self.generation += 1

        # Reset the fitness dictionary.
        self.specimen_fitness = {}

        self.log("Breeding finished.")

    # A function to choose a parent.
    def choose_parent(self):
        # Generate a random choosing point.
        fitness_sum = math.floor(sum(self.specimen_fitness.values()))
        passing_point = rng.randint(0, fitness_sum)

        # Keep track of a running sum
        running_sum = 0

        for specimen_id in range(len(self.specimen)):
            # Add the current fitness to the running sum.
            running_sum += self.specimen_fitness[specimen_id]

            # Check if the running sum passed the passing point.
            # If it has, return the specimen id that passed.
            if running_sum > passing_point:
                self.log(f"Parent {specimen_id} has been chosen with fitness {self.specimen_fitness[specimen_id]}", level=logging.DEBUG)
                return specimen_id

    # A function to apply random mutation to networks to provide the genetic variation.
    def mutate(self, network):
        self.log("Starting mutation...", level=logging.DEBUG)

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
                    weight_delta = max(-1, min(current_weight + rng.random_gaussian() / 5, 1))
                    network.layers[mutation_layer][mutation_neuron].connections[mutation_connection][1] += weight_delta

        return network

    # A function to apply mutation randomly to networks to provide the genetic variation.
    def mutate_all(self, network):
        self.log("Starting mutation...", level=logging.DEBUG)

        # Mutate the nnetwork parameters.
        for parameter_id in range(len(network.structure)):
            if rng.random_number() <= self.mutation_chance:
                current_parameter = network.structure[parameter_id]
                new_parameter = current_parameter + round(rng.random_gaussian())

                network.structure[parameter_id] = new_parameter

        # Mutate the weights and biases.
        for layer_index in range(len(network.layers)):
            for neuron_index in range(len(network.layers[layer_index])):
                # Only mutate bias if chance is met.
                if rng.random_number() <= self.mutation_chance:
                    # Get the current bias.
                    current_bias = network.layers[layer_index][neuron_index].bias

                    # Add a random Gaussian variable to that.
                    new_bias = current_bias + rng.random_gaussian() / 5

                    # Set the new bias (with a max of 1 and a min of -1)
                    network.layers[layer_index][neuron_index].bias = max(-1, min(1, current_bias + new_bias))

                # Go through the connections to mutate them.
                for connection_index in range(len(network.layers[layer_index][neuron_index].connections)):
                    # Only mutate if chance is met.
                    if rng.random_number() <= self.mutation_chance:
                        # Get the current weight.
                        current_weight = network.layers[layer_index][neuron_index].connections[connection_index][1]

                        # Add a random Gaussian variable to that.
                        new_weight = current_weight + rng.random_gaussian() / 5

                        # Set the new weight (with a max of 1 and a min of -1).
                        network.layers[layer_index][neuron_index].connections[connection_index][1] = max(-1, min(1, new_weight))

        return network

    # Save the nnetwork to a file.
    def save_network(self, filename: str = "neat.pickle"):
        # Open the file.
        with open(filename, "wb") as fp:
            # Store it as pickle object.
            pickle.dump(self, fp)
