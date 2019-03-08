import os
import pickle
import queue
import select
import socket
import logging
import traceback
import errno

from network.classes.neat import NEAT


class CustomNEAT(NEAT):

    def __init__(self, layer_count: int, neuron_counts: list, population_size: int = 50, breed_using=0.35, mutation_chance: float = 0.02, mutation_severity: int = 3, activation_function="tanh", breeding_function="crossover", data_filename="data.csv"):
        super().__init__(layer_count, neuron_counts, population_size, breed_using, mutation_chance, mutation_severity, activation_function, breeding_function)
        self.data_filename = data_filename

        # Make an empty file and write the CSV header.
        try:
            with open(self.data_filename, 'wt') as fp:
                fp.write("Generation,Score,BestScore\n")
        except OSError:
            print("No write permissions for data.csv!")
            exit(1)

    def fitness(self, inputs: list, outputs: list):
        pass

    def breed(self):
        # Append data to data.csv.
        with open(self.data_filename, 'at') as fp:
            fp.write("{},{},{}\n".format(
                self.generation,
                sum(self.specimen_fitness.values()),
                sorted(self.specimen_fitness.values(), reverse=True)[0]
            ))

        # Store the very best to a file.
        # Create a directory for them if it doesn't exist already.
        if not os.path.exists("BestNetworks"):
            os.mkdir("BestNetworks")

        # Save the best.
        with open(os.path.join("BestNetworks", f"{self.generation}.pickle"), 'wb') as fp:
            best_id = sorted(self.specimen_fitness.keys(), key=lambda x: self.specimen_fitness[x])[0]
            best_network = self.specimen[best_id]

            pickle.dump(best_network, fp)

        # Run the original breeding function.
        super().breed()


class SnekAI:
    def __init__(self, load_neat_file="", log_everything=False):
        # Check if the file exists, and if it does, load it.
        neat_loaded = False

        if load_neat_file:
            if os.path.exists(load_neat_file):
                with open(load_neat_file, "rb") as fp:
                    self.neat_object = pickle.load(fp)
                    neat_loaded = True

        # If it failed to load the file, generate a new NEAT object.
        if not neat_loaded:
            self.neat_object = CustomNEAT(layer_count=2, neuron_counts=[24, 18, 18, 4], population_size=1000, breed_using=0.4, mutation_chance=0.02, activation_function="sigmoid", breeding_function="crossover")

        # Make a central server socket.
        self.server_socket = None

        # Keep a write queue to write to clients.
        self.write_queue = {}

        # Keep track of the current score of the AI.
        self.current_score = 0

        # Make a logger if requested.
        if log_everything:
            self.logger = logging.getLogger("SnekAI")
            self.logger.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            file_handler = logging.FileHandler("snekai.log", mode='w')
            file_handler.setLevel(logging.DEBUG)

            log_format = logging.Formatter("[%(name)s] %(asctime)s: %(levelname)s - %(message)s")

            console_handler.setFormatter(log_format)
            file_handler.setFormatter(log_format)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    # Make a function to log.
    def log(self, loglevel, message):
        self.logger.log(loglevel, message)

    # This function is obsolete, learning is handled by networked functions.
    def fitness(self, inputs: list, outputs: list):
        pass

    # Run the AI over the network.
    def start_server(self, port=6969):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        with self.server_socket:
            inputs = [self.server_socket]
            outputs = []

            # Start accepting connections.
            self.server_socket.setblocking(False)
            self.server_socket.bind(('', port))
            self.server_socket.listen(1)

            self.log(logging.INFO, f"Ready for connections on {port}.")

            while inputs:
                readable, writable, exceptions = select.select(inputs, outputs, inputs)

                for s in readable:
                    if s is self.server_socket:
                        # Accept a new client if available.
                        c_sock, _ = self.server_socket.accept()

                        self.log(logging.INFO, "Snek Game connected.")
                        self.log(logging.INFO, "Starting with:")
                        self.log(logging.INFO, f"Generation: {self.neat_object.generation}")
                        self.log(logging.INFO, f"Individual: {self.neat_object.current_specimen}\n")

                        c_sock.setblocking(False)

                        inputs.append(c_sock)
                    else:
                        # Read the data sent to us.
                        data = s.recv(1024)

                        # The data has to be something, otherwise the client isn't connected.
                        if data:
                            # Add s to the write queue after receiving.
                            if s not in self.write_queue.keys():
                                self.write_queue[s] = queue.Queue()

                            # Check if the AI died.
                            if "DEAD" in data.decode("utf-8"):
                                self.handle_death()
                                self.write_queue[s].put(
                                    ";".join(map(str, ["U", self.neat_object.generation, self.neat_object.current_specimen, self.neat_object.previous_generation_score / self.neat_object.population_size]))
                                )  # Otherwise Unity dies.
                            else:
                                # Let the AI make a move.
                                response = self.parse_game_data(data)
                                self.write_queue[s].put(response)

                            if s not in outputs:
                                outputs.append(s)
                        else:
                            # The client is disconnected. Remove from lists and break the connection.
                            if s in outputs:
                                outputs.remove(s)

                            inputs.remove(s)
                            s.close()

                            if s in self.write_queue.keys():
                                del self.write_queue[s]

                            self.log(logging.INFO, "Lost connection to Snek.")

                for s in writable:
                    # Write back to the client if a message awaits them.
                    if s in self.write_queue.keys():
                        try:
                            # Get the message waiting to be sent from the queue.
                            next_message = self.write_queue[s].get_nowait()
                        except queue.Empty:
                            outputs.remove(s)
                        else:
                            # Send the message.
                            s.send(next_message.encode("utf-8"))

                for s in exceptions:
                    # Something went wrong with the client. Break off connection.
                    if s in outputs:
                        outputs.remove(s)

                    inputs.remove(s)
                    s.close()
                    del self.write_queue[s]

                    self.log(logging.INFO, "Lost connection to Snek.")

    def parse_game_data(self, data):
        # Data is separated by a semicolon.
        data = data.decode("utf-8").split(';')

        # Define a placeholder for the return data.
        return_data = ""

        self.log(logging.DEBUG, len(data))

        # Data is valid.
        if len(data) == 25:
            # All data in data is of type int.
            data = [float(x) for x in data]

            # Separate the data passed into the AI, and the ones used for the fitness function.
            guess = self.neat_object.specimen[self.neat_object.current_specimen].make_prediction(data[:24])  # data is the surroundings of the snake, together with the distance differentials.
            current_score = data[24]  # data[24] is de score.

            # Update the current score variable.
            self.current_score = current_score

            # Return the move the game makes.  (0 = left, 1 = right, 2 = up, 3 = down)
            move = "LRUD"[guess.index(max(guess))]

            # Send the move, plus the generation and individual number back.
            return_data = ";".join(map(str, [move, self.neat_object.generation, self.neat_object.current_specimen, self.neat_object.previous_generation_score / self.neat_object.population_size]))
            self.log(logging.DEBUG, f"Received this data: {repr(data)}")
            self.log(logging.DEBUG, f"Sending this data back: {return_data}")
        else:
            self.log(logging.WARN, f"Received invalid data!: {repr(data)}")

        return return_data

    # A function to handle a client disconnect, meaning the AI died.
    def handle_death(self):
        # Calculate the fitness of the network.
        fitness = self.current_score

        # Store that in the global fitness dictionary.
        self.neat_object.specimen_fitness[self.neat_object.current_specimen] = fitness

        # Let the next AI have a go.
        self.neat_object.next_specimen()

        # Notify that the AI died.
        self.log(logging.INFO, "The AI died.")
        self.log(logging.INFO, f"Score: {self.current_score}, Best of Previous Generation: {self.neat_object.best_of_previous}, Average of Previous: {round(self.neat_object.previous_generation_score / self.neat_object.population_size)}")

        # Show the next individual having a go.
        self.log(logging.INFO, "Now trying:")
        self.log(logging.INFO, f"Generation: {self.neat_object.generation}")
        self.log(logging.INFO, f"Individual: {self.neat_object.current_specimen}\n")


def main(s: SnekAI):
    try:
        s.start_server(port=6969)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            s.start_server(port=6968)
        else:
            print(traceback.format_exc())


if __name__ == "__main__":
    # Make a SnakeAI object and try to resume where it left off training.
    s = SnekAI(log_everything=False)

    try:
        # Run the main function.
        main(s)

    except Exception as e:
        s.log(logging.ERROR, "Crashed!")
        s.log(logging.ERROR, traceback.format_exc())
        s.server_socket.close()
        s.log(logging.ERROR, "Exiting...")
        exit(1)

    except KeyboardInterrupt:
        s.log(logging.INFO, "Interrupt received.")

        s.log(logging.INFO, "Saving NEAT to file...")
        s.neat_object.save_network()

        s.log(logging.INFO, "Closing server socket...")
        s.server_socket.close()

        s.log(logging.INFO, "Exiting...")
        exit(0)
