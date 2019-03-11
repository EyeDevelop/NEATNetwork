import errno
import logging
import os
import pickle
import queue
import select
import socket
import time
import traceback

from nnetwork.classes.gennetic import GeNNetic


class CustomGeNN(GeNNetic):

    def __init__(self, hidden_layer_count: int, network_structure: list, population_size: int = 50, mutation_chance: float = 0.02, mutation_severity: int = 3, activation_function="tanh", breeding_function="crossover", console_log_level=logging.INFO, file_log_level=None, data_filename="data.csv"):
        super().__init__(hidden_layer_count, network_structure, population_size, mutation_chance, mutation_severity, activation_function, breeding_function, console_log_level, file_log_level)
        self.data_filename = data_filename

        # Make an empty file and write the CSV header.
        with open(self.data_filename, 'wt') as fp:
            fp.write("Generation,Score,BestScore\n")

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
    def __init__(self, load_genn_file="", console_log_level=logging.INFO, file_log_level=None):
        # Check if the file exists, and if it does, load it.
        genn_loaded = False

        if load_genn_file:
            if os.path.exists(load_genn_file):
                with open(load_genn_file, "rb") as fp:
                    self.genn_object = pickle.load(fp)
                    genn_loaded = True

        # If it failed to load the file, generate a new genn object.
        if not genn_loaded:
            self.genn_object = CustomGeNN(hidden_layer_count=3, network_structure=[24, 40, 40, 40, 4], population_size=2000, mutation_chance=0.05, activation_function="sigmoid", breeding_function="crossover", console_log_level=console_log_level, file_log_level=file_log_level, data_filename=f"{__file__}.csv")

        # Make a central server socket.
        self.server_socket = None

        # Keep a write queue to write to clients.
        self.write_queue = {}
        self.last_read_from = {}

        # Keep track of the current score of the AI.
        self.current_score = 0

        # Make a logger if requested.
        if console_log_level is not None or file_log_level is not None:
            self.logger = logging.getLogger("SnekAI")
            self.logger.setLevel(logging.DEBUG)

            if console_log_level is not None:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(console_log_level)

            if file_log_level is not None:
                file_handler = logging.FileHandler("genn.log", mode='w')
                file_handler.setLevel(file_log_level)

            log_format = logging.Formatter("[%(name)s] %(asctime)s: %(levelname)s - %(message)s")

            if console_log_level is not None:
                console_handler.setFormatter(log_format)
                self.logger.addHandler(console_handler)

            if file_log_level is not None:
                file_handler.setFormatter(log_format)
                self.logger.addHandler(file_handler)

    # Make a function to log.
    def log(self, loglevel, message):
        self.logger.log(loglevel, message)

    # This function is obsolete, learning is handled by networked functions.
    def fitness(self, inputs: list, outputs: list):
        pass

    # Run the AI over the nnetwork.
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

                        self.log(logging.INFO, f"Snek Game connected. Starting with Gen {self.genn_object.generation} and Ind {self.genn_object.current_specimen}")

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
                                self.handle_death(s, data.split(";")[1])
                            else:
                                # Let the AI make a move.
                                response = self.parse_game_data(data)
                                self.write_queue[s].put(response)

                            self.last_read_from[s] = time.time()

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

                for s in self.last_read_from.keys():
                    # Check the time we last sent a message to Snek.
                    if time.time() - self.last_read_from[s] >= 30:
                        # Wake up Snek if it is stuck in a Receive() loop.
                        if s not in self.write_queue.keys():
                            self.write_queue[s] = queue.Queue()

                        self.write_queue[s].put("{};{};{};{}".format(
                            "U",
                            self.genn_object.generation,
                            self.genn_object.current_specimen,
                            "0"
                        ))

    def parse_game_data(self, data):
        # Data is separated by a semicolon.
        data = data.decode("utf-8").split(';')

        # Define a placeholder for the return data.
        return_data = ""

        self.log(logging.DEBUG, len(data))

        # Data is valid.
        if len(data) == 24:
            # All data in data is of type int.
            data = [float(x) for x in data]

            # Separate the data passed into the AI, and the ones used for the fitness function.
            guess = self.genn_object.specimen[self.genn_object.current_specimen].make_prediction(data[:24])  # data is the surroundings of the snake, together with the distance differentials.

            # Return the move the game makes.  (0 = left, 1 = right, 2 = up, 3 = down)
            move = "LRUD"[guess.index(max(guess))]

            # Send the move, plus the generation and individual number back.
            return_data = ";".join(map(str, [move, self.genn_object.generation, self.genn_object.current_specimen, self.genn_object.previous_generation_score / self.genn_object.population_size]))
            self.log(logging.DEBUG, f"Received this data: {repr(data)}")
            self.log(logging.DEBUG, f"Sending this data back: {return_data}")
        else:
            self.log(logging.WARN, f"Received invalid data!: {repr(data)}")

        return return_data

    # A function to handle a client disconnect, meaning the AI died.
    def handle_death(self, s, score):
        # Update the fitness of the nnetwork.
        fitness = float(score)

        # Store that in the global fitness dictionary.
        self.genn_object.specimen_fitness[self.genn_object.current_specimen] = fitness

        # Notify that the AI died.
        self.log(logging.INFO, f"AI {self.genn_object.generation}:{self.genn_object.current_specimen} died. Score: {self.current_score}.")

        # Check if we're about to breed.
        if self.genn_object.current_specimen >= self.genn_object.population_size - 1:
            # Notify Unity we're breeding.
            self.write_queue[s].put("BREED")
        else:
            self.write_queue[s].put(
                ";".join(map(str, ["U", self.genn_object.generation, self.genn_object.current_specimen, self.genn_object.previous_generation_score / self.genn_object.population_size]))
            )  # Otherwise Unity dies.

        # Let the next AI have a go.
        self.genn_object.next_specimen()


def main(s: SnekAI):
    port_no = 6969
    ai_started = False

    while not ai_started:
        try:
            s.log(logging.INFO, f"Trying to start server on port {port_no}")
            s.start_server(port=port_no)
            ai_started = True
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                port_no -= 1
                if port_no < 1000:
                    raise Exception("Cannot open nnetwork connection on all ports.")
            else:
                print(traceback.format_exc())


if __name__ == "__main__":
    # Make a SnakeAI object and try to resume where it left off training.
    s = SnekAI(file_log_level=None)
    filename = "{}.pickle".format(__file__.split(".")[0])

    try:
        # Run the main function.
        main(s)

    except Exception as e:
        s.log(logging.ERROR, "Crashed!")

        s.log(logging.INFO, f"Saving GeNN to {filename}...")
        s.genn_object.save_network(filename)

        s.log(logging.ERROR, traceback.format_exc())
        s.server_socket.close()
        s.log(logging.ERROR, "Exiting...")
        exit(1)

    except KeyboardInterrupt:
        s.log(logging.INFO, "Interrupt received.")

        s.log(logging.INFO, f"Saving GeNN to {filename}...")
        s.genn_object.save_network(filename)

        s.log(logging.INFO, "Closing server socket...")
        s.server_socket.close()

        s.log(logging.INFO, "Exiting...")
        exit(0)
