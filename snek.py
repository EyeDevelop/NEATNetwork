import os
import pickle
import queue
import select
import socket

from neat import NEAT


class SnekAI:
    def __init__(self, load_neat_file=""):
        # Check if the file exists, and if it does, load it.
        neat_loaded = False

        if load_neat_file:
            if os.path.exists(load_neat_file):
                with open(load_neat_file, "rb") as fp:
                    self.neat_object = pickle.load(fp)
                    neat_loaded = True

        # If it failed to load the file, generate a new NEAT object.
        if not neat_loaded:
            self.neat_object = NEAT(layer_count=2, neuron_counts=[24, 20, 20, 4], population_size=100, retention_rate=5, mutation_chance=5, mutation_severity=5, activation_function="tanh", breeding_function="crossover")

        # Make a central server socket.
        self.server_socket = None

        # Keep a write queue to write to clients.
        self.write_queue = {}

        # Keep track of the current score of the AI.
        self.current_score = 0

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

            print("Ready for connections.")

            while inputs:
                readable, writable, exceptions = select.select(inputs, outputs, inputs)

                for s in readable:
                    if s is self.server_socket:
                        # Accept a new client if available.
                        c_sock, _ = self.server_socket.accept()

                        print("\nSnek Game connected.")
                        print("\nStarting with:")
                        print("Generation:", self.neat_object.generation)
                        print("Individual:", self.neat_object.current_specimen)
                        print()

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
                                self.write_queue[s].put("U;0;0")
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

                            print("\nLost connection to Snek.")

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

                    print("\nLost connection to Snek.")

    def parse_game_data(self, data):
        # Data is separated by a semicolon.
        data = data.decode("utf-8").split(';')

        # All data in data is of type int.
        data = [int(x) for x in data]

        # Separate the data passed into the AI, and the ones used for the fitness function.
        guess = self.neat_object.specimen[self.neat_object.current_specimen].make_prediction(data[:24])  # data is the surroundings of the snake, together with the distance differentials.
        current_score = data[24]  # data[24] is de score.

        # Update the current score variable.
        self.current_score = current_score

        # Return the move the game makes.  (0 = left, 1 = right, 2 = up, 3 = down)
        move = "LRUD"[guess.index(max(guess))]

        # Send the move, plus the generation and individual number back.
        return_data = ";".join(map(str, [move, self.neat_object.generation, self.neat_object.current_specimen]))

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
        print("\nThe AI died.")
        print("Fitness:", fitness)

        # Show the next individual having a go.
        print("\nNow trying:")
        print("Generation:", self.neat_object.generation)
        print("Individual:", self.neat_object.current_specimen)
        print()


def main(s: SnekAI):
    s.start_server(port=6969)


if __name__ == "__main__":
    # Make a SnakeAI object and try to resume where it left off training.
    s = SnekAI()

    try:
        # Run the main function.
        main(s)

    except Exception as e:
        print("\nCrashed!")
        print(e)
        # traceback.print_exc(e)
        s.server_socket.close()
        print("Exiting...")
        exit(1)

    except KeyboardInterrupt:
        print("\nInterrupt received.")

        print("Saving NEAT to file...")
        s.neat_object.save_network()

        print("Closing server socket...")
        s.server_socket.close()

        print("Exiting...")
        exit(0)
