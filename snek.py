import pickle
import queue
import select
import socket

from neat import NEAT


class SnekAI:
    def __init__(self, load_neat_file=""):
        if load_neat_file:
            with open(load_neat_file, "rb") as fp:
                self.neat_object = pickle.load(fp)
        else:
            self.neat_object = NEAT(2, [225, 400, 200, 4], 50, 20, 3)

        self.server_socket = None
        self.write_queue = {}
        self.current_score = 0

    def fitness(self, inputs: list, outputs: list):
        pass

    def start_server(self, port=6969):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        with self.server_socket:
            inputs = [self.server_socket]
            outputs = []

            self.server_socket.setblocking(False)
            self.server_socket.bind(('', port))
            self.server_socket.listen(1)

            print("Ready for connections.")

            write_queue = {}

            while inputs:
                readable, writable, exceptions = select.select(inputs, outputs, inputs)

                for s in readable:
                    if s is self.server_socket:
                        c_sock, _ = self.server_socket.accept()

                        print("\nSnek Game connected.")
                        print("\nStarting with:")
                        print("Generation:", self.neat_object.generation)
                        print("Individual:", self.neat_object.current_specimen)
                        print()

                        c_sock.setblocking(False)

                        inputs.append(c_sock)
                    else:
                        data = s.recv(1024)

                        if data:
                            if s not in write_queue.keys():
                                write_queue[s] = queue.Queue()

                            if "DEAD" in data.decode("utf-8"):
                                self.handle_death()
                                write_queue[s].put("U;0;0")
                            else:
                                response = self.parse_game_data(data)
                                write_queue[s].put(response)

                            if s not in outputs:
                                outputs.append(s)
                        else:
                            if s in outputs:
                                outputs.remove(s)

                            inputs.remove(s)
                            s.close()

                            if s in write_queue.keys():
                                del write_queue[s]

                            print("\nLost connection to Snek.")

                for s in writable:
                    if s in write_queue.keys():
                        try:
                            next_message = write_queue[s].get_nowait()
                        except queue.Empty:
                            outputs.remove(s)
                        else:
                            s.send(next_message.encode("utf-8"))

                for s in exceptions:
                    if s in outputs:
                        outputs.remove(s)

                    inputs.remove(s)
                    s.close()
                    del write_queue[s]

                    print("\nLost connection to Snek.")

    def parse_game_data(self, data):
        # Data is separated by a semicolon.
        data = data.decode("utf-8").split(';')

        # All data in data is of type int.
        data = [int(x) for x in data]

        # Separate the data passed into the AI, and the ones used for the fitness function.
        guess = self.neat_object.specimen[self.neat_object.current_specimen].make_prediction(data[:225])  # data contains the grid (15x15)
        current_score = data[225]  # data[225] is the score assigned by the game.

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
    s = SnekAI(load_neat_file="neat.pickle")

    try:
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
