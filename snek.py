import queue
import select
import socket

from neat import NEAT


class SnekAI(NEAT):
    def __init__(self):
        # Network structure is pre-defined.
        # 225 inputs for the 15x15 grid.
        # 400 for the first hidden layer.
        # 200 for the next.
        # 4 outputs, being a direction. (left, right, up, down)
        super().__init__(layer_count=2, neuron_counts=[225, 400, 200, 4], population_size=50, mutation_chance=20, mutation_severity=3)

        # Set up a server socket, on which connections can be received.
        self.server_socket = None

        # Keep a write queue to send data back to the game.
        self.write_queue = {}

        # Keep track of the game score.
        self.current_score = 0

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
                        print("Generation:", self.generation)
                        print("Individual:", self.current_specimen)
                        print()

                        c_sock.setblocking(False)

                        inputs.append(c_sock)
                    else:
                        try:
                            data = s.recv(1024)
                        except:
                            continue

                        if data:
                            print("Received message.", repr(data))
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
        guess = self.specimen[self.current_specimen].make_prediction(data[:225])  # data contains the grid (15x15)
        current_score = data[225]  # data[225] is the score assigned by the game.

        print("Current score:", current_score)

        # Update the current score variable.
        self.current_score = current_score

        # Return the move the game makes.  (0 = left, 1 = right, 2 = up, 3 = down)
        move = "LRUD"[guess.index(max(guess))]

        # Notify what move the AI decided on.
        print("AI tried:", move)

        # Send the move, plus the generation and individual number back.
        return_data = ";".join(map(str, [move, self.generation, self.current_specimen]))

        return return_data

    # A function to handle a client disconnect, meaning the AI died.
    def handle_death(self):
        # Calculate the fitness of the network.
        fitness = self.current_score

        # Store that in the global fitness dictionary.
        self.specimen_fitness[self.specimen[self.current_specimen]] = fitness

        # Let the next AI have a go.
        self.next_specimen()

        # Notify that the AI died.
        print("\nThe AI died.")
        print("Fitness:", fitness)

        # Show the next individual having a go.
        print("\nNow trying:")
        print("Generation:", self.generation)
        print("Individual:", self.current_specimen)
        print()


def main(s: SnekAI):
    s.start_server(port=6969)


if __name__ == "__main__":
    s = SnekAI()

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
        print("Exiting...")
        s.server_socket.close()

        exit(0)
