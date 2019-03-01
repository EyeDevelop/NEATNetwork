import math


# Apply sigmoid activation.
def sigmoid(inputs: list) -> float:
    try:
        value = 1 / (1 + math.e ** (- sum(inputs)))
    except OverflowError:
        if sum(inputs) > 0:
            value = 1
        else:
            value = 0

    return value


# Apply TanH activation.
def tanh(inputs: list) -> float:
    try:
        value = (math.e ** sum(inputs) - math.e ** -sum(inputs)) / (math.e ** sum(inputs) + math.e ** -sum(inputs))
    except OverflowError:
        if sum(inputs) > 0:
            value = 1
        else:
            value = -1

    return value


# Apply binary activation.
def binary_step(inputs: list) -> float:
    return 1 if sum(inputs) > 0 else 0


# Apply ReLU activation.
def relu(inputs: list) -> float:
    return 0 if sum(inputs) < 0 else sum(inputs)
