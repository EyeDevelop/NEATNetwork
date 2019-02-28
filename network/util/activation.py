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
