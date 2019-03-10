# Apply TanH activation.
import math


def tanh(inputs: list) -> float:
    try:
        value = (math.e ** sum(inputs) - math.e ** -sum(inputs)) / (math.e ** sum(inputs) + math.e ** -sum(inputs))
    except OverflowError:
        if sum(inputs) > 0:
            value = 1
        else:
            value = -1

    return value
