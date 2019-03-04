# Apply sigmoid activation.
import math


def sigmoid(inputs: list) -> float:
    try:
        value = 1 / (1 + math.e ** (- sum(inputs)))
    except OverflowError:
        if sum(inputs) > 0:
            value = 1
        else:
            value = 0

    return value
