from .binary import binary_step
from .relu import relu
from .sigmoid import sigmoid
from .tanh import tanh

activation_functions = {
    "binary": binary_step,
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
}
