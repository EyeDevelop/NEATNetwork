# Apply ReLU activation.
def relu(inputs: list) -> float:
    return 0 if sum(inputs) < 0 else sum(inputs)
