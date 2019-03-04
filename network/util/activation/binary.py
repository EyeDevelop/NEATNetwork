# Apply binary activation.
def binary_step(inputs: list) -> float:
    return 1 if sum(inputs) > 0 else 0
