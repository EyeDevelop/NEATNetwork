import random


NUMBERS_GENERATED = 0


def random_check():
    random.seed()


def randint(a, b):
    random_check()
    return random.randint(a, b)


def random_number():
    random_check()
    return random.random()


def choice(seq):
    random_check()
    return random.choice(seq)
