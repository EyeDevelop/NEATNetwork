from .crossover import crossover
from .random_gene_copy import random_gene_copy

breeding_functions = {
    "crossover": crossover,
    "random_gene_copy": random_gene_copy,
}
