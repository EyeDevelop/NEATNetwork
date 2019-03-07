from .crossover_half import crossover_half
from .crossover import crossover
from .random_gene_copy import random_gene_copy

breeding_functions = {
    "crossover": crossover,
    "crossover_half": crossover_half,
    "random_gene_copy": random_gene_copy,
}
