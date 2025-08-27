#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

from typing import Collection
import numpy as np

from .node import Node

from .random import gxgp_random
from .utils import arity


__all__ = ['DagGP']


class DagGP:
    def __init__(self, operators: Collection, variables: int | Collection, constants: int | Collection):
        self._operators = list(operators)
        if isinstance(variables, int):
            self._variables = [Node(DagGP.default_variable(i)) for i in range(variables)]
        else:
            self._variables = [Node(t) for t in variables]
        if isinstance(constants, int):
            self._constants = [Node(gxgp_random.randint(0, constants)) for i in range(constants)]
        else:
            self._constants = [Node(t) for t in constants]

    def create_individual(self, n_nodes=7):
        pool = self._variables * (1 + len(self._constants) // len(self._variables)) + self._constants
        individual = None
        while individual is None or len(individual) < n_nodes:
            op = gxgp_random.choice(self._operators)
            params = gxgp_random.choices(pool, k=arity(op))
            individual = Node(op, params)
            pool.append(individual)
        return individual

    @staticmethod
    def default_variable(i: int) -> str:
        return f'x{i}'

    @staticmethod
    def evaluate(individual: Node, X, variable_names=None):
        if variable_names:
            names = variable_names
        else:
            names = [DagGP.default_variable(i) for i in range(X.shape[0])]
        kwargs = {name: X[i] for i, name in enumerate(names)}
        y_pred = individual(**kwargs)
        return np.asarray(y_pred)


    @staticmethod
    def plot_evaluate(individual: Node, X, variable_names=None):
        import matplotlib.pyplot as plt

        y_pred = DagGP.evaluate(individual, X, variable_names)
        plt.figure()
        plt.title(individual.long_name)
        plt.scatter(X[0], y_pred)

        return y_pred

    @staticmethod
    def mse(individual: Node, X, y, variable_names=None):
        y_pred = DagGP.evaluate(individual, X, variable_names)
        if not np.all(np.isfinite(y_pred)):
            return float('inf')
        return np.mean(np.square(y - y_pred))