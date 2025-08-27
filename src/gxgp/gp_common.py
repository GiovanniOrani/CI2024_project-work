#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

from copy import deepcopy

from .node import Node, NodeType

from .utils import arity
from .random import gxgp_random
import numpy as np


def xover_swap_subtree(tree1: Node, tree2: Node) -> Node:
    offspring = deepcopy(tree1)
    internal_nodes = [n for n in offspring.subtree if not n.is_leaf]
    if not internal_nodes:
        return offspring 
    node = gxgp_random.choice(internal_nodes)
    successors = node.successors
    i = gxgp_random.randrange(len(successors))
    successors[i] = deepcopy(gxgp_random.choice(list(tree2.subtree)))
    node.successors = successors
    return offspring


def random_mutation(tree: Node, num_variables: int, operators, probability: float = 0.5) -> Node:
    mask = np.random.choice([True, False], len(tree), p=[probability, 1 - probability])
    while not np.any(mask):
        mask = np.random.choice([True, False], len(tree), p=[probability, 1 - probability])

    def traverse_tree(node, mask, idx=0):
        if idx >= len(mask):
            return node, idx

        if mask[idx]:
            if node.type == NodeType.FUNCTION:
                while True:
                    op = gxgp_random.choice(operators)
                    if node.arity != arity(op):
                        continue
                    n = Node(op, node.successors)
                    if n.short_name != node.short_name:
                        node = n
                        break

            elif node.type == NodeType.CONSTANT:
                if gxgp_random.random() < 0.8:
                    node = Node(gxgp_random.uniform(-5, 5))
                else:
                    j = gxgp_random.randint(0, num_variables - 1) if num_variables > 1 else 0
                    node = Node(f'x{j}')

            elif node.type == NodeType.VARIABLE:
                if gxgp_random.random() < 0.8 and num_variables > 1:
                    i = int(node.short_name[1:])
                    while True:
                        j = gxgp_random.randint(0, num_variables - 1)
                        if j != i:
                            node = Node(f'x{j}')
                            break
                else:
                    node = Node(gxgp_random.uniform(-5, 5))

            else:
                raise ValueError(f'Unknown node type: {type(node)}')

        succ = node.successors
        for i in range(len(succ)):
            succ[i], idx = traverse_tree(node.successors[i], mask, idx + 1)

        node.successors = succ
        return node, idx

    offspring = deepcopy(tree)
    mutated_tree, _ = traverse_tree(offspring, mask)
    return mutated_tree

def subtree_mutation(tree: Node, num_variables: int, operators, idx=None) -> Node:
    offspring = deepcopy(tree)
    
    if idx is None:
        idx = gxgp_random.randint(0, len(offspring) - 1)

    def random_leaf():
        if gxgp_random.random() < 0.5:
            return Node(gxgp_random.uniform(-5, 5))
        else:
            j = gxgp_random.randint(0, num_variables - 1)
            return Node(f'x{j}')

    def wrap_random_leaf(child):
        unary_ops = [op for op in operators if arity(op) == 1]
        if unary_ops and gxgp_random.random() < 0.3:
            u = gxgp_random.choice(unary_ops)
            return Node(u, [child])
        return child

    def traverse_tree(node, idx, current_idx=0):
        if current_idx == idx:
            if node.is_leaf:
                op = gxgp_random.choice(operators)
                successors = []
                for _ in range(arity(op)):
                    successors.append(wrap_random_leaf(random_leaf()))
                return Node(op, successors), current_idx + 1
            else:
                binary_ops = [op for op in operators if arity(op) == 2]
                
                choice_pool = binary_ops
                if not choice_pool:
                    return node, current_idx + 1
                op = gxgp_random.choice(choice_pool)
                if gxgp_random.random() < 0.5:
                    left = wrap_random_leaf(random_leaf())
                    right = node
                else:
                    left = wrap_random_leaf(random_leaf())
                    right = wrap_random_leaf(random_leaf())
                successors = [left, right]
                return Node(op, successors), current_idx + 1
        
        succ = node.successors
        for i in range(len(node.successors)):
            succ[i], current_idx = traverse_tree(node.successors[i], idx, current_idx + 1)

        node.successors = succ

        return node, current_idx
    
    mutated_tree, _ = traverse_tree(offspring, idx)
    return mutated_tree

def shrink_mutation(tree: Node, num_variables: int, target_fraction: float = 0.25) -> Node:
    
    mutant = deepcopy(tree)
    n_nodes = len(mutant)
    target_size = max(1, int(n_nodes * target_fraction))

    def random_leaf():
        if gxgp_random.random() < 0.5:
            return Node(gxgp_random.uniform(-5, 5))
        else:
            j = gxgp_random.randint(0, num_variables - 1)
            return Node(f'x{j}')

    def traverse_tree(node, target_idx, current_idx=0):
        if current_idx == target_idx:
            return random_leaf(), current_idx + 1
        succ = node.successors
        for i in range(len(node.successors)):
            succ[i], current_idx = traverse_tree(node.successors[i], target_idx, current_idx + 1)
        node.successors = succ
        return node, current_idx

    idx = gxgp_random.randint(0, max(1, int(n_nodes * 0.25)))
    shrunk_tree, _ = traverse_tree(mutant, idx)

    attempts = 0
    while len(shrunk_tree) > target_size and attempts < 10:
        target_idx = gxgp_random.randint(0, len(shrunk_tree) - 1)
        shrunk_tree, _ = traverse_tree(shrunk_tree, target_idx)
        attempts += 1

    return shrunk_tree

