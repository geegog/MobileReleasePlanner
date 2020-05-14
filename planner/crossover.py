import random


def ordered(keys, first_solution, second_solution):
    """
    Performs crossover operation on chromosomes. The crossover operator takes two parents,
    randomly selects items in one parent and fixes their place in the second parent

    :param keys: Feature unique key list
    :param first_solution: A solution
    :param second_solution: A solution
    :return: Performs crossover on first and second solutions at crossover rate cr.
    """
    size = len(keys)
    [fp1, fp2] = random.sample(range(0, size), 2)

    key1 = first_solution[fp1]
    key2 = first_solution[fp2]

    offspring = []
    for index, key in enumerate(second_solution):
        if index == fp1:
            offspring.append(key1)
            continue
        if index == fp2:
            offspring.append(key2)
            continue
        else:
            offspring.append(first_solution[index])

    return offspring


def partially_matched(parent1, parent2):
    """Return a new chromosome created via partially matched crossover (PMX).

    This is suitable for permutation encoded GAs. Partially matched crossover
    respects the absolute position of alleles.

    Args:
        parent1 (List): A parent chromosome.
        parent2 (List): A parent chromosome.

    Returns:
        List[List]: Two new chromosomes descended from the given parents.
    """
    third = len(parent1) // 3
    l1 = int(random.triangular(1, third, third * 2))
    l2 = int(random.triangular(third, third * 2, len(parent1) - 1))

    if l2 < l1:
        l1, l2 = l2, l1

    def pmx(parent1, parent2):
        matching = parent2[l1:l2]
        displaced = parent1[l1:l2]
        child = parent1[0:l1] + matching + parent1[l2:]

        tofind = [item for item in displaced if item not in matching]
        tomatch = [item for item in matching if item not in displaced]

        for k, v in enumerate(tofind):
            subj = tomatch[k]
            locus = parent1.index(subj)
            child[locus] = v

        return child

    return [pmx(parent1, parent2), pmx(parent2, parent1)]


def edge_recombination(parent1, parent2):
    """Return a new chromosome created using an edge recombination operation.

    This is suitable for permutation encoded GAs.

    Args:
        parent1 (List): A parent chromosome.
        parent2 (List): A parent chromosome.

    Returns:
        List[List]: A new chromosome (element 0) descended from the given
                    parents.
    """
    return [recombine(parent1, parent2)]


def recombine(parent1, parent2):
    """Return a new chromosome based on two parents via edge recombination.

    Args:
        parent1 (List): A parent chromosome.
        parent2 (List): A parent chromosome.

    Returns:
        List: A new chromosome descended from the parents.
    """

    # Build a child chromosome
    child = []
    neighbors = adjacency_matrix(parent1, parent2)
    node = parent1[0]
    unused = list(parent1)

    while len(child) < len(parent1):
        # Add a node to the child
        child.append(node)
        unused.remove(node)

        # Remove the node from neighbor lists
        for s in list(neighbors.values()):
            if node in s:
                s.remove(node)

        if len(neighbors[node]):
            node = fewest_neighbors(node, neighbors)

        elif len(unused) > 0:
            # Or pick a node at random if the selected node has no neighbors
            node = random.choice(unused)

    return child


def adjacency_matrix(parent1, parent2):
    """Return the union of parent chromosomes adjacency matrices."""
    neighbors = {}
    end = len(parent1) - 1

    for parent in [parent1, parent2]:
        for k, v in enumerate(parent):
            if v not in neighbors:
                neighbors[v] = set()

            if k > 0:
                left = k - 1
            else:
                left = end

            if k < end:
                right = k
            else:
                right = 0

            neighbors[v].add(parent[left])
            neighbors[v].add(parent[right])

    return neighbors


def fewest_neighbors(node, neighbors):
    """Return the neighbor of this node with the fewest neighbors."""
    edges = [(n, len(neighbors[n])) for n in neighbors[node]]
    edges.sort(key=lambda n: n[1])
    return edges[0][0]
