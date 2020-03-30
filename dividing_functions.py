import random
a = random.randint(0,10000000000)
import math
from functools import partial
import numpy as np
from gerrychain import MarkovChain, Graph
from gerrychain.tree import recursive_tree_part, bipartition_tree
from gerrychain.partition import Partition
import networkx as nx


################################################################
# dividing functions take two parent partitions and form resulting child
# partition(s), dividing the plan based on the parent boundaries
################################################################



def half_half(partitionA, partitionB, half_pop, pop_col="TOTPOP"):
    # print('start half-half')
    half_assign = recursive_tree_part(partitionA.graph, range(2), half_pop, pop_col, .02, 3)
    half_part = Partition(partitionA.graph, half_assign, partitionA.updaters)
    V0 = [v for v in partitionA.graph.nodes() if half_assign[v] == 0]
    V1 = [v for v in partitionA.graph.nodes() if half_assign[v] == 1]
    A_split = common_refinement(partitionA, half_part)
    B_split = common_refinement(partitionB, half_part)

    assignmentA0 = {v:A_split.assignment[v] for v in V0}
    assignmentA1 = {v:A_split.assignment[v] for v in V1}
    assignmentB0 = {v:B_split.assignment[v] for v in V0}
    assignmentB1 = {v:B_split.assignment[v] for v in V1}

    mapA0 = {i[0]:i[1] for i in zip(list(set(assignmentA0.values())),range(len(assignmentA0)))}
    mapB1 = {i[0]:i[1]+len(mapA0) for i in zip(list(set(assignmentB1.values())),range(len(assignmentB1)))}
    mapA1 = {i[0]:i[1] for i in zip(list(set(assignmentA1.values())),range(len(assignmentA1)))}
    mapB0 = {i[0]:i[1]+len(mapA1) for i in zip(list(set(assignmentB0.values())),range(len(assignmentB0)))}

    assignmentA0B1 = {}
    assignmentA1B0 = {}
    for v in V0:
        assignmentA0B1[v] = mapA0[assignmentA0[v]]
        assignmentA1B0[v] = mapB0[assignmentB0[v]]
    for v in V1:
        assignmentA0B1[v] = mapB1[assignmentB1[v]]
        assignmentA1B0[v] = mapA1[assignmentA1[v]]
    return Partition(partitionA.graph, assignmentA0B1, partitionA.updaters), Partition(partitionA.graph, assignmentA1B0, partitionA.updaters)


def common_refinement(partition1, partition2):
    graph_pared = partition1.graph.copy()
    graph_pared.remove_edges_from(partition1["cut_edges"].union(partition2["cut_edges"]))
    refine_dict = {}
    counter = 0
    for i in nx.connected_components(graph_pared):
        for v in list(i):
            refine_dict[v] = counter
        counter += 1
    return Partition(partition1.graph, refine_dict, partition1.updaters)
    



################################  testing ########################################