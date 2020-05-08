import random
a = random.randint(0,10000000000)
import math
from functools import partial
import numpy as np
from gerrychain import MarkovChain, Graph
from gerrychain.tree import recursive_tree_part, bipartition_tree
from gerrychain.partition import Partition
import networkx as nx
from networkx import is_connected, connected_components


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

def tiled(partition1, partition2):
    remaining_nodes = set(partition1.graph.nodes)  
    tile_assign = {}
    highest_key = len(partition1)-1
    
    while len(remaining_nodes) > 0:        
        random_node = random.sample(remaining_nodes,1)[0]
        part_choice = random.choice([partition1, partition2])
        assigned_dist = part_choice.assignment[random_node]
        other_nodes_in_dist = set([n for n in remaining_nodes if part_choice.assignment[n] == assigned_dist])
        
        sub_graph = partition1.graph.subgraph(other_nodes_in_dist)
        components = list(connected_components(sub_graph))
        for component in components:
            highest_key += 1
            for node in component:
                tile_assign[node] = highest_key
  
        remaining_nodes -= other_nodes_in_dist
    return Partition(partition1.graph, tile_assign, partition1.updaters)

def half_split(partition, gdf, pop_col="TOTPOP"):
    tot_pop = gdf[pop_col].sum()
    rand_split_assign = recursive_tree_part(partition.graph, range(2), tot_pop/2, pop_col, .01, node_repeats = 3)
    
    return Partition(partition.graph, rand_split_assign, partition.my_updaters)
        
    
    



################################  testing ########################################