import random
a = random.randint(0,10000000000)
import math
from functools import partial
import numpy as np
from gerrychain import MarkovChain, Graph
from recursive_tree_timeout import *
#from gerrychain.tree import recursive_tree_part, bipartition_tree
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

def tiled_recom(partition1, partition2, verbose = False):
    remaining_nodes = set(partition1.graph.nodes)  
    tile_assign = {}
    highest_key = len(partition1)-1
    
    while len(remaining_nodes) > 0:        
        random_node = random.sample(remaining_nodes,1)[0]
        part_choice = random.choice([partition1, partition2])
        part_choice_name = '1' if part_choice == partition1 else '2'
        assigned_dist = part_choice.assignment[random_node]       
        other_nodes_in_dist = set([n for n in remaining_nodes if part_choice.assignment[n] == assigned_dist])

        if verbose:
            print("node", random_node, "part", part_choice_name, "assigned dist", assigned_dist )
            print("len other nodes", len(other_nodes_in_dist), "len part's dist", len(part_choice.parts[assigned_dist]))
        
        if len(other_nodes_in_dist) == len(part_choice.parts[assigned_dist]):
            highest_key += 1
            for node in other_nodes_in_dist:
                tile_assign[node] = highest_key                    
            remaining_nodes = remaining_nodes.difference(other_nodes_in_dist)
            
        else: 
            part_choice = partition2 if part_choice == partition1 else partition1
            assigned_dist = part_choice.assignment[random_node] 
            part_choice_name = '1' if part_choice == partition1 else '2'
            other_nodes_in_dist = set([n for n in remaining_nodes if part_choice.assignment[n] == assigned_dist])
            
            if verbose:
                print("node", random_node, "ALT part", part_choice_name, "ALT assigned dist", assigned_dist )
                print("len other nodes", len(other_nodes_in_dist), "len part's dist", len(part_choice.parts[assigned_dist]))
           
            if len(other_nodes_in_dist) == len(part_choice.parts[assigned_dist]): 
                highest_key += 1
                for node in other_nodes_in_dist:
                    tile_assign[node] = highest_key
                remaining_nodes = remaining_nodes.difference(other_nodes_in_dist)
                  
            else:
                if verbose:
                    print("lone node!")
                tile_assign[random_node] = 1000                      
                remaining_nodes = remaining_nodes.difference({random_node})
            
                
    return Partition(partition1.graph, tile_assign, partition1.updaters)

#part_result = tiled_recom(part1, part2)
#ideal_pop = sum(part_result["population"].values())/len(part1.parts)
#print("ideal pop", ideal_pop, ideal_pop- ideal_pop*.02, ideal_pop + ideal_pop*.02)
#sub_graph = part_result.graph.subgraph(part_result.parts[1000])
#components = list(connected_components(sub_graph))
#print([len(c) for c in components])
#for c in components:
#    tot_pop = 0
#    for node in c:
#        tot_pop += graph.nodes[node]["TOTPOP"]
#    print("tot pop", tot_pop)       
#  
    
def half_split(partition, gdf, pop_col = "TOTPOP"):
    tot_pop = sum([partition.graph.nodes[v][pop_col] for v in partition.graph.nodes()])
    rand_split_assign = recursive_tree_part(partition.graph, range(2), tot_pop/2, pop_col, .01, 3)
    
    return Partition(partition.graph, rand_split_assign, partition.updaters)
        
    
    



################################  testing ########################################