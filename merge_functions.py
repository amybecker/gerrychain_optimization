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
# merge functions take a target number of districts, k, and a partition with
# more than k districts and merge the parts until there are k parts and
# return a partition with exactly k districts
################################################################


def merge_parts_chen(partition, k, dist_func, draw_map = False):
    #until there are k parts, merge adjacent parts with smallest sum
    assert (len(partition.parts) >= k)
    while len(partition.parts) > k:
        part_rand = random.choice(range(len(partition.parts)))
        # print(partition.centroids[part_rand])
        closest_part = (part_rand + 1) % len(partition.parts)
        dist_min = math.inf
        for e in partition["cut_edges"]:
            if partition.assignment[e[0]] == part_rand:
                i = partition.assignment[e[1]]
                dist_i = dist_func(partition.centroids[part_rand], partition.centroids[i])
                if dist_i < dist_min:
                    closest_part = i
                    dist_min = dist_i
            if partition.assignment[e[1]] == part_rand:
                i = partition.assignment[e[0]]
                dist_i = dist_func(partition.centroids[part_rand], partition.centroids[i])
                if dist_i < dist_min:
                    closest_part = i
                    dist_min = dist_i
        if draw_map:
            gdf_print_map(partition, './figures/iter_merge_'+str(len(partition.parts))+'.png', gdf, unit_key)
        merge_dict = {v:part_rand if partition.assignment[v] == closest_part else partition.assignment[v] for v in partition.graph.nodes()}
        partition = Partition(partition.graph, merge_dict, partition.updaters)
        keyshift = dict(zip(list(partition.parts.keys()), range(len(partition.parts.keys()))))
        keydict = {v:keyshift[partition.assignment[v]] for v in partition.graph.nodes()}
        partition = Partition(partition.graph, keydict, partition.updaters)
    return partition


def merge_parts_smallest_sum(partition, k):
    #until there are k parts, merge adjacent parts with smallest sum
    assert (len(partition.parts) >= k)
    while len(partition.parts) > k:
        min_pair = (partition.assignment[list(partition["cut_edges"])[0][0]],partition.assignment[list(partition["cut_edges"])[0][1]])
        min_sum = math.inf
        for e in partition["cut_edges"]:
            edge_sum = partition["population"][partition.assignment[e[0]]] + partition["population"][partition.assignment[e[1]]]
            if  edge_sum < min_sum:                
                min_sum = edge_sum
                min_pair = (partition.assignment[e[0]], partition.assignment[e[1]])
        merge_dict = {v:min_pair[0] if partition.assignment[v] == min_pair[1] else partition.assignment[v] for v in partition.graph.nodes()}
        partition = Partition(partition.graph, merge_dict, partition.updaters)
    keyshift = dict(zip(list(partition.parts.keys()), range(len(partition.parts.keys()))))
    keydict = {v:keyshift[partition.assignment[v]] for v in partition.graph.nodes()}
    return Partition(partition.graph, keydict, partition.updaters)


def merge_parts_smallest_sum_alt(partition, k):
    # print('start merge-parts')
    #until there are k parts, merge adjacent parts with smallest sum
    assert (len(partition.parts) >= k)
    neighbor_districts = {}
    for e in partition["cut_edges"]:
        pair = (min(partition.assignment[e[0]], partition.assignment[e[1]]),max(partition.assignment[e[0]], partition.assignment[e[1]]))
        neighbor_districts.update({pair:partition["population"][partition.assignment[e[0]]] + partition["population"][partition.assignment[e[1]]]})
    while len(partition.parts) > k:
        min_sum = min(neighbor_districts.values())
        min_key = [key for key in neighbor_districts.keys() if neighbor_districts[key] == min_sum][0]
        merge_dict = {v:min_key[0] if partition.assignment[v] == min_key[1] else partition.assignment[v] for v in partition.graph.nodes()}
        partition = Partition(partition.graph, merge_dict, partition.updaters)
        #update neighbor_districts
        new_dict = {}
        for key in neighbor_districts.keys():
            if key == min_key:
                continue
            elif key[0] == min_key[1]:
                new_key = (min(min_key[0], key[1]),max(min_key[0], key[1]))
            elif key[1] == min_key[1]:
                new_key = (min(min_key[0], key[0]),max(min_key[0], key[0]))
            elif min_key[0] in [key[0],key[1]]:
                new_key = key
            else:
                new_key = key
            new_val = partition["population"][new_key[0]] + partition["population"][new_key[1]]
            new_dict[new_key] = new_val
        neighbor_districts = new_dict
    keyshift = dict(zip(list(partition.parts.keys()), range(len(partition.parts.keys()))))
    keydict = {v:keyshift[partition.assignment[v]] for v in partition.graph.nodes()}
    return Partition(partition.graph, keydict, partition.updaters)
