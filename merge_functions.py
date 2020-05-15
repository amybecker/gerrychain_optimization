import random
a = random.randint(0,10000000000)
import math
from functools import partial
import numpy as np
from gerrychain import MarkovChain, Graph, constraints
from recursive_tree_timeout import *
#from gerrychain.tree import recursive_tree_part, bipartition_tree
from gerrychain.partition import Partition
from dividing_functions import half_split
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

def merge_small_neighbor(partition,k):
    #finds smallest district in map and merges with its smallest neighbor (by population)
    #repeats until k dists reached
    merged_assign = dict(partition.assignment)
    cut_edge_list = list(partition["cut_edges"])
    while len(partition) > k:  
        smallest_pop_dist = min(partition["population"], key= partition["population"].get)
        dist_edges = [e for e in cut_edge_list for z in [0,1] if partition.assignment[e[z]] == smallest_pop_dist]
        neighbor_dists= list(np.unique([partition.assignment[e[z]] for z in [0,1] for e in dist_edges] ))
        neighbor_dists.remove(smallest_pop_dist)
        neighbor_pops = {q: partition["population"][q] for q in neighbor_dists}
        neighbor_smallest_pop = min(neighbor_pops, key=neighbor_pops.get)  

        for n in partition.parts[smallest_pop_dist]:
            merged_assign[n] = neighbor_smallest_pop
        
        partition = Partition(partition.graph, merged_assign, partition.updaters)
    
    return Partition(partition.graph, merged_assign, partition.updaters)

def seam_split_merge(partition1, partition2, k, ep, gdf, pop_col = 'TOTPOP'):
    #merged assignment (put 2 seed maps together along the random split line)
    subgraph_connect = False
    while not subgraph_connect:
        split_part = half_split(partition1, gdf)
        
        merge_assign = {n: partition1.assignment[n] if split_part.assignment[n] == 0 else partition2.assignment[n] for n in split_part.assignment.keys() }
        merge_part = Partition(split_part.graph, merge_assign, split_part.updaters)
        
        num_extra_parts = len(merge_part) - len(partition1)
        
        #districts cut/changed in map merging (i.e. districts eligible for merging/rebalancing)
        dists1 = [k for k in partition1.parts.keys() if k in merge_part.parts.keys()]
        dists2 = [k for k in partition2.parts.keys() if k in merge_part.parts.keys()]
        changed_dists1 = [k for k in dists1 if merge_part.parts[k] != partition1.parts[k]]
        changed_dists2 = [k for k in dists2 if merge_part.parts[k] != partition2.parts[k]]
        changed_dists = changed_dists1 + changed_dists2
        
        num_changed_dists = len(changed_dists)
        dists_to_make = num_changed_dists - num_extra_parts
        
        #lump together all nodes from districts that have been cut
        changed_nodes = []
        for l in changed_dists:
            changed_nodes = changed_nodes + list(merge_part.parts[l])
        
        change_part_assign = {n:1 if n in changed_nodes else 2 for n in split_part.assignment.keys()}
        seam_part = Partition(partition1.graph, change_part_assign, partition1.updaters)
                
         #check if "seam" subgraph is connected, if not, resplit the state and start again
        subgraph = split_part.graph.subgraph(changed_nodes)
        subgraph_connect = nx.is_connected(subgraph)
        
        
        #want graph of just nodes in changed/cut districts
    subgraph_pop = sum([subgraph.nodes[l][pop_col] for l in changed_nodes])
    ideal_subgraph_pop = subgraph_pop/dists_to_make

    #gets stuck here
    
    new_dists_assign0 = recursive_tree_part(subgraph, range(dists_to_make), ideal_subgraph_pop, pop_col, ep, node_repeats = 5) #prob not here, if bad seam, num node repeats wont help
    new_dists_assign = {n: (new_dists_assign0[n] + 2*k) for n in new_dists_assign0.keys()}
 
    final_assign = {d: merge_assign[d] if d not in changed_nodes else new_dists_assign[d] for d in split_part.assignment.keys()}  
    
    return Partition(partition1.graph, final_assign, partition1.updaters), split_part, merge_part, seam_part
               
          
