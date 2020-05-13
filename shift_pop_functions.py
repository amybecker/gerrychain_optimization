import random
a = random.randint(0,10000000000)
import math
from functools import partial
import numpy as np
from gerrychain import MarkovChain, Graph, constraints
from recursive_tree_timeout import *
#from gerrychain.tree import recursive_tree_part, bipartition_tree
from gerrychain.constraints import (
    Validator,
    single_flip_contiguous,
    within_percent_of_ideal_population,
)
from gerrychain.partition import Partition, GeographicPartition
from gerrychain.proposals import propose_random_flip
import networkx as nx
from county_splits import county_bipartition_tree
from utility_functions import *


################################################################
# shift pop functions take an unbalanced partition and stopping conditions
# and return a balanced partition or an unbalanced partition if the stopping conditions 
# are met before a balanced partition is found
################################################################



    
def shift_pop(partition, ep, rep_max, ideal_pop, name = 'shift_orig', draw_map= False):
    counter = 0
    while max([abs(partition["population"][i]-ideal_pop) for i in dict(partition["population"]).keys()]) > ep*ideal_pop and counter < rep_max:
        best_pair = list(partition["cut_edges"])[0]
        best_score = 0
        for e in partition["cut_edges"]:
            score =  abs(partition["population"][partition.assignment[e[0]]] - partition["population"][partition.assignment[e[1]]])
            if max([abs(partition["population"][partition.assignment[e[i]]]-ideal_pop) for i in [0,1]]) > ep*ideal_pop and score > best_score:
                if partition["population"][partition.assignment[e[0]]]> partition["population"][partition.assignment[e[1]]]:
                    max_part = 0
                    min_part = 1
                else:
                    max_part = 1
                    min_part = 0
                if partition["population"][partition.assignment[e[max_part]]]- ideal_pop > ep*ideal_pop:
                    max_high = True
                else:
                    max_high = False
                if partition["population"][partition.assignment[e[min_part]]]- ideal_pop > ep*ideal_pop:
                    min_high = True
                else:
                    min_high = False
                if ideal_pop - partition["population"][partition.assignment[e[max_part]]] > ep*ideal_pop:
                    max_low = True
                else:
                    max_low = False
                if ideal_pop - partition["population"][partition.assignment[e[min_part]]] > ep*ideal_pop:
                    min_low = True
                else:
                    min_low = False
                val = partition.graph.nodes[e[max_part]]["TOTPOP"]
                # print(e, score, best_score, max_high, min_high, max_low, min_low, val)
                if (min_low and not max_low) or (max_high and not min_high):
                    if partition["population"][partition.assignment[e[max_part]]]-val > (1-ep)*ideal_pop and partition["population"][partition.assignment[e[min_part]]]+val < (1+ep)*ideal_pop:
                        subg = partition.graph.subgraph(partition.parts[partition.assignment[e[max_part]]]).copy()
                        subg.remove_node(e[max_part])
                        if nx.is_connected(subg):
                            best_pair = (e[max_part], e[min_part])
                            best_score = score
        if best_score == 0:
            break
        else:
            shift_dict = {v:partition.assignment[best_pair[1]] if v == best_pair[0] else partition.assignment[v] for v in partition.graph.nodes()}
        partition = Partition(partition.graph, shift_dict, partition.updaters)
        if draw_map:
            print_map(partition.assignment, name+str(counter))
            
        counter += 1
    # print(max([abs(partition["population"][i]-ideal_pop) for i in dict(partition["population"]).keys()]),ep*ideal_pop)
    # print('ep:',ep,'ideal pop:',ideal_pop)
    # print("pop_dict:",partition["population"])
    # print('counter:', counter, "pop dev:",pop_dev(partition))
    return partition 

def shift_pop_relaxed(partition, ep, rep_max, ideal_pop, name = 'shift_orig', draw_map= False):
    counter = 0
    while max([abs(partition["population"][i]-ideal_pop) for i in dict(partition["population"]).keys()]) > ep*ideal_pop and counter < rep_max:
        best_pair = list(partition["cut_edges"])[0]
        best_score = 0
        diff_dict = {i:partition["population"][i]-ideal_pop for i in dict(partition["population"]).keys()}
        for e in partition["cut_edges"]:
            if max([abs(partition["population"][partition.assignment[e[i]]]-ideal_pop) for i in [0,1]]) > ep*ideal_pop:
                if partition["population"][partition.assignment[e[0]]]> partition["population"][partition.assignment[e[1]]]:
                    max_part = 0
                    min_part = 1
                else:
                    max_part = 1
                    min_part = 0
                if partition["population"][partition.assignment[e[max_part]]]- ideal_pop > ep*ideal_pop:
                    max_high = True
                else:
                    max_high = False
                if ideal_pop - partition["population"][partition.assignment[e[min_part]]] > ep*ideal_pop:
                    min_low = True
                else:
                    min_low = False
                val = partition.graph.nodes[e[max_part]]["TOTPOP"]
                # print(e, score, best_score, max_high, min_high, max_low, min_low, val)
                if min_low or max_high:
                    score = max(abs(diff_dict[partition.assignment[e[max_part]]]),abs(diff_dict[partition.assignment[e[min_part]]])) - max(abs(diff_dict[partition.assignment[e[max_part]]]-val),abs(diff_dict[partition.assignment[e[min_part]]]+val))
                    if  score > best_score:
                        subg = partition.graph.subgraph(partition.parts[partition.assignment[e[max_part]]]).copy()
                        subg.remove_node(e[max_part])
                        if nx.is_connected(subg):
                            best_pair = (e[max_part], e[min_part])
                            best_score = score
        if best_score == 0:
            break
        else:
            shift_dict = {v:partition.assignment[best_pair[1]] if v == best_pair[0] else partition.assignment[v] for v in partition.graph.nodes()}
        partition = Partition(partition.graph, shift_dict, partition.updaters)
        if draw_map:
            print_map(partition.assignment, name+str(counter))
            
        counter += 1
    # print(max([abs(partition["population"][i]-ideal_pop) for i in dict(partition["population"]).keys()]),ep*ideal_pop)
    # print('ep:',ep,'ideal pop:',ideal_pop)
    # print("pop_dict:",partition["population"])
    # print('counter:', counter, "pop dev:",pop_dev(partition))
    return partition 


def shift_pop_alt(partition, ep, rep_max, ideal_pop, name = 'shift_alt', draw_map= False):
    # can improve further: 
    #     - priority queues for candidate pairs
    #     - store boundary nodes instead of edges (ideally in priority queues) so don't have to iterate over them
    #     - avoid recreating and looping over dictionaries and lists

    
    counter = 0
    switch_count = 0
    neighbor_districts = {}
    cut_edge_dict = {}
    for e in partition["cut_edges"]:
        pair = (min(partition.assignment[e[0]], partition.assignment[e[1]]),max(partition.assignment[e[0]], partition.assignment[e[1]]))
        neighbor_districts.update({pair:partition["population"][pair[0]] - partition["population"][pair[1]]})
        cut_edge_dict.setdefault(pair, []).append([min(e[0],e[1]), max(e[0],e[1])])

    exclude_set = set()
    while max([abs(partition["population"][i]-ideal_pop) for i in dict(partition["population"]).keys()]) > ep*ideal_pop and counter < rep_max:
        if len(exclude_set) >= len(neighbor_districts):
            # print('broke')
            # print(exclude_set)
            # print(neighbor_districts)
            break
        #do we want to avoid moving pop from underpopulated districts?
        max_diff = max([abs(neighbor_districts[key]) for key in set(neighbor_districts.keys()).difference(exclude_set)])
        max_diff_key = [key for key in set(neighbor_districts.keys()).difference(exclude_set) if abs(neighbor_districts[key]) == max_diff][0]
        if max_diff_key[1] >= max_diff_key[0]:
            max_part = max_diff_key[1]
            min_part = max_diff_key[0]
        else:
            max_part = max_diff_key[0]
            min_part = max_diff_key[1]
        
        consider_nodes = {u: partition.graph.nodes[u]["TOTPOP"] for [u,v] in cut_edge_dict[max_diff_key] if partition.assignment[u]==max_part}
        consider_nodes.update({v: partition.graph.nodes[v]["TOTPOP"] for [u,v] in cut_edge_dict[max_diff_key] if partition.assignment[v]==max_part})
        shifted_part = False
        while len(consider_nodes.keys()) > 0 and shifted_part == False:
            max_pop_val = max([consider_nodes[key] for key in consider_nodes.keys()])
            max_pop_node = [key for key in consider_nodes.keys() if consider_nodes[key] == max_pop_val][0]
            if partition["population"][max_part]-max_pop_val > (1-ep)*ideal_pop and partition["population"][min_part]+max_pop_val < (1+ep)*ideal_pop:
                subg = partition.graph.subgraph(partition.parts[max_part]).copy()
                subg.remove_node(max_pop_node)
                if nx.is_connected(subg):
                    #can change this to an update...
                    shift_dict = {v:min_part if v == max_pop_node else partition.assignment[v] for v in partition.graph.nodes()}
                    partition = Partition(partition.graph, shift_dict, partition.updaters)
                    if draw_map:
                        print_map(partition.assignment, name+str(counter))
                    shifted_part = True
                    switch_count += 1
                    exclude_set = set()

                    neighbor_districts[max_diff_key] = partition["population"][max_diff_key[0]] - partition["population"][max_diff_key[1]]

                    for neighbor in partition.graph.neighbors(max_pop_node):
                        assert(len(set(cut_edge_dict.keys()).difference(set(neighbor_districts.keys())))==0)
                        assert(len(set(neighbor_districts.keys()).difference(set(cut_edge_dict.keys())))==0)
                        if partition.assignment[neighbor] == min_part:
                            cut_edge_dict[max_diff_key] = [[u,v] for [u,v] in cut_edge_dict[max_diff_key] if [u,v] not in [[neighbor, max_pop_node],[max_pop_node, neighbor]]]
                        elif partition.assignment[neighbor] == max_part:
                            cut_edge_dict[max_diff_key].append([min(neighbor, max_pop_node),max(max_pop_node, neighbor)])
                        else:
                            max_neighbor_key = (min(max_part, partition.assignment[neighbor]),max(max_part, partition.assignment[neighbor]))                            
                            cut_edge_dict[max_neighbor_key] = [[u,v] for [u,v] in cut_edge_dict[max_neighbor_key] if [u,v] not in [[neighbor, max_pop_node],[max_pop_node, neighbor]]]
                            if len(cut_edge_dict[max_neighbor_key]) == 0:
                                neighbor_districts = {key:neighbor_districts[key] for key in neighbor_districts.keys() if key != max_neighbor_key}
                                cut_edge_dict = {key:cut_edge_dict[key] for key in cut_edge_dict.keys() if key != max_neighbor_key}
                            else:
                                neighbor_districts[max_neighbor_key] = partition["population"][max_neighbor_key[0]] - partition["population"][max_neighbor_key[1]]

                            min_neighbor_key = (min(min_part, partition.assignment[neighbor]),max(min_part, partition.assignment[neighbor]))
                            if min_neighbor_key not in neighbor_districts.keys():
                                assert(min_neighbor_key not in cut_edge_dict.keys())
                                cut_edge_dict[min_neighbor_key] = [[min(neighbor, max_pop_node),max(max_pop_node, neighbor)]]
                            else:
                                cut_edge_dict[min_neighbor_key].append([min(neighbor, max_pop_node),max(max_pop_node, neighbor)])
                            neighbor_districts[min_neighbor_key] = partition["population"][min_neighbor_key[0]] - partition["population"][min_neighbor_key[1]]

            if not shifted_part:
                consider_nodes = {key:consider_nodes[key] for key in consider_nodes.keys() if key != max_pop_node}
        
        if not shifted_part:
            exclude_set.add(max_diff_key)
        
        counter += 1
    
    #will return unbalanced partition if failed to balance
    # print(counter, switch_count)
    return partition 


def find_pop_granularity(graph, half_pop):
    min_pop = min([graph.nodes[v]["TOTPOP"] for v in graph.nodes()])
    min_pop_node = [v for v in graph.nodes() if graph.nodes[v]["TOTPOP"]==min_pop][0]
    frontier = set(graph.neighbors(min_pop_node))
    growing_dist = [min_pop_node]
    counter = 0
    while sum([graph.nodes[v]["TOTPOP"]for v in growing_dist]) < half_pop:
        # print(len(graph.nodes()), counter)
        min_neighbor_pop = min([graph.nodes[v]["TOTPOP"] for v in frontier])
        min_neighbor = [v for v in frontier if graph.nodes[v]["TOTPOP"]==min_neighbor_pop][0]
        growing_dist.append(min_neighbor)
        frontier = frontier.union(set(graph.neighbors(min_neighbor)))
        frontier = frontier.difference(set(growing_dist))
        counter += 1
    return sum([graph.nodes[v]["TOTPOP"]for v in growing_dist])/half_pop - 1




def shift_pop_recom(partition, ep, rep_max, ideal_pop, assign_col = 'COUNTYFP10', name = 'shift_alt', draw_map= False, pop_col= "TOTPOP", bipartion_ep = 0.02, node_repeats=3, method=county_bipartition_tree):
    # iteratively perform recom steps to rebalance two neighboring districts 
    # with greatest population difference until either the plan is balances
    # or maximum allowed number of steps is reached
    
    # find population difference between all pairs of neighboring districts
    neighbor_districts = {}
    for e in partition["cut_edges"]:
        pair = (min(partition.assignment[e[0]], partition.assignment[e[1]]),max(partition.assignment[e[0]], partition.assignment[e[1]]))
        neighbor_districts.update({pair:partition["population"][pair[0]] - partition["population"][pair[1]]})

    counter = 0

    # while populations remain unbalanced and we haven't reached maximum number of allowed repititions
    remove_set = set()
    while max([abs(partition["population"][i]-ideal_pop) for i in dict(partition["population"]).keys()]) > ep*ideal_pop and counter < rep_max and len(remove_set)< len(neighbor_districts.keys()):
        # print(counter, len(remove_set), len(neighbor_districts))

        #find neighboring districts with largest population difference
        max_diff = max([abs(neighbor_districts[key]) for key in set(neighbor_districts.keys()).difference(remove_set)])
        max_diff_key = [key for key in set(neighbor_districts.keys()).difference(remove_set) if abs(neighbor_districts[key]) == max_diff][0]
        parts_to_merge = (max_diff_key[0], max_diff_key[1])

        subgraph = partition.graph.subgraph(partition.parts[parts_to_merge[0]] | partition.parts[parts_to_merge[1]])
        avg_pop = (partition.population[parts_to_merge[0]]+partition.population[parts_to_merge[1]])/2

        #perform recom to rebalance populations of these neighboring districts
        granularity_ep = find_pop_granularity(subgraph, avg_pop)
        adjusted_bipartition_ep = max(bipartion_ep, granularity_ep)
        flips = recursive_tree_part(subgraph,parts_to_merge,pop_col=pop_col,pop_target=avg_pop,epsilon=adjusted_bipartition_ep,node_repeats=node_repeats,method=(partial(county_bipartition_tree, county_col = assign_col)))
        partition = partition.flip(flips)
        if abs(partition.population[parts_to_merge[0]]-partition.population[parts_to_merge[1]]) == max_diff:
            remove_set.add(max_diff_key)
        else:
            remove_set = set()

        if draw_map:
            print_map(partition.assignment, name+str(counter))
        
        # update neighboring district populations
        neighbor_districts = {(min(partition.assignment[e[0]], partition.assignment[e[1]]),max(partition.assignment[e[0]], partition.assignment[e[1]])):partition["population"][min(partition.assignment[e[0]], partition.assignment[e[1]])] - partition["population"][max(partition.assignment[e[0]], partition.assignment[e[1]])] for e in partition["cut_edges"]}
        counter+= 1

    return partition 




def shift_chen(partition, ep, rep_max, ideal_pop, dist_func, draw_map= False):
    counter = 0
    past_10 = []
    while max([abs(partition["population"][i]-ideal_pop) for i in dict(partition["population"]).keys()]) > ep*ideal_pop and counter < rep_max:
        if len(past_10) == 10 and len(set(past_10)) <=2:
            # print('****** past 10 fail', past_10,counter)
            # print(max([abs(partition["population"][i]-ideal_pop) for i in dict(partition["population"]).keys()]),ep*ideal_pop)
            # print('ep:',ep,'ideal pop:',ideal_pop)
            # print("pop_dict:",partition["population"])
            # print('counter:', counter, 'past_10:', past_10, "pop dev:",pop_dev(partition))
            return partition
        max_diff_pair = (partition.assignment[list(partition["cut_edges"])[0][0]], partition.assignment[list(partition["cut_edges"])[0][1]])
        max_diff_score = 0
        max_diff_edges = []
        for e in partition["cut_edges"]:
            score =  abs(partition["population"][partition.assignment[e[0]]] - partition["population"][partition.assignment[e[1]]])
            if score > max_diff_score:
                max_diff_edges = [e]
                max_diff_score = score
                max_diff_pair = (partition.assignment[e[0]], partition.assignment[e[1]])
            elif partition.assignment[e[0]] in max_diff_pair and partition.assignment[e[1]] in max_diff_pair:
                max_diff_edges.append(e)
        if partition["population"][max_diff_pair[0]] >= partition["population"][max_diff_pair[1]]:
            unit_m = max_diff_pair[0]
            unit_l = max_diff_pair[1]
        else:
            unit_m = max_diff_pair[1]
            unit_l = max_diff_pair[0]

        moveable_units = {}
        for e in max_diff_edges:
            if partition.assignment[e[0]] == unit_m:
                assert(partition.assignment[e[1]] == unit_l)
                edge_unit_max = e[0]
                edge_unit_min = e[1]
            else:
                assert(partition.assignment[e[0]] == unit_l)
                assert(partition.assignment[e[1]] == unit_m)
                edge_unit_max = e[1]
                edge_unit_min = e[0]
            #check contiguity
            subg = partition.graph.subgraph(partition.parts[partition.assignment[edge_unit_max]]).copy()
            subg.remove_node(edge_unit_max)
            if nx.is_connected(subg):
                unit_centroid = (float(partition.graph.nodes[edge_unit_max]['x']), float(partition.graph.nodes[edge_unit_max]['y']))
                moveable_units[(edge_unit_max, edge_unit_min)] = dist_func(partition.centroids[unit_m], unit_centroid) - dist_func(partition.centroids[unit_l], unit_centroid)
        
        if len(moveable_units) == 0:
            # print(past_10, counter)
            # print(max([abs(partition["population"][i]-ideal_pop) for i in dict(partition["population"]).keys()]),ep*ideal_pop)
            # print('ep:',ep,'ideal pop:',ideal_pop)
            # print("pop_dict:",partition["population"])
            # print('counter:', counter, 'past_10:', past_10, "pop dev:",pop_dev(partition))
            return partition
        max_dp = max(moveable_units.values())
        move_unit = [i for i in moveable_units.keys() if moveable_units[i] == max_dp][0]
        
        if len(past_10) >= 10:
            past_10 = past_10[-9:]
        past_10.append((move_unit[0], unit_m))
        shift_dict = {v:unit_l if v == move_unit[0] else partition.assignment[v] for v in partition.graph.nodes()}
        partition = Partition(partition.graph, shift_dict, partition.updaters)
        if draw_map:
            gdf_print_map(partition, './book_figs/iter_merge_shift'+str(counter)+'.png', gdf, unit_key)
            
        counter += 1
    # print(past_10, counter)
    # print(max([abs(partition["population"][i]-ideal_pop) for i in dict(partition["population"]).keys()]),ep*ideal_pop)
    # print('ep:',ep,'ideal pop:',ideal_pop)
    # print("pop_dict:",partition["population"])
    # print('counter:', counter, 'past_10:', past_10, "pop dev:",pop_dev(partition))
    return partition

def shift_flip(partition, ep, ideal_pop, max_steps = 150000, chain_bound = .02):
    
    #TODO: Amy, where do we think this function should be stored? Its
    #needed in the Chain I run within this shift function
    def pop_accept(partition):
        if not partition.parent:
            return True   
        proposal_dev_score = max_pop_dev(partition, ideal_pop)
        parent_dev_score = max_pop_dev(partition.parent, ideal_pop)
        if proposal_dev_score < parent_dev_score:
            return True
        else:
            draw = random.random()
            return draw < chain_bound
        
    # print("ideal pop", ideal_pop)
    
    pop_tol_initial = max_pop_dev(partition, ideal_pop)
    # print("pop tol initial", pop_tol_initial)
    #if error again from initial state, just make constraint value
    chain = MarkovChain(
    proposal = propose_random_flip,
    constraints=[
        constraints.within_percent_of_ideal_population(partition, pop_tol_initial+.05),
        single_flip_contiguous 
    ],
    accept = pop_accept,
    initial_state = partition,
    total_steps = max_steps
    )
    
    step_Num = 0
    for step in chain:
        partition = step
     #   print("step num", step_Num, max_pop_dev(partition, ideal_pop))
        if max_pop_dev(partition, ideal_pop) <= ep:
            break 
        step_Num += 1
    
    return Partition(partition.graph, partition.assignment, partition.updaters) 



################################  testing ########################################
