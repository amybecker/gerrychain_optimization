import random
a = random.randint(0,10000000000)

import matplotlib.pyplot as plt

import math
from functools import partial
import seaborn as sns
import numpy as np
import time
import csv

from gerrychain.random import random
random.seed(a)
from gerrychain.proposals import recom
from gerrychain import MarkovChain, Graph
from gerrychain.constraints import (
    Validator,
    single_flip_contiguous,
    within_percent_of_ideal_population,
)
from gerrychain.accept import always_accept
from gerrychain.tree import recursive_tree_part
from gerrychain.updaters import Tally, cut_edges
from gerrychain.partition import Partition
import networkx as nx
from crossover_functions import *
from utility_functions import *

####################################################################################


def hill_climb(partition):
    bound = 1
    if partition.parent is not None:
        if len(partition.parent['cut_edges']) < len(partition['cut_edges']):
            bound = 0
    
    return random.random() < bound
        
        
def anneal(partition, t):
    tmax = 1.5
    tmin = .005    
    Temp = tmax-((tmax-tmin)/num_steps)*t
    bound = 1
    if partition.parent is not None:
        if len(partition.parent['cut_edges']) < len(partition['cut_edges']): 
            bound = np.e**((len(partition.parent['cut_edges'])-len(partition['cut_edges']))/Temp)
    
    return random.random() < bound        


def slow_reversible_propose(partition):
    """Proposes a random boundary flip from the partition in a reversible fashion
    by selecting a boundary node at random and uniformly picking one of its
    neighboring parts.
    Temporary version until we make an updater for this set.
    :param partition: The current partition to propose a flip from.
    :return: a proposed next `~gerrychain.Partition`
    """

    b_nodes = {x[0] for x in partition["cut_edges"]}.union(
        {x[1] for x in partition["cut_edges"]}
    )

    flip = random.choice(list(b_nodes))
    neighbor_assignments = list(
        set(
            [
                partition.assignment[neighbor]
                for neighbor in partition.graph.neighbors(flip)
            ]+[partition.assignment[flip]]
        )
    )
    neighbor_assignments.remove(partition.assignment[flip])

    flips = {flip: random.choice(neighbor_assignments)}

    return partition.flip(flips)


def reversible_propose(partition):
    boundaries1 = {x[0] for x in partition["cut_edges"]}.union(
        {x[1] for x in partition["cut_edges"]}
    )

    flip = random.choice(list(boundaries1))
    return partition.flip({flip: -partition.assignment[flip]})


##############################HILLCLIMBING##############################
def hillclimb_run(proposal_func,constraints, initial_partition, num_steps, print_step, print_map = False):
    print("STARTING HILL")
    gchain = MarkovChain(
        proposal_func,
        constraints = constraints,
        accept=hill_climb,
        initial_state=initial_partition,
        total_steps=num_steps,
    )

    t = 0
    cuts = []
    for part in gchain:
        cuts.append(len(part["cut_edges"]))
        t += 1
        if t % print_step == 0:
            print(t)
            record_partition(part, t, 'mid_hill', graph_name, exp_num)
            if print_map:
                gdf_print_map(part, './opt_plots/Hill_middle_'+ str(t)+'_'+ str(exp_num) +'.png', gdf, unit_name)

    print("FINISHED HILL")
    record_partition(part, t, 'end_hill', graph_name, exp_num)
    return part, cuts

##############################Anneal##############################
def anneal_run(proposal_func,constraints, initial_partition, num_steps, print_step, print_map = False):
    print("STARTING ANNEAL")
    t = 0
    gchain = MarkovChain(
        proposal_func,
        constraints=constraints,
        accept=lambda x: anneal(x,t),
        initial_state=initial_partition,
        total_steps=num_steps,
    )

    cuts = []
    for part in gchain:
        cuts.append(len(part["cut_edges"]))
        t += 1
        if t % print_step == 0:
            print(t)
            record_partition(part, t, 'mid_anneal', graph_name, exp_num)
            if print_map:
                gdf_print_map(part, './opt_plots/Anneal_middle_'+ str(t)+'_'+ str(exp_num) +'.png', gdf, unit_name)

    print("FINISHED ANNEAL")
    record_partition(part, t, 'end_anneal', graph_name, exp_num)
    return part, cuts


##############################Evolutionary##############################
def evol_run(crossover_func, crossover_prob, proposal_func,validator, acceptance_func, initial_partitions, num_steps, print_step, print_map = False):
    print("STARTING EVOL")
    chain_list = []
    for i in range(population_size):
        gchain = MarkovChain(
            proposal=proposal_func,
            constraints=constraints,
            accept=acceptance_func,
            initial_state=initial_partitions[i],
            total_steps=num_steps,
        )
        chain_list.append(gchain)
    zipped = zip(*chain_list)

    t = 0
    cuts = []
    #can change functions so that not every step has mutation + crossover
    for step in zipped:
        cuts.append(min([len(part["cut_edges"]) for part in step]))

        if random.random() < crossover_prob:
            # can change so that parents are chosen optimally
            # can change so that multiple crossover steps occur each chain step
            p1, p2 = random.sample(range(len(step)), 2)
            part1 = step[p1]
            part2 = step[p2]
            child_1, child_2 = crossover_func(part1, part2)

            chain_list[p1].state = child_1
            chain_list[p2].state = child_2
            #should these be the parent states instead?
            chain_list[p1].state.parent = None
            chain_list[p1].state.parent = None

        t += 1
        if t % print_step == 0:
            print(t)
            for part in step:
                record_partition(part, t, 'mid_evol', graph_name, exp_num)
                if print_map:
                    gdf_print_map_set(step, './opt_plots/evol_middle_'+ str(t)+'_'+ str(exp_num) +'.png', gdf, unit_name)

    print("FINISHED EVOL")
    end_partitions = [part for part in step]
    for part in end_partitions:
        record_partition(part, t, 'end_evol', graph_name, exp_num)
    return end_partitions, cuts

######################################################################

# intialize graph
exp_num = 1
#IOWA
# k = 4
# graph_name = 'iowa'
# graph_path = './input_data/'+graph_name+'.json'
# graph = Graph.from_json(graph_path)
# num_districts = k
# ideal_pop = sum([graph.nodes[v]["TOTPOP"] for v in graph.nodes()])/num_districts
# unit_name = 'GEOID10'
# area_name = 'area'
# x_name = 'INTPTLON10'
# y_name = 'INTPTLAT10'
# shapefile_name = 'IA_counties'
# gdf = gpd.read_file('./input_data/'+shapefile_name)
# gdf = gdf.to_crs({'init': 'epsg:26775'})

#TEXAS
k=36
graph_name = 'Texas'
graph_path = './input_data/tx.json'
graph = Graph.from_json(graph_path)
shapefile_path = './input_data/Texas_xy/Texas_xy.shp'
gdf = gpd.read_file(shapefile_path)
num_districts = k
ideal_pop = sum([graph.nodes[v]["TOTPOP"] for v in graph.nodes()])/num_districts
unit_name = 'CNTYVTD'
area_name = 'Shape_area'
x_name = 'x_val'
y_name = 'y_val'
gdf = gdf.to_crs({'init': 'epsg:26775'})


for node in graph.nodes():
    graph.nodes[node]["x"] = float(graph.nodes[node][x_name])
    graph.nodes[node]["y"] = float(graph.nodes[node][y_name])
    graph.nodes[node]["area"] = float(graph.nodes[node][area_name])

#######################################################################
# ititialize partitions
tree_walk = False
population_size = 10



#IOWA starting
# init_dists = {88: 0, 75: 0, 98: 0, 82: 0, 45: 0, 29: 0, 33: 0, 37: 0, 96: 0, 80: 0, 78: 0, 40: 0, 81: 0, 63: 0, 83: 0, 1: 0, 74: 0, 0: 0, 62: 0, 4: 0, 86: 0, 27: 0, 6: 0, 52: 0, 89: 0, 11: 0, 91: 0, 15: 0, 23: 0, 31: 0, 85: 0, 59: 0, 5: 1, 10: 1, 38: 1, 8: 1, 92: 1, 16: 1, 24: 1, 53: 1, 76: 1, 94: 1, 28: 1, 35: 1, 34: 1, 51: 2, 93: 2, 54: 2, 95: 2, 25: 2, 73: 2, 22: 2, 47: 2, 71: 2, 41: 2, 60: 3, 17: 3, 12: 3, 30: 3, 65: 3, 79: 3, 36: 3, 50: 3, 56: 3, 58: 3, 49: 3, 18: 3, 9: 3, 7: 3, 87: 3, 90: 3, 44: 3, 77: 3, 13: 3, 14: 3, 66: 3, 42: 3, 20: 3, 69: 3, 55: 3, 70: 3, 46: 3, 19: 3, 61: 3, 2: 3, 67: 3, 97: 3, 43: 3, 72: 3, 26: 3, 39: 3, 84: 3, 32: 3, 64: 3, 21: 3, 57: 3, 3: 3, 48: 3, 68: 3}
# cddict = {v:int(init_dists[v]) for v in graph.nodes()}

#general starting
cddict = recursive_tree_part(graph, range(k), ideal_pop, "TOTPOP", .02, 3)

updaters = {
    "population": Tally("TOTPOP", alias="population"),
    "cut_edges": cut_edges,
    "centroids": centroids_x_y_area
}

init_partition = Partition(graph, assignment=cddict, updaters=updaters)


if tree_walk:
    ideal_population = sum(init_partition["population"].values()) / len(init_partition)

    proposal = partial(
        recom,
        pop_col="TOTPOP",
        pop_target=ideal_population,
        epsilon=0.02,
        node_repeats=1,
    )

    popbound = within_percent_of_ideal_population(init_partition, 0.05)

    gchain = MarkovChain(
        proposal, 
        Validator([popbound]),
        accept=always_accept,
        initial_state=init_partition,
        total_steps=100,
    )

    t = 0
    for part in gchain:
        t += 1

    init_partition = part
    print("FINISHED TREE WALK")


gdf_print_map(init_partition, './opt_plots/starting_plan.png', gdf, unit_name)

record_partition(init_partition, 0, 'starting', graph_name, exp_num)


partition_list = [init_partition]
for i in range(population_size-1):
    new_plan = recursive_tree_part(graph, range(k), ideal_pop, "TOTPOP", .02, 3)
    partition = Partition(graph, assignment=new_plan, updaters=updaters)
    partition_list.append(partition)
print([len(i['cut_edges']) for i in partition_list])
gdf_print_map_set(partition_list, './opt_plots/starting_population.png', gdf, unit_name)

print("FINISHED INITIALIZING")
############################## Compare ##############################


evol_num_steps = 100
num_steps = evol_num_steps*population_size
max_adjust = 200
max_adjust_chen = 10
prob_crossover = 0.05
evol_print_step = 10
print_step = evol_print_step*population_size
print_maps = False
ep = 0.05
hill_anneal_for_population = False
crossover_func_book = partial(book_chapter_crossover, k= k, ep = ep, max_adjust = max_adjust, ideal_pop = ideal_pop)
crossover_func_chen = partial(chen_crossover, k= k, ep = ep, max_adjust = max_adjust_chen, ideal_pop = ideal_pop)
crossover_func_half_recom = partial(half_half_recom_crossover, k= k, ep = ep, max_adjust = max_adjust, ideal_pop = ideal_pop)

popbound = within_percent_of_ideal_population(init_partition, ep)
constraints = Validator([single_flip_contiguous, popbound])

if hill_anneal_for_population:
    min_hill_cut_len = math.inf
    min_hill_cuts = []
    for i in range(population_size):
        hill_part, hill_cuts = hillclimb_run(slow_reversible_propose,Validator([single_flip_contiguous, popbound]), partition_list[i], evol_num_steps, evol_print_step, print_map = print_maps)
        gdf_print_map(hill_part, './opt_plots/hill_end_'+ str(exp_num) +'_'+str(i)+'.png', gdf, unit_name)
        if min(hill_cuts) < min_hill_cut_len:
            min_hill_cut_len = min(hill_cuts)
            min_hill_cuts = hill_cuts

    min_anneal_cut_len = math.inf
    min_anneal_cuts = []
    for i in range(population_size):
        anneal_part, anneal_cuts = anneal_run(slow_reversible_propose,Validator([single_flip_contiguous, popbound]), partition_list[i], evol_num_steps, evol_print_step, print_map = print_maps)
        gdf_print_map(anneal_part, './opt_plots/anneal_end_'+ str(exp_num) +'_'+str(i)+'.png', gdf, unit_name)
        if min(anneal_cuts) < min_anneal_cut_len:
            min_anneal_cut_len = min(anneal_cuts)
            min_anneal_cuts = anneal_cuts

else:
    hill_part, min_hill_cuts = hillclimb_run(slow_reversible_propose,Validator([single_flip_contiguous, popbound]), init_partition, num_steps, print_step, print_map = print_maps)
    anneal_part, min_anneal_cuts = anneal_run(slow_reversible_propose,Validator([single_flip_contiguous, popbound]), init_partition, num_steps, print_step, print_map = print_maps)
    gdf_print_map(hill_part, './opt_plots/hill_end_'+ str(exp_num) + '.png', gdf, unit_name)
    gdf_print_map(anneal_part, './opt_plots/anneal_end_'+ str(exp_num) + '.png', gdf, unit_name)

evol_parts_book, evol_min_cuts_book = evol_run(crossover_func_book, prob_crossover, slow_reversible_propose, Validator([single_flip_contiguous, popbound]), hill_climb, partition_list, evol_num_steps, evol_print_step, print_map = print_maps)
evol_parts_chen, evol_min_cuts_chen = evol_run(crossover_func_chen, prob_crossover, slow_reversible_propose, Validator([single_flip_contiguous, popbound]), hill_climb, partition_list, evol_num_steps, evol_print_step, print_map = print_maps)
evol_parts_half_recom, evol_min_cuts_half_recom = evol_run(crossover_func_half_recom, prob_crossover, slow_reversible_propose, Validator([single_flip_contiguous, popbound]), hill_climb, partition_list, evol_num_steps, evol_print_step, print_map = print_maps)


gdf_print_map_set(evol_parts_book, './opt_plots/evol_book_end_'+ str(exp_num) +'.png', gdf, unit_name)
gdf_print_map_set(evol_parts_chen, './opt_plots/evol_chen_end_'+ str(exp_num) +'.png', gdf, unit_name)
gdf_print_map_set(evol_parts_half_recom, './opt_plots/evol_half_recom_end_'+ str(exp_num) +'.png', gdf, unit_name)

plt.figure()
plt.title("Cut Lengths")
plt.plot(min_hill_cuts,'r',label='Hill')
plt.plot(min_anneal_cuts,'b',label='Anneal')
plt.plot(evol_min_cuts_book,'darkgreen',label='EvolutionaryMin_book')
plt.plot(evol_min_cuts_chen,'lightgreen',label='EvolutionaryMin_chen')
plt.plot(evol_min_cuts_half_recom,'purple',label='EvolutionaryMin_half_recom')
plt.legend()
plt.savefig("./opt_plots/cuts_comparison" + str(exp_num) + ".png")
plt.close()

with open('./output_data/'+ graph_name+'_cuts' +'_'+ str(exp_num) + ".txt", 'a+') as partition_file:
    writer = csv.writer(partition_file)
    writer.writerow([time.time(), graph_name, num_steps, 'hill', len(graph.nodes())]+min_hill_cuts)
    writer.writerow([time.time(), graph_name, num_steps, 'anneal', len(graph.nodes())]+min_anneal_cuts)
    writer.writerow([time.time(), graph_name, num_steps, 'evol_max_book', len(graph.nodes())]+evol_min_cuts_book)
    writer.writerow([time.time(), graph_name, num_steps, 'evol_max_chen', len(graph.nodes())]+evol_min_cuts_chen)
    writer.writerow([time.time(), graph_name, num_steps, 'evol_max_half_recom', len(graph.nodes())]+evol_min_cuts_half_recom)


