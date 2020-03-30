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
from gerrychain.updaters import Tally, cut_edges
from gerrychain.partition import Partition
import networkx as nx
from gerrychain.tree import recursive_tree_part, bipartition_tree

from merge_functions import *
from dividing_functions import *
from shift_pop_functions import *
from utility_functions import *




def book_chapter_crossover(part1, part2, k, ep, max_adjust, ideal_pop, draw_map = False, gdf = None, unit_name = None):
        part_refine = common_refinement(part1, part2)
        part_merge = merge_parts_smallest_sum(part_refine,k)
        part_shift = shift_pop(part_merge, ep, max_adjust, ideal_pop)
        
        if draw_map:
            try:
                assert(len(gdf) >= 0)
                assert(len(unit_name) >= 0)
            except:
                print("Must specify gdf and unit name if draw_map=True")
                return part1, part2

            gdf_print_map(part1, './figures/book_chapter_crossover_parent1.png', gdf, unit_name)
            gdf_print_map(part2, './figures/book_chapter_crossover_parent2.png', gdf, unit_name)
            gdf_print_map(part_refine, './figures/book_chapter_crossover_refine.png', gdf, unit_name)
            gdf_print_map(part_merge, './figures/book_chapter_crossover_merge.png', gdf, unit_name)
            gdf_print_map(part_shift, './figures/book_chapter_crossover_shift.png', gdf, unit_name)
            print('parent 1:', len(part1["cut_edges"]), ' parent 2:', len(part2["cut_edges"]),' refine:', len(part_refine["cut_edges"]),' merge:', len(part_merge["cut_edges"]),' shift', len(part_shift["cut_edges"]))
        if len(part1["cut_edges"]) > len(part2["cut_edges"]):
            better_parent = part2
        else:
            better_parent = part1
        if max([abs(part_shift["population"][i]-ideal_pop) for i in range(len(part_shift))]) <= ep*ideal_pop:
            return part_shift, better_parent
        else:
            return part1, part2


def chen_crossover(part1, part2, k, ep, max_adjust, ideal_pop, dist_func = centroid_dist_lat_lon,draw_map = False,gdf = None, unit_name = None):
        part_refine = common_refinement(part1, part2)
        part_merge = merge_parts_chen(part_refine, k, dist_func)
        part_shift = shift_chen(part_merge, ep, max_adjust, ideal_pop, dist_func)
                
        if draw_map:
            try:
                assert(len(gdf) >= 0)
                assert(len(unit_name) >= 0)
            except:
                print("Must specify gdf and unit name if draw_map=True")
                return part1, part2

            gdf_print_map(part1, './figures/chen_crossover_parent1.png', gdf, unit_name)
            gdf_print_map(part2, './figures/chen_crossover_parent2.png', gdf, unit_name)
            gdf_print_map(part_refine, './figures/chen_crossover_refine.png', gdf, unit_name)
            gdf_print_map(part_merge, './figures/chen_crossover_merge.png', gdf, unit_name)
            gdf_print_map(part_shift, './figures/chen_crossover_shift.png', gdf, unit_name)
            print('parent 1:', len(part1["cut_edges"]), ' parent 2:', len(part2["cut_edges"]),' refine:', len(part_refine["cut_edges"]),' merge:', len(part_merge["cut_edges"]),' shift', len(part_shift["cut_edges"]))

        if len(part1["cut_edges"]) > len(part2["cut_edges"]):
            better_parent = part2
        else:
            better_parent = part1
        if max([abs(part_shift["population"][i]-ideal_pop) for i in range(len(part_shift))]) <= ep*ideal_pop:
            return part_shift, better_parent
        else:
            return part1, part2


def half_half_recom_crossover(part1, part2, k, ep, max_adjust, ideal_pop,draw_map = False,gdf = None, unit_name = None):
        half_pop = sum(part1.population.values())/2
        part_half1, part_half2 = half_half(part1, part2, half_pop)
        part_merge1 = merge_parts_smallest_sum_alt(part_half1, k)
        part_merge2 = merge_parts_smallest_sum_alt(part_half2, k)

        for v in part_merge1.graph.nodes():
            part_merge1.graph.nodes[v]['assign_col'] = part_merge1.assignment[v]
        for v in part_merge2.graph.nodes():
            part_merge2.graph.nodes[v]['assign_col'] = part_merge2.assignment[v]    

        part_shift_recom1 = shift_pop_recom(part_merge1, ep, max_adjust, ideal_pop, assign_col = 'assign_col', node_repeats = 10)
        part_shift_recom2 = shift_pop_recom(part_merge2, ep, max_adjust, ideal_pop, assign_col = 'assign_col', node_repeats = 10)
                
        if draw_map:
            try:
                assert(len(gdf) >= 0)
                assert(len(unit_name) >= 0)
            except:
                print("Must specify gdf and unit name if draw_map=True")
                return part1, part2

            gdf_print_map(part1, './figures/half_recom_crossover_parent1.png', gdf, unit_name)
            gdf_print_map(part2, './figures/half_recom_crossover_parent2.png', gdf, unit_name)
            gdf_print_map(part_half1, './figures/half_recom_crossover_half1.png', gdf, unit_name)
            gdf_print_map(part_half2, './figures/half_recom_crossover_half2.png', gdf, unit_name)
            gdf_print_map(part_merge1, './figures/half_recom_crossover_merge1.png', gdf, unit_name)
            gdf_print_map(part_merge2, './figures/half_recom_crossover_merge2.png', gdf, unit_name)
            gdf_print_map(part_shift_recom1, './figures/half_recom_crossover_shift_recom1.png', gdf, unit_name)
            gdf_print_map(part_shift_recom2, './figures/half_recom_crossover_shift_recom2.png', gdf, unit_name)

            print('parent 1:', len(part1["cut_edges"]), ' parent 2:', len(part2["cut_edges"]),' half1:', len(part_half1["cut_edges"]),' half2:', len(part_half2["cut_edges"]), ' merge1:', len(part_merge1["cut_edges"]), ' merge2:', len(part_merge2["cut_edges"]), ' shift_recom1:', len(part_shift_recom1["cut_edges"]), ' shift_recom2:', len(part_shift_recom2["cut_edges"]))

        if len(part1["cut_edges"]) > len(part2["cut_edges"]):
            better_parent = part2
        else:
            better_parent = part1
        if len(part_shift_recom1["cut_edges"]) > len(part_shift_recom2["cut_edges"]):
            better_child = part_shift_recom2
            worse_child = part_shift_recom1
        else:
            better_child = part_shift_recom1
            worse_child = part_shift_recom2
        if max([abs(better_child["population"][i]-ideal_pop) for i in range(len(better_child))]) <= ep*ideal_pop:
            return better_child, better_parent
        elif max([abs(worse_child["population"][i]-ideal_pop) for i in range(len(worse_child))]) <= ep*ideal_pop:
            return worse_child, better_parent
        else:
            return part1, part2


################################  testing ########################################


def crossover_test():
    k = 4
    graph_name = 'iowa'
    graph_path = './input_data/'+graph_name+'.json'
    graph = Graph.from_json(graph_path)
    num_districts = k
    ideal_pop = sum([graph.nodes[v]["TOTPOP"] for v in graph.nodes()])/num_districts
    unit_name = 'GEOID10'
    area_name = 'area'
    x_name = 'INTPTLON10'
    y_name = 'INTPTLAT10'
    # areaC_X = "areaC_X"
    # areaC_Y = "areaC_Y"
    # area = 'area'

    for node in graph.nodes():
        graph.nodes[node]["x"] = float(graph.nodes[node][x_name])
        graph.nodes[node]["y"] = float(graph.nodes[node][y_name])
        # graph.nodes[node]["areaC_X"] = float(graph.nodes[node][area_name])*float(graph.nodes[node][x_name])
        # graph.nodes[node]["areaC_Y"] = float(graph.nodes[node][area_name])*float(graph.nodes[node][y_name])
        graph.nodes[node]["area"] = float(graph.nodes[node][area_name])


    shapefile_name = 'IA_counties'
    gdf = gpd.read_file('./input_data/'+shapefile_name)
    gdf = gdf.to_crs({'init': 'epsg:26775'})

    updaters = {
        "population": Tally("TOTPOP", alias="population"),
        "cut_edges": cut_edges,
        "centroids": centroids_x_y_area
    }

    new_plan1 = recursive_tree_part(graph, range(k), ideal_pop, "TOTPOP", .02, 3)
    part1 = Partition(graph, assignment=new_plan1, updaters=updaters)
    new_plan2 = recursive_tree_part(graph, range(k), ideal_pop, "TOTPOP", .02, 3)
    part2 = Partition(graph, assignment=new_plan2, updaters=updaters)

    max_adjust = 10000
    ep = 0.05

    print("book chapter crossover test:")
    book_child1, book_child2 = book_chapter_crossover(part1, part2, k, ep, max_adjust, ideal_pop, draw_map = True, gdf = gdf, unit_name = unit_name)
    print(len(book_child1.cut_edges), len(book_child2.cut_edges))
    print("chen crossover test:")
    chen_child1, chen_child2 = chen_crossover(part1, part2, k, ep, max_adjust, ideal_pop, draw_map = True, gdf = gdf, unit_name = unit_name)
    print(len(chen_child1.cut_edges), len(chen_child2.cut_edges))
    print("half-half recome crossover test:")
    half_recom_child1, half_recom_child2 = half_half_recom_crossover(part1, part2, k, ep, max_adjust, ideal_pop, draw_map = True, gdf = gdf, unit_name = unit_name)
    print(len(half_recom_child1.cut_edges), len(half_recom_child2.cut_edges))

if __name__ == "__main__":
    crossover_test()
