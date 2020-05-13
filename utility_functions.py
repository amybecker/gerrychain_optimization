#TEST CHANGE@@
import random
a = random.randint(0,10000000000)
# import matplotlib
# matplotlib.use("Agg")

import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import networkx as nx
import geopandas as gpd
from math import sin, cos, sqrt, atan2, radians, ceil, floor
import csv
import time
from gerrychain.partition import Partition, GeographicPartition

def print_map(partition_assign, name):
    plt.figure()
    nx.draw(graph,pos,node_color=[partition_assign[x]+5 for x in graph.nodes()],node_size=ns,node_shape="s",cmap="gist_ncar")
    plt.title(name)
    plt.savefig(name+".png")
    plt.close()

def gdf_print_map(partition, filename, gdf, unit_name, dpi = 300):
    cdict = {partition.graph.nodes[i][unit_name]:partition.assignment[i] for i in partition.graph.nodes()}
    gdf['color'] = gdf.apply(lambda x: cdict[x[unit_name]], axis=1)
    plt.figure()
    gdf.plot(column='color')
    plt.savefig(filename, dpi = dpi)
    plt.close('all')

def gdf_print_map_set(partitions, filename, gdf, unit_name):
    fig, axes = plt.subplots(ceil(len(partitions)/5), 5, figsize=(3*5, 3*ceil(len(partitions)/5)))
    for i in range(len(partitions)):
        cdict = {partitions[i].graph.nodes[v][unit_name]:partitions[i].assignment[v] for v in partitions[i].graph.nodes()}
        gdf['color'] = gdf.apply(lambda x: cdict[x[unit_name]], axis=1)
        axes[floor(i/5),i%5].set_title('map '+str(i))
        axes[floor(i/5),i%5].set_axis_off()
        gdf.plot(ax=axes[floor(i/5),i%5], column='color')
    plt.savefig(filename)
    plt.close('all')

def record_partition(partition, t, run_type, graph_name, exp_num):
    with open('./output_data/Partition_records'+ graph_name+'_'+str(exp_num)+'.txt', 'a+') as partition_file:
        writer = csv.writer(partition_file)
        out_row = [time.time(), exp_num, graph_name, run_type, t, len(partition.graph.nodes())]+[partition.assignment[x] for x in partition.graph.nodes()]
        writer.writerow(out_row)   

def centroids(partition):
    CXs = {k: partition["Sum_areaCX"][k]/partition["Sum_area"][k] for k in list(partition.parts.keys())}
    CYs = {k: partition["Sum_areaCY"][k]/partition["Sum_area"][k] for k in list(partition.parts.keys())}
    centroids = {k: (CXs[k], CYs[k]) for k in list(partition.parts.keys())}
    return centroids
    
def centroids_x_y_area(partition):
    CXs = {k: sum([partition.graph.nodes[v]['area']*partition.graph.nodes[v]['x'] for v in partition.parts[k]])/sum([partition.graph.nodes[v]['area'] for v in partition.parts[k]]) for k in partition.parts.keys()}
    CYs = {k: sum([partition.graph.nodes[v]['area']*partition.graph.nodes[v]['y'] for v in partition.parts[k]])/sum([partition.graph.nodes[v]['area'] for v in partition.parts[k]]) for k in partition.parts.keys()}
    centroids = {k: (CXs[k], CYs[k]) for k in list(partition.parts.keys())}
    return centroids
    
def centroid_dist_euclidean(u,v):
    return math.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)

def centroid_dist_lat_lon(u,v):
    R = 6373.0

    lat1 = radians(u[1])
    lon1 = radians(u[0])
    lat2 = radians(v[1])
    lon2 = radians(v[0])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def check_pop(partition, ideal_pop):
    test_pop = {k : abs(partition["population"][k]- ideal_pop)/ideal_pop for k in partition.parts.keys()}
    return sorted(list(test_pop.values()))

def ordered_pop(partition):
    return sorted(list(partition["population"].keys()))


def max_pop_dev(partition, ideal_pop):
    #returns max deviation from ideal across all districts in partition
    dev_from_ideal = {k : abs(partition["population"][k]- ideal_pop)/ideal_pop for k in partition.parts.keys()}
    return max(dev_from_ideal.values())

def parts_connected(partition):
    #returns True if all parts in partition are connected
    dist_checks = []
    for part in partition.parts.values():
        sub_graph = partition.graph.subgraph(set(part))
        dist_checks.append(nx.is_connected(sub_graph))
    return False not in dist_checks

    
def shift_part_keys(partition):
     #get final partition to have parts in 0 indexed range of # total parts
    final_values = list(set(dict(partition.assignment).values()))
    final_assign = {}
    for key,value in dict(partition.assignment).items():
        final_assign[key] = final_values.index(partition.assignment[key])
    
    return Partition(partition.graph, final_assign, partition.updaters)

def pop_dev(partition):
    return (max(partition.population.values())- min(partition.population.values()))/(sum(partition.population.values())/len(partition))




################################  testing ########################################