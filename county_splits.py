import random
a = random.randint(0,10000000000)
import networkx as nx
from gerrychain.random import random
random.seed(a)
import csv
import os
from functools import partial
import json
import random
import numpy as np

import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

#test comment!!
from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    accept,
    constraints,
    updaters,
)
from gerrychain.metrics import efficiency_gap, mean_median
from gerrychain.proposals import recom
from gerrychain.updaters import cut_edges
from gerrychain.tree import PopulatedGraph, contract_leaves_until_balanced_or_none, recursive_tree_part, predecessors, bipartition_tree
from networkx.algorithms import tree
from collections import deque, namedtuple


first_check_counties = True

def draw_graph(G, plan_assignment, fig_name):
    cdict = {graph.nodes[i]['GEOID10']:plan_assignment[i] for i in plan_assignment.keys()}
    df['color'] = df.apply(lambda x: cdict[x['GEOID10']], axis=1)
    fig,ax = plt.subplots()
    counties.geometry.boundary.plot(color=None,edgecolor='k',linewidth = 2,ax=ax)
    df.plot(column='color',ax = ax, cmap = 'tab20')
    ax.set_axis_off()
    plt.savefig(fig_name)
    plt.close()

def num_splits(partition):
    df["current"] = df[unique_label].map(dict(partition.assignment))
    splits = sum(df.groupby("COUNTYFP10")["current"].nunique() > 1)
    return splits


def cut_length(partition):
    return len(partition["cut_edges"])

def county_random_spanning_tree(graph, county_col="COUNTYFP10"):
    for edge in graph.edges:
        if graph.nodes[edge[0]][county_col] == graph.nodes[edge[1]][county_col]:
            graph.edges[edge]["weight"] = 1 + random.random()
        else:
            graph.edges[edge]["weight"] = 10 + random.random()
    spanning_tree = tree.minimum_spanning_tree(
        graph, algorithm="kruskal", weight="weight"
    )
    return spanning_tree

def split_tree_at_county(h, choice=random.choice, county_col="COUNTYFP10"):
    root = choice([x for x in h if h.degree(x) > 1])
    # BFS predecessors for iteratively contracting leaves
    pred = predecessors(h.graph, root)

    leaves = deque(x for x in h if h.degree(x) == 1)
    while len(leaves) > 0:
        leaf = leaves.popleft()
        parent = pred[leaf]
        if h.graph.nodes[parent][county_col] != h.graph.nodes[leaf][county_col] and h.has_ideal_population(leaf):
            return h.subsets[leaf]
        # Contract the leaf:
        h.contract_node(leaf, parent)
        if h.degree(parent) == 1 and parent != root:
            leaves.append(parent)
    return None


def county_bipartition_tree(
    graph,
    pop_col,
    pop_target,
    epsilon,
    county_col="COUNTYFP10",
    node_repeats=1,
    spanning_tree=None,
    choice=random.choice, 
    attempts_before_giveup = 100):

    populations = {node: graph.nodes[node][pop_col] for node in graph}

    balanced_subtree = None
    if spanning_tree is None:
        spanning_tree = county_random_spanning_tree(graph, county_col=county_col)
    restarts = 0
    counter = 0
    while balanced_subtree is None and counter < attempts_before_giveup:
        # print(counter)
        if restarts == node_repeats:
            spanning_tree = county_random_spanning_tree(graph, county_col=county_col)
            restarts = 0
            counter +=1
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        if first_check_counties and restarts == 0:
            balanced_subtree = split_tree_at_county(h, choice=choice, county_col=county_col)
        if balanced_subtree is None:
            h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
            balanced_subtree = contract_leaves_until_balanced_or_none(h, choice=choice)
        restarts += 1

    if counter >= attempts_before_giveup:
        return set()
    return balanced_subtree


