# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

import pandas as pd
import graph_tool.all as gt




from leg_joint.epithelium import graph_dataframe_interface as gdf_i

def min_df():

    vertex_df = pd.DataFrame(columns=['a', 'b', 'c', 't'])
    vertex_df.a = np.linspace(0., 1., 10)
    vertex_df.b = np.random.randint(0, 2, 10)
    vertex_df.c = np.arange(0, 10).astype(np.int)
    vertex_df.t = ['a'] * 10

    edge_df = pd.DataFrame(columns=['d', 'e', 'f'])
    edge_df.d = np.linspace(0., 1., 10)
    edge_df.e = np.arange(0, 10).astype(np.int)
    edge_df.f = np.random.randint(0, 2, 10)

    return vertex_df, edge_df


def test_complete_pmaps():

    vertex_df, edge_df = min_df()
    graph = gt.Graph()
    gdf_i.complete_pmaps(graph, vertex_df, edge_df)

    assert all([col in graph.vertex_properties for col in ['a', 'b', 'c']])
    assert all([col in graph.edge_properties for col in ['d', 'e', 'f']])

def test_update_pmaps():

    vertex_df, edge_df = min_df()
    graph = gt.Graph()
    verts = graph.add_vertex(10)
    edges = graph.add_edge_list([[i, i] for i in range(10)])

    gdf_i.complete_pmaps(graph, vertex_df, edge_df)
    gdf_i.update_pmaps(graph, vertex_df, edge_df)
    for col, prop in graph.vertex_properties.items():
        np.testing.assert_array_equal(prop.fa, vertex_df[col])
    for col, prop in graph.edge_properties.items():
        np.testing.assert_array_equal(prop.fa, edge_df[col])

def test_update_dframes():

    vertex_df, edge_df = min_df()
    graph = gt.Graph()
    verts = graph.add_vertex(10)
    edges = graph.add_edge_list([[i, i] for i in range(10)])
    gdf_i.complete_pmaps(graph, vertex_df, edge_df)
    gdf_i.update_pmaps(graph, vertex_df, edge_df, vcols=['a', 'b'])
    graph.vertex_properties['b'].a = 0
    gdf_i.update_dframes(graph, vertex_df, edge_df, vcols=['b'])
    assert not np.all(vertex_df.b)
