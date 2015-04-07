# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import graph_tool.all as gt

vertex_data = {
    ## Coordinates
    'x': (0., np.float),
    'y': (0., np.float),
    'z': (0., np.float),
    'rho': (0., np.float),
    'theta': (0., np.float),
    'height': (0., np.float),
    ## Geometry
    'perimeter': (0., np.float),
    'area': (0., np.float),
    'vol': (0., np.float),
    ## Topology
    'is_cell_vert': (0, np.bool),
    'is_alive': (1, np.bool),
    'is_active_vert': (1, np.bool),
    'num_sides': (1, np.int),
    ## Dynamical
    'contractility': (0.014, np.float),
    'vol_elasticity': (0.014, np.float)}
edge_data = {
    ## Coordinates
    'dx': (0., np.float),
    'dy': (0., np.float),
    'dz': (0., np.float),
    'edge_length': (0., np.float),
    ## Gradients
    'gx': (0., np.float),
    'gy': (0., np.float),
    'gz': (0., np.float),
    ## Topological
    'is_junction_edge': (0, np.bool),
    ## Dynamical parameters
    'line_tension': (0., np.float),
    'radial_tension': (0., np.float)}


def complete_dframes(vertex_df, edge_df,
                     vertex_data, edge_data):
    for key, (default, dtype) in vertex_data.items():
        if not key in vertex_df.columns:
            vertex_df[key] = default
            vertex_df[key] = vertex_df[key].astype(dtype)

    for key, (default, dtype) in edge_data.items():
        if not key in edge_df.columns:
            edge_df[key] = default
            edge_df[key] = edge_df[key].astype(dtype)


def base_grid(n_cells_x, n_cells_y,
              delta_x=1, delta_y=1,
              centered=True):

    n_x = n_cells_x
    n_y = n_cells_y * 3

    xy_grid = np.mgrid[:n_x, :n_y]

    xs = xy_grid[0].astype('float')
    ys = xy_grid[1].astype('float')

    ys[::2, ...] += 0.5
    is_cell_vert = np.zeros_like(xy_grid[1]).astype(np.bool)
    is_cell_vert[::2, ::3] = 1
    is_cell_vert[1::2, 2::3] = 1
    is_cell_vert = is_cell_vert.flatten()

    vertex_df = np.zeros((n_x * n_y, 2))
    vertex_df[:, 0] = xs.flatten()
    vertex_df[:, 1] = ys.flatten()
    vertex_df = pd.DataFrame(vertex_df, columns=['x', 'y'])
    vertex_df.index.name = 'vertex_index'
    vertex_df.x *= delta_x
    vertex_df.y *= delta_y
    vertex_df['is_cell_vert'] = is_cell_vert
    if centered:
        vertex_df.x -= vertex_df.x.max() / 2
        vertex_df.y -= vertex_df.y.max() / 2

    return vertex_df


def cylindrical(n_cells_length,
                n_cells_circum,
                l_0, h_0):

    n_circum = n_cells_circum * 3
    rho_c = n_circum * l_0 / (2 * np.pi)

    delta_theta = 2 * np.pi / n_circum
    delta_z = delta_theta * rho_c * np.sqrt(3)/2.

    _vertex_df = base_grid(n_cells_length,
                           n_cells_circum,
                           delta_x=delta_z,
                           delta_y=delta_theta)

    vertex_df = _vertex_df.copy()
    vertex_df['z'] = _vertex_df.x
    vertex_df['theta'] = _vertex_df.y

    vertex_df['x'] = np.cos(vertex_df.theta) * rho_c
    vertex_df['y'] = np.sin(vertex_df.theta) * rho_c
    vertex_df['rho'] = np.hypot(vertex_df.y, vertex_df.x)
    vertex_df['theta'] = np.arctan2(vertex_df.y, vertex_df.x)
    vertex_df['height'] = h_0

    graph, pos_vp = gt.geometric_graph(vertex_df[['x', 'y', 'z']],
                                       l_0*1.1)
    graph.set_directed(True)

    is_cell_vert = graph.new_vertex_property('bool')
    is_cell_vert.a = vertex_df.is_cell_vert
    is_junction_edge = graph.new_edge_property('bool')
    is_junction_edge.a = 1
    reorient_edges(graph, is_cell_vert, is_junction_edge)

    edge_idx = [(graph.vertex_index[s], graph.vertex_index[t])
                 for (s, t) in graph.edges()]
    edge_idx = pd.MultiIndex.from_tuples(edge_idx,
                                         names=('source', 'target'))
    edge_df = pd.DataFrame(index=edge_idx)
    edge_df['is_junction_edge'] = is_junction_edge.fa.astype(np.bool)
    complete_dframes(vertex_df, edge_df,
                     vertex_data, edge_data)

    return graph, vertex_df, edge_df



def reorient_edges(graph, is_cell_vert, is_junction_edge):
    '''Reorient the graph such that cell to junction edges
    are always from cell to junction. Modifies `graph` and
    `is_junction_edge` inplace

    '''
    to_remove = []
    for edge in graph.edges():
        srce, trgt = edge
        if is_cell_vert[srce] and not is_cell_vert[trgt]:
            is_junction_edge[edge] = 0
        elif not is_cell_vert[srce] and is_cell_vert[trgt]:
            ### Flip the edge
            to_remove.append(edge)
            new = graph.add_edge(trgt, srce)
            is_junction_edge[new] = 0
        elif is_cell_vert[srce] and is_cell_vert[trgt]:
            raise ValueError(
                'Invalid cell to cell edge {}'.format(edge))
        else:
            is_junction_edge[edge] = 1
    for edge in to_remove:
        graph.remove_edge(edge)
