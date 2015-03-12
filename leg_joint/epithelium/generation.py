# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import graph_tool.all as gt


def cylindrical(n_cells_circum, n_cells_length, l_0, h_0):

    n_circum = n_cells_circum * 3
    n_length = n_cells_length * 3
    rho_c = n_circum * l_0 / (2 * np.pi)
    rho_lumen = rho_c - h_0

    delta_theta = 2 * np.pi / n_circum
    delta_z = delta_theta * rho_c * np.sqrt(3)/2.

    zt_grid = np.mgrid[:n_length, :n_circum]
    thetas = zt_grid[1].astype('float')
    thetas[::2, ...] += 0.5
    is_cell_vert = np.zeros_like(zt_grid[1])
    is_cell_vert[::2, ::3] = 1
    is_cell_vert[1::2, 2::3] = 1
    is_cell_vert = is_cell_vert.flatten()

    zeds = zt_grid[0].astype('float')
    vertex_df = np.zeros((n_circum * n_length, 2))
    vertex_df[:, 0] = thetas.flatten()
    vertex_df[:, 1] = zeds.flatten()

    vertex_df = pd.DataFrame(vertex_df, columns=['theta', 'z'])
    #vertex_df = vertex_df.sort_index(by=['z', 'theta'])
    vertex_df.index.name = 'vertex'
    vertex_df.theta *= delta_theta

    vertex_df.z *= delta_z
    vertex_df.z -= vertex_df.z.max() / 2
    vertex_df['x'] = np.cos(vertex_df.theta) * rho_c
    vertex_df['y'] = np.sin(vertex_df.theta) * rho_c
    vertex_df['is_cell_vert'] = is_cell_vert
    graph, pos_vp = gt.geometric_graph(vertex_df[['x', 'y', 'z']], l_0*1.1)
    graph.set_directed(True)

    _is_cv = graph.new_vertex_property('bool')
    _is_cv.a = is_cell_vert
    is_cell_vert = _is_cv

    is_junction_edge = graph.new_edge_property('bool')
    is_junction_edge.a = 1
    to_remove = []

    for edge in graph.edges():
        srce, trgt = edge
        if is_cell_vert[srce] and not is_cell_vert[trgt]:
            is_junction_edge[edge] = 0
        elif not is_cell_vert[srce] and is_cell_vert[trgt]:
            ### Flip the edge
            to_remove.append(edge)
            new = graph.add_edge(trgt, srce)
            is_junction_edge[edge] = 0
        elif is_cell_vert[srce] and is_cell_vert[trgt]:
            raise ValueError('Invalid cell to cell edge {}'.format(edge))

    for edge in to_remove:
        graph.remove_edge(edge)

    edges_idx = [(graph.vertex_index[s], graph.vertex_index[t])
                 for (s, t) in graph.edges()]
    edges_idx = pd.MultiIndex.from_tuples(edges_idx, names=('source', 'target'))
    edges_df = pd.DataFrame(index=edges_idx)
    edges_df['is_junction_edge'] = is_junction_edge.fa

    return graph, vertex_df, edges_df