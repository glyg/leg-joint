# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function



import pandas as pd
import numpy as np

def prepare(vertex_df, edges_df, triangles):

    srcs = edges_df.index.get_level_values('source')
    trgts = edges_df.index.get_level_values('target')

    coords = ['ixs', 'wys', 'zeds']
    dcoords = ['d'+c for c in coords]
    ucoords = ['u'+c for c in coords]

    edges_df[dcoords] = vertex_df[coords].loc[trgts].values - vertex_df[coords].loc[srcs].values
    edges_df.edge_lengths = (edges_df[dcoords]**2).sum(axis=1)**0.5
    deltas = edges_df[dcoords]

    cell_idxs = vertex_df[vertex_df['is_cell_vert']==1].index

    tri_cells = pd.Index(triangles[:, 0], name='cells')
    tri_jv0s = pd.Index(triangles[:, 1], name='jv_i')
    tri_jv1s = pd.Index(triangles[:, 2], name='jv_j')
    tri_jes = pd.MultiIndex.from_arrays((triangles[:, 1], triangles[:, 2]), names=('source', 'target'))
    tri_ak = pd.MultiIndex.from_arrays((triangles[:, 0], triangles[:, 1]), names=('source', 'target'))
    tri_am = pd.MultiIndex.from_arrays((triangles[:, 0], triangles[:, 2]), names=('source', 'target'))


    r_ijs = deltas.loc[tri_jes]
    r_ak = deltas.loc[tri_ak]
    r_am = deltas.loc[tri_am]
    crosses = pd.DataFrame(np.cross(r_ak, r_am), index=tri_cells)
    cxs_norms = ((crosses**2).sum(axis=1)**0.5)
    areas = cxs_norms.sum(level='cells')/2
    normals = crosses / cxs_norms.repeat(3).reshape((cxs_norms.size, 3))
