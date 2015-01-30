# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function



import pandas as pd
import numpy as np
import hdfgraph
from leg_joint.objects import get_faces

#local_graph = None
eptm = None


def _to_3d(df):
    return df.repeat(3).reshape((df.size, 3))

def parse_graph(graph, coords):

    vertex_df, edges_df  = hdfgraph.graph_to_dataframes(graph)

    triangles = get_faces(graph)
    return vertex_df, edges_df, triangles

class Triangles:

    def __init__(self, vertex_df, edges_df, triangles, coords):

        self.vertex_df = vertex_df.copy()
        self.edges_df = edges_df.copy()

        self.coords = coords
        self.dcoords = ['d'+c for c in self.coords]
        self.normal_coords = ['u'+c for c in self.coords]

        self._build_indices(triangles)
        self.triangles = pd.DataFrame(index=self.tri_midx)

    def _build_indices(self, triangles):

        self.tri_midx = pd.MultiIndex.from_arrays(triangles.T,
                                                  names=('cell', 'jv_i', 'jv_j'))
        self.tri_cells = self.tri_midx.get_level_values('cell')
        self.tri_jvis = self.tri_midx.get_level_values('jv_i')
        self.tri_jvjs = self.tri_midx.get_level_values('jv_j')

        self.tri_ij = self.tri_midx.droplevel('cell')
        self.tri_ak = self.tri_midx.droplevel('jv_j')
        self.tri_am = self.tri_midx.droplevel('jv_i')

        self.cell_idxs = self.tri_cells.unique()
        self.jvi_idxs = self.tri_jvis.unique()
        self.jvj_idxs = self.tri_jvjs.unique()

        self.jv_idxs = np.array(set(self.jvi_idxs).union(self.jvj_idxs))
        self.je_idxs = self.tri_ij.unique()

    def _check_structure(self):

        ### mendatory columns
        vertex_cols = self.coords + ['is_cell_vert', 'areas', 'num_sides',
                                     'cell_heights', 'vols', 'rhos', 'perimeters',
                                     'contractilities', 'vol_elasticities']

        missing = set(vertex_cols).difference(self.vertex_df.columns)
        for col in missing:
            self.vertex_df[col] = 0
        edges_cols = self.dcoords + ['edge_length', 'line_tensions']

        missing = set(edges_cols).difference(self.edges_df.columns)
        for col in missing:
            self.edges_df[col] = 0

    def geometry(self, rho_lumen):

        srcs = self.edges_df.index.get_level_values('source')
        trgts = self.edges_df.index.get_level_values('target')

        self.edges_df[self.dcoords] = self.vertex_df.loc[trgts, self.coords].values\
                                      - self.vertex_df.loc[srcs, self.coords].values
        self.edges_df['edge_lengths'] = (self.edges_df[self.dcoords]**2).sum(axis=1)**0.5
        self.triangles['ell_ij'] = self.edges_df.loc[self.tri_ij, 'edge_lengths'].values

        cell_idxs = self.cell_idxs
        ### This should be done elsewhere
        self.vertex_df.loc[cell_idxs, 'cell_heights'] = self.vertex_df.loc[cell_idxs, 'rhos'] - rho_lumen

        num_sides = self.tri_midx.get_level_values('cell').value_counts()
        self.vertex_df.loc[cell_idxs, 'num_sides'] = num_sides.loc[cell_idxs]
        r_ak = self.edges_df[self.dcoords].loc[self.tri_ak].set_index(self.tri_midx)
        r_am = self.edges_df[self.dcoords].loc[self.tri_am].set_index(self.tri_midx)

        crosses = pd.DataFrame(np.cross(r_ak, r_am), index=self.tri_cells)

        sub_areas = ((crosses**2).sum(axis=1)**0.5) / 2
        self.triangles['sub_areas'] = sub_areas.values
        normals = 2 * crosses / _to_3d(sub_areas)
        normals.columns = self.normal_coords
        self.triangles.append(normals)

        self.vertex_df.loc[cell_idxs, 'areas'] = sub_areas.sum(level='cell').loc[cell_idxs]
        self.vertex_df.loc[cell_idxs, 'perimeters'] = self.triangles.ell_ij.sum(level='cell').loc[cell_idxs]
        self.vertex_df.loc[cell_idxs, 'vols'] = (self.vertex_df.cell_heights * self.vertex_df.areas).loc[cell_idxs]

    def gradient(self):

        self.grad_coords = ['g'+c for c in self.coords]
        grad_i = pd.DataFrame(0, index=self.jv_idxs,
                              columns=self.grad_coords)

        grad_i_lij = self.edges_df[self.dcoords] / _to_3d(self.edges_df.edge_lengths)
        grad_i_lij = grad_i_lij.loc[self.je_idxs]

        tensions = self.edges_df.loc[self.je_idxs, 'line_tensions']
        tensions.index.names = ('jv_i', 'jv_j')

        contract = self.vertex_df.loc[self.cell_idxs, 'contractilities']
        contract.index.name = 'cell'
        perimeters = self.vertex_df.loc[self.cell_idxs, 'perimeters']

        gamma_L = contract * perimeters
        gamma_L = gamma_L.loc[self.tri_cells]
        gamma_L.index = self.tri_midx

        area_term_alpha = gamma_L.groupby(level='jv_i').apply(
            lambda df: df.sum(level='jv_j'))
        area_term_beta = gamma_L.groupby(level='jv_j').apply(
            lambda df: df.sum(level='jv_i')).swaplevel(0,1).sortlevel()

        big_term_a = grad_i_lij * _to_3d(tensions + area_term_alpha)
        big_term_b = grad_i_lij * _to_3d(tensions + area_term_beta)

        grad_i.loc[self.jvi_idxs] += big_term_a.sum(level='jv_i').loc[self.jvi_idxs].values
        grad_i.loc[self.jvj_idxs] -= big_term_b.sum(level='jv_j').loc[self.jvj_idxs].values
