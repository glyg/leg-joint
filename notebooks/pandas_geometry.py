    # -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function



import pandas as pd
import numpy as np
import hdfgraph
from leg_joint.objects import get_faces
import logging
log = logging.getLogger(__name__)



'''
The purpose of this small module is to efficiently compute
the geometrical an dynamical properties of the epithelium graph.

As those computations are vectorial calculus, we use pandas to perfor them

'''

class Unique_index:
    ''' constructor class to retrieve unique indices from the faces'''
    def __init__(self, idx):
        self._full_idx = idx
        self._idx = None

    @property
    def idx(self):
        '''Unique indices'''
        return self._full_idx.unique()

class DFSelector:
    ''' constructor class to downcast and upcast
    from the 3 DataFrame'''
    def __init__(self, df, idx, *columns):

        self.idx
        self._idx = None

    @property
    def idx(self):
        '''Unique indices'''
        return self._full_idx.unique()


def _to_3d(df):
    return df.repeat(3).reshape((df.size, 3))

def parse_graph(graph):

    vertex_df, edges_df  = hdfgraph.graph_to_dataframes(graph)
    triangles = get_faces(graph)
    return vertex_df, edges_df, triangles

class Triangles:
    '''
    Data structure defined to index the ensembles of sub-graph homologue to
    this topology:

     jv_i (i) -- > (j) jv_j
           ^       ^
            \     /
             \   /
              (a) cell

    In the 3D coordinate system, this represents an oriented
    triangular face.  Note that as long as the cell to junction edges
    go outward from the cell, the complete epithelium is a unique
    set of faces homologue to this topology (regardess of the
    orientation of jv_i and jv_j) thus the triangle is the unit cell
    of the geometrical network - in the cristalography sense of unit
    cell.

    '''

    def __init__(self, vertex_df, edges_df, triangles, coords):
        '''
        Creates a container class for the triangles geometry

        Parameters:
        -----------
        vertex_df:  :class:`pandas.DataFrame` class

        '''

        self.vertex_df = vertex_df.copy()
        self.edges_df = edges_df.copy()

        self.coords = coords
        self.dcoords = ['d'+c for c in self.coords]
        self.normal_coords = ['u'+c for c in self.coords]

        self._build_indices(triangles)
        self.faces = pd.DataFrame(index=self.tri_midx)
        self._complete_df_cols()


    def _build_indices(self, faces):

        self.indices = {}
        names =('cell', 'jv_i', 'jv_j')
        letters = ('a', 'i', 'j')

        ### MultiIndex named (cell, jv_i, jv_j) for each triangle
        ### Those indices must be coherent with the original graph
        self.tri_midx = pd.MultiIndex.from_arrays(faces.T,
                                                  names=names)
        self.indices[tuple(names)] = self.tri_midx
        for letter, name  in zip(letters, names):

            ### single level index on current
            ### vertex type (contains repeated values)
            idx_name = 'tri_{}'.format(letter)
            idx = self.tri_midx.get_level_values(name)
            setattr(self, idx_name, idx)
            self.indices[name] = idx

            unique = Unique_index(idx)
            unique_name = 'uix_{}'.format(letter)
            setattr(self, unique_name, unique.idx)

            ### 2 level MultiIndex on the oriented edge opposed
            ### to the current vertex
            other_letters = list(letters)
            other_letters.remove(letter)
            other_names = list(names)
            other_names.remove(name)

            idx_name = 'tri_{}{}'.format(*other_letters)
            idx = self.tri_midx.droplevel(name)
            setattr(self, idx_name, idx)

            self.indices[tuple(other_names)] = idx
            unique_name = 'uix_{}{}'.format(*other_letters)
            unique = Unique_index(idx)
            setattr(self, unique_name, unique.idx)

    @property
    def mandatory_vcols(self):

        cols = set(self.coords)
        topology = {'is_cell_vert',
                    'num_sides'}
        cols.update(topology)
        cell_geom = {'heights',
                     'perimeters',
                     'areas',
                     'vols'}
        cols.update(cell_geom)

        dyn_parameters = ['contractilities',
                          'vol_elasticities']
        cols.update(dyn_parameters)
        return cols

    @property
    def mandatory_ecols(self):
        cols = set(self.dcoords)
        cols.update({'edge_length',
                     'line_tensions'})
        return cols


    def _complete_df_cols(self):

        ### mendatory columns
        missing = self.mandatory_vcols.difference(self.vertex_df.columns)
        for col in missing:
            log.debug('appending null column {}'.format(col),
                      'to vertex_df')
            self.vertex_df[col] = 0
        missing = self.mandatory_ecols.difference(self.edges_df.columns)
        for col in missing:
            log.debug('appending null column {}'.format(col),
                      'to edges_df')
            self.edges_df[col] = 0

    def geometry(self, rho_lumen):

        srcs = self.edges_df.index.get_level_values('source')
        trgts = self.edges_df.index.get_level_values('target')

        self.edges_df[self.dcoords] = self.vertex_df.loc[trgts, self.coords].values\
                                      - self.vertex_df.loc[srcs, self.coords].values
        self.edges_df['edge_lengths'] = (self.edges_df[self.dcoords]**2).sum(axis=1)**0.5
        self.faces['ell_ij'] = self.edges_df.loc[self.tri_ij, 'edge_lengths'].values

        cell_idxs = self.cell_idxs
        ### This should be done elsewhere
        self.vertex_df.loc[cell_idxs, 'cell_heights'] = self.vertex_df.loc[cell_idxs, 'rhos'] - rho_lumen

        num_sides = self.tri_midx.get_level_values('cell').value_counts()
        self.vertex_df.loc[cell_idxs, 'num_sides'] = num_sides.loc[cell_idxs]
        r_ak = self.edges_df[self.dcoords].loc[self.tri_ak].set_index(self.tri_midx)
        r_am = self.edges_df[self.dcoords].loc[self.tri_am].set_index(self.tri_midx)

        crosses = pd.DataFrame(np.cross(r_ak, r_am), index=self.tri_cells)

        sub_areas = ((crosses**2).sum(axis=1)**0.5) / 2
        self.faces['sub_areas'] = sub_areas.values
        normals = 2 * crosses / _to_3d(sub_areas)
        normals.columns = self.normal_coords
        self.faces.append(normals)

        self.vertex_df.loc[cell_idxs, 'areas'] = sub_areas.sum(level='cell').loc[cell_idxs]
        self.vertex_df.loc[cell_idxs, 'perimeters'] = self.faces.ell_ij.sum(level='cell').loc[cell_idxs]
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
