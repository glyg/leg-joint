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

class DataView:
    '''constructor class to get and set
    columns on **views** of a subset of a dataframe '''
    def __init__(self, df, ix):
        self._data = df
        self._ix = ix

    @property
    def values(self):
        return self._data.loc[self._ix]

    def __getitem__(self, key):
        return self._data.loc[self._ix, key]

    def __setitem__(self, key, data):
        self._data.loc[self._ix, key] = data



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
        self.faces = pd.DataFrame(index=self.tix_aij)
        self._complete_df_cols()
        self._ucells = None



    def _build_indices(self, faces):

        self.indices = {}
        names =('cell', 'jv_i', 'jv_j')
        letters = ('a', 'i', 'j')

        ### MultiIndex named (cell, jv_i, jv_j) for each triangle
        ### Those indices must be coherent with the original graph
        self.tix_aij = pd.MultiIndex.from_arrays(faces.T,
                                                  names=names)
        self.indices[tuple(names)] = self.tix_aij
        for letter, name  in zip(letters, names):

            ### single level index on current
            ### vertex type (contains repeated values)
            idx_name = 'tix_{}'.format(letter)
            idx = self.tix_aij.get_level_values(name)
            setattr(self, idx_name, idx)
            self.indices[name] = idx

            view_name = 'tdf_{}'.format(name)
            dv = DataView(self.vertex_df, idx)
            setattr(self, view_name, dv)

            unique = Unique_index(idx)
            unique_name = 'uix_{}'.format(letter)
            setattr(self, unique_name, unique.idx)

            view_name = 'udf_{}'.format(name)
            dv = DataView(self.vertex_df, unique.idx)
            setattr(self, view_name, dv)

            ### 2 level MultiIndex on the oriented edge opposed
            ### to the current vertex
            other_letters = list(letters)
            other_letters.remove(letter)
            other_names = list(names)
            other_names.remove(name)

            idx_name = 'tix_{}{}'.format(*other_letters)
            idx = self.tix_aij.droplevel(name)
            setattr(self, idx_name, idx)
            view_name = 'tdf_{}to{}'.format(*other_letters)
            dv = DataView(self.edges_df, idx)
            setattr(self, view_name, dv)

            self.indices[tuple(other_names)] = idx
            unique_name = 'uix_{}{}'.format(*other_letters)
            unique = Unique_index(idx)
            setattr(self, unique_name, unique.idx)
            view_name = 'udf_{}to{}'.format(*other_letters)
            dv = DataView(self.edges_df, unique.idx)
            setattr(self, view_name, dv)

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

        for c in self.normal_coords:
            self.faces[c] = 0



    def geometry(self, rho_lumen):

        srcs = self.edges_df.index.get_level_values('source')
        trgts = self.edges_df.index.get_level_values('target')

        self.edges_df[self.dcoords] = self.vertex_df.loc[trgts, self.coords].values\
                                      - self.vertex_df.loc[srcs, self.coords].values
        self.edges_df['edge_lengths'] = (self.edges_df[self.dcoords]**2).sum(axis=1)**0.5
        self.faces['ell_ij'] = self.tdf_itoj['edge_lengths'].values

        ### This must be computed before hand
        self.udf_cell['cell_heights'] = self.udf_cell['rhos'] - rho_lumen

        num_sides = self.tix_aij.get_level_values('cell').value_counts()
        self.udf_cell['num_sides'] = num_sides.loc[self.uix_a]
        r_ak = self.tdf_atoi[self.dcoords].set_index(self.tix_aij)
        r_am = self.tdf_atoj[self.dcoords].set_index(self.tix_aij)

        crosses = pd.DataFrame(np.cross(r_ak, r_am), index=self.tix_a)

        sub_areas = ((crosses**2).sum(axis=1)**0.5) / 2
        self.faces['sub_areas'] = sub_areas.values
        normals = 2 * crosses / _to_3d(sub_areas)
        self.faces[self.normal_coords] = normals.values

        self.udf_cell['areas'] = sub_areas.sum(level='cell').loc[self.uix_a]
        self.udf_cell['perimeters'] = self.faces.ell_ij.sum(level='cell').loc[self.uix_a]
        ### We're neglecting curvature here
        self.udf_cell['vols'] = self.udf_cell['cell_heights'] * self.udf_cell['areas']

    def energy(self):

        self.vertex_df['E_v'] = 0
        self.udf_cell['E_v'] = 0.5 * (self.udf_cell['vol_elasticities']
                                      * (self.udf_cell['vols']
                                         - self.udf_cell['prefered_vol'])**2)
        self.vertex_df['E_c'] = 0
        self.udf_cell['E_c'] = 0.5 * (self.udf_cell['contractilities']
                                      * self.udf_cell['perimeters']**2)
        self.edges_df['E_t'] = 0
        E_t = self.udf_itoj['line_tensions'] * self.udf_itoj['edge_lengths']
        self.udf_itoj['E_t'] = E_t.values

    def gradient(self, components=False):
        '''
        If components is tTrue, returns the individual terms
        (grad_t, grad_c, grad_v)
        '''
        self.grad_coords = ['g'+c for c in self.coords]
        uix_jv = set(self.uix_i).union(self.uix_j)
        grad_i = pd.DataFrame(0, index=uix_jv,
                              columns=self.grad_coords).sort_index()
        grad_i_lij = - self.udf_itoj[self.dcoords] / _to_3d(self.udf_itoj['edge_lengths'])
        grad_i_lij.index = pd.MultiIndex.from_tuples(self.uix_ij, names=('jv_i', 'jv_j'))

        grad_t = self.tension_grad(grad_i, grad_i_lij)
        grad_c = self.contractile_grad(grad_i, grad_i_lij)
        grad_v = self.volume_grad(grad_i, grad_i_lij)

        grad_i = grad_t + grad_c + grad_v
        if components:
            return grad_i, grad_t, grad_c, grad_v
        return grad_i


    def tension_grad(self, grad_i, grad_i_lij):

        grad_t = grad_i.copy()
        grad_t[:] = 0

        tensions = self.udf_itoj['line_tensions']
        tensions.index.names = ('jv_i', 'jv_j')

        _grad_t = grad_i_lij * _to_3d(tensions)
        grad_t.loc[self.uix_i] += _grad_t.sum(level='jv_i').loc[self.uix_i].values
        grad_t.loc[self.uix_j] += _grad_t.sum(level='jv_j').loc[self.uix_j].values
        return grad_t

    def contractile_grad(self, grad_i, grad_i_lij):

        grad_c = grad_i.copy()
        grad_c[:] = 0

        contract = self.udf_cell['contractilities']
        contract.index.name = 'cell'
        perimeters = self.udf_cell['perimeters']

        gamma_L = contract * perimeters
        gamma_L = gamma_L.loc[self.tix_a]
        gamma_L.index = self.tix_aij

        area_term = gamma_L.groupby(level='jv_i').apply(
            lambda df: df.sum(level='jv_j'))

        _grad_c = grad_i_lij.loc[self.uix_ij] * _to_3d(area_term.loc[self.uix_ij])
        grad_c.loc[self.uix_i] += _grad_c.sum(level='jv_i').loc[self.uix_i].values
        grad_c.loc[self.uix_j] += _grad_c.sum(level='jv_j').loc[self.uix_j].values

        return grad_c

    def volume_grad(self, grad_i, grad_i_lij):

        grad_v = grad_i.copy()
        grad_v[:] = 0

        elasticity = self.udf_cell['vol_elasticities']
        pref_V = self.udf_cell['prefered_vol']
        V = self.udf_cell['vols']
        KV_V0 = elasticity * (V - pref_V)
        tri_KV_V0 = KV_V0.loc[self.tix_a]
        tri_KV_V0.index = self.tix_aij

        r_ijs = self.tdf_itoj[self.dcoords]
        cross_ur = pd.DataFrame(np.cross(self.faces[self.normal_coords], r_ijs),
                                index=self.tix_aij, columns=self.coords)

        h_nu = self.udf_cell['cell_heights'] / (2 * self.udf_cell['num_sides'])
        grad_i_V_cell = cross_ur.sum(level='cell') * _to_3d(KV_V0 * h_nu)

        cell_term = grad_i_V_cell.loc[self.tix_a].set_index(self.tix_aij)
        cell_term.columns = self.coords

        _r_to_rho_i = self.udf_jv_i[self.coords] / _to_3d(2 * self.udf_jv_i['rhos'])
        _r_to_rho_j = self.udf_jv_j[self.coords] / _to_3d(2 * self.udf_jv_j['rhos'])
        r_to_rho_i = _r_to_rho_i.loc[self.tix_i].set_index(self.tix_aij)
        r_to_rho_j = _r_to_rho_j.loc[self.tix_j].set_index(self.tix_aij)

        r_ai = self.tdf_atoi[self.dcoords]
        r_aj = self.tdf_atoj[self.dcoords]
        normals = self.faces[self.normal_coords]
        cross_ai = pd.DataFrame(np.cross(normals, r_ai),
                                index=self.tix_aij, columns=self.coords)
        cross_aj = pd.DataFrame(np.cross(normals, r_aj),
                                index=self.tix_aij, columns=self.coords)

        tri_heights = self.tdf_cell['cell_heights']
        tri_heights.index = self.tix_aij
        sub_areas = self.faces['sub_areas']

        radial_term_a = _to_3d(tri_KV_V0 * sub_areas) * r_to_rho_i - _to_3d(tri_heights / 2) * cross_aj
        radial_term_b = _to_3d(tri_KV_V0 * sub_areas) * r_to_rho_j + _to_3d(tri_heights / 2) * cross_ai

        ij_term = radial_term_a.groupby(level='jv_i').apply(lambda df: df.sum(level='jv_j'))
        jk_term = radial_term_b.groupby(level='jv_j').apply(lambda df: df.sum(level='jv_i')).swaplevel(0, 1)

        _cell_term = cell_term.groupby(level='jv_i').apply(lambda df: df.sum(level='jv_j'))
        print(_cell_term.loc[self.uix_ij].head())

        print(jk_term.loc[self.uix_ij].head())
        alpha_term = (_cell_term.loc[self.uix_ij] + ij_term.loc[self.uix_ij].values)
        beta_term = (_cell_term.loc[self.uix_ij] + jk_term.loc[self.uix_ij].values)
        grad_v.loc[self.uix_i] += alpha_term.sum(level='jv_i').loc[self.uix_i].values
        grad_v.loc[self.uix_j] += beta_term.sum(level='jv_j').loc[self.uix_j].values
        return grad_v
