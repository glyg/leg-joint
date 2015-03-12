# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np
import hdfgraph
from ..objects import get_faces
from ..utils import _to_3d

import logging
log = logging.getLogger(__name__)


'''
The purpose of this small module is to efficiently compute
the geometrical and dynamical properties of the epithelium graph.

As those computations are vectorial calculus, we use pandas to perform them

'''



def update_graph(triangles, graph):

    for col in triangles.vertex_df.columns:
        data = triangles.vertex_df[col]
        try:
            graph.vp[col].fa = data
        except KeyError:
            log.debug('Vertex Property {} not found'.format(col))

    for col in triangles.edges_df.columns:
        data = triangles.edges_df[col]
        try:
            graph.ep[col].fa = data
        except KeyError:
            log.debug('Edge Property {} not found'.format(col))

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

    Methods
    -------



    '''

    def __init__(self, vertex_df, edges_df,
                 triangles, coords):
        '''
        Creates a container class for the triangles geometry

        Parameters
        ----------

        vertex_df:  :class:`pandas.DataFrame` table
          This data frame should the vertices data. It is indexed by the
          vertices indices in the graph. See `self.mandatory_vcols` for a
          list of columns of this dataframe
        edges_df:  :class:`pandas.DataFrame` table
          DataFrame with the edges data. It is indexed by a
          :class:`pandas.MultiIndex` object indexed by
          (source, target) pairs. For a list of columns,
          see `self.mandatory_ecols`
        triangles: ndarray
          trianges is a (N_t, 3) 2D array where each line contains
          a triple with the indices of the cell, the source (jv_i)
          and the target (jv_j) junction vertices.
        coords: list of strings
          the names of the three columns corresponding to the
          3D positions

        See Also
        --------

        hdfgraph.graph_to_dataframes: utility to convert a graph_tool.Graph
          to a dataframe pairs
        leg_joint.objects.get_faces: utility to obtain  the triangles list
          from a graph

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
        self._init_gradient()

    def copy(self):
        return Triangles(self.vertex_df.copy(), self.edges_df.copy(),
                         self.triangles.copy(), self.coords)

    def _init_gradient(self):

        self.grad_coords = ['g'+c for c in self.coords]
        self.grad_i = pd.DataFrame(0, index=self.uix_active,
                                   columns=self.grad_coords).sort_index()


    def _build_indices(self, faces):

        self.indices = {}
        names =('cell', 'jv_i', 'jv_j')
        letters = ('a', 'i', 'j')

        self.uix_active = self.vertex_df[
            self.vertex_df.is_active_vert==1].index

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

            unique = idx.unique()
            unique_name = 'uix_{}'.format(letter)
            setattr(self, unique_name, unique)

            view_name = 'udf_{}'.format(name)
            dv = DataView(self.vertex_df, unique)
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
            unique = idx.unique()
            setattr(self, unique_name, unique)
            view_name = 'udf_{}to{}'.format(*other_letters)
            dv = DataView(self.edges_df, unique)
            setattr(self, view_name, dv)

        self.uix_active_i = np.array(
            list(set(self.uix_active).intersection(self.uix_i)))
        self.uix_active_j = np.array(
            list(set(self.uix_active).intersection(self.uix_j)))


    @property
    def mandatory_vcols(self):
        ''' List of vertex data used in the computations
        '''
        cols = set(self.coords)
        cols.update({'rhos', 'heights'})
        topology = {'is_cell_vert',
                    'num_sides'}
        cols.update(topology)
        cell_geom = {'perimeters',
                     'areas',
                     'vols'}
        cols.update(cell_geom)

        dyn_parameters = ['contractilities',
                          'vol_elasticities']
        cols.update(dyn_parameters)
        return cols

    @property
    def mandatory_ecols(self):
        ''' List of edge data used in the computations
        '''
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

        cell_columns = ['rhos', 'heights', 'num_sides',
                        'areas', 'perimeters', 'vols']
        cell_columns.extend(self.coords)
        cell_data = self.udf_cell[cell_columns]

        ### update cell pos
        cell_data[self.coords] = (
            self.tdf_jv_i[self.coords].set_index(self.tix_aij).mean(level='cell')
            +self.tdf_jv_j[self.coords].set_index(self.tix_aij).mean(level='cell'))/2
        ### update rhos
        self.vertex_df['rhos'] = np.hypot(self.vertex_df[self.coords[0]],
                                          self.vertex_df[self.coords[1]])
        ### update lengths
        srcs = self.edges_df.index.get_level_values('source')
        trgts = self.edges_df.index.get_level_values('target')
        self.edges_df[self.dcoords] = self.vertex_df.loc[trgts, self.coords].values\
                                      - self.vertex_df.loc[srcs, self.coords].values
        self.edges_df['edge_lengths'] = np.linalg.norm(self.edges_df[self.dcoords], axis=1)
        self.faces['ell_ij'] = self.tdf_itoj['edge_lengths'].values

        ### This should be computed before hand
        cell_data['heights'] = cell_data['rhos'] - rho_lumen

        num_sides = self.tix_aij.get_level_values('cell').value_counts()
        cell_data['num_sides'] = num_sides.loc[self.uix_a]
        r_ak = self.tdf_atoi[self.dcoords].set_index(self.tix_aij)
        r_am = self.tdf_atoj[self.dcoords].set_index(self.tix_aij)

        crosses = pd.DataFrame(np.cross(r_ak, r_am), index=self.tix_a)

        sub_areas = np.linalg.norm(crosses, axis=1)/2
        self.faces['sub_areas'] = sub_areas
        normals = crosses / _to_3d(2 * sub_areas)
        self.faces[self.normal_coords] = normals.values

        cell_data['areas'] = self.faces.sub_areas.sum(level='cell').loc[self.uix_a]
        cell_data['perimeters'] = self.faces.ell_ij.sum(level='cell').loc[self.uix_a]
        ### We're neglecting curvature here
        cell_data['vols'] = cell_data['heights'] * cell_data['areas']

        self.udf_cell[cell_columns] = cell_data

    def scale(self, scaling_factor):

        '''Multiply all the distances by a factor `scaling_factor`

        Parameter
        =========
        scaling_factor: float
        '''
        self.vertex_df[self.coords] *= scaling_factor
        self.vertex_df['heights'] *= scaling_factor
        self.rho_lumen *= scaling_factor
        self.geometry()

    def update_polar(self):
        self.vertex_df['thetas'] = np.arctan2(self.vertex_df[self.coords[1]],
                                              self.vertex_df[self.coords[0]])
        self.vertex_df['rhos'] = np.hypot(self.vertex_df[self.coords[0]],
                                          self.vertex_df[self.coords[1]])

    def update_heights(self):
        self.vertex_df['heights'] = self.vertex_df['rhos'] - self.rho_lumen

    def update_cartesian(self):

        rhos = self.vertex_df['rhos']
        thetas = self.vertex_df['thetas']
        self.vertex_df[self.coords[0]] = rhos * np.cos(thetas)
        self.vertex_df[self.coords[1]] = rhos * np.sin(thetas)

    def rotate(self, angle, inplace=False):
        '''Rotates the epithelium by an angle `angle` around
        the :math:`z` axis
        '''

        self.update_polar()
        if inplace:
            self.vertex_df['thetas'] += angle
            self.update_cartesian()
        else:
            new = self.copy()
            new.vertex_df['thetas'] += angle
            new.update_cartesian()
            return new

    def periodic_boundary_condition(self):
        '''
        Applies the periodic boundary condition
        to the vertices positions along the sigma axis,
        with their curent value for rho.
        '''
        self.update_polar()
        buf_theta = self.vertex_df['thetas'] + np.pi
        buf_theta = (buf_theta % (2 * np.pi)) - np.pi
        self.vertex_df['thetas'] = buf_theta
        self.update_cartesian()

    def proj_sigma(self):
        ''' return an array of the positions projected on the
        cylinder with average rho radius
        '''
        self.update_polar()
        rho_mean = self.vertex_df.rhos.mean()
        sigmas = self.vertex_df.thetas * rho_mean
        return sigmas

    def translate(self, vector):
        raise NotImplementedError

    def closest_vert(self, position, condition=None):

        relative_pos = pd.DataFrame([self.vertex_df[coord] - u
                                     for coord, u in zip(self.coords, position)])
        relative_pos.set_index(self.vertex_df.index)
        if condition is not None:
            relative_pos = relative_pos[condition]

        dist = np.linalg.norm(relative_pos)
        return dist.argmin()

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
