# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np
import hdfgraph
import graph_tool.all as gt

from ..epithelium.generation import (vertex_data, cell_data,
                                     edge_data, junction_data,
                                     face_data)
from ..utils import _to_3d

import logging
log = logging.getLogger(__name__)


'''
The purpose of this small module is to efficiently compute
the geometrical and dynamical properties of the epithelium graph.

As those computations are vectorial calculus, we use pandas to perform them

'''

def get_faces(graph, as_array=True):
    '''
    Retrieves all the triangular subgraphs of the form

       1 -- > 2
        ^   ^
         \ /
          0

    In our context, vertex 0 always corresponds to a cell
    and vertices 1 and 2 to junction vertices

    Parameters
    ----------

    graph : a :class:`GraphTool` graph instance
    as_array: bool, optional, default `True`
      if `True`, the output of `subraph_isomorphism` is converted
      to a (N, 3) ndarray.

    Returns
    -------

    triangles:  list of gt.PropertyMaps or (N, 3) ndarray
      each line corresponds to a triplet (cell, jv0, jv1)
      where cell, jv0 and jv1 are indices of the input graph
      if
    '''
    tri_graph = gt.Graph()
    ## the vertices
    verts = tri_graph.add_vertex(3)
    ## edges
    tri_graph.add_edge_list([(0, 1), (0, 2), (1, 2)])
    _triangles = gt.subgraph_isomorphism(tri_graph, graph)
    if not as_array:
        return tri_graph, _triangles
    triangles = np.array([tri.a for tri in _triangles], dtype=np.int)
    return triangles


class Mesh:
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

    def __init__(self, graph, rho_lumen):
        '''
        Creates a container class for the triangles geometry

        Parameters
        ----------

        triangles: ndarray
          trianges is a (N_t, 3) 2D array where each line contains
          a triple with the indices of the cell, the source (jv_i)
          and the target (jv_j) junction vertices.
        graph: the underlying graph

        See Also
        --------

        hdfgraph.graph_to_dataframes: utility to convert a graph_tool.Graph
          to a dataframe pairs
        leg_joint.topology.get_faces: utility to obtain  the triangles list
          from a graph

        '''
        self.triangles = get_faces(graph)
        self.graph = graph
        self.coords = ['x', 'y', 'z']
        self.dcoords = ['d'+c for c in self.coords]
        self.normal_coords = ['u'+c for c in self.coords]
        self.rho_lumen = rho_lumen
        self._build_graphviews()
        self._build_faceviews()
        self.faces = pd.DataFrame(index=self.tix_aij,
                                  columns=face_data.keys())

        for key, tup in face_data.items():
            self.faces[key] = np.ones(self.triangles.shape[0],
                                      dtype=tup[1]) * tup[0]

    def copy(self):
        return Mesh(self.graph.copy())

    def reset(self):
        self._build_graphviews()
        self.triangles = get_faces(self.graph)
        self._build_indices()

    def _build_graphviews(self):

        live_cells = self.graph.vp['is_cell_vert'].copy()
        live_cells.a = self.graph.vp['is_cell_vert'].a * self.graph.vp['is_alive'].a
        self.cell_graph = gt.GraphView(self.graph,
                                        vfilt=live_cells)

        for key, tup in cell_data.items():
            self.cell_graph.vp[key] = self.cell_graph.new_vertex_property(tup[1])
            self.cell_graph.vp[key].a = tup[0]

        self.active_graph = gt.GraphView(self.graph,
                                         vfilt=self.graph.vp['is_active_vert'])

        self.junction_graph = gt.GraphView(self.graph,
                                           efilt=self.graph.ep['is_junction_edge'])
        for key, tup in junction_data.items():
            self.junction_graph.ep[key] = self.junction_graph.new_edge_property(tup[1])
            self.junction_graph.ep[key].a = tup[0]

    def _build_faceviews(self):

        names = ['cell', 'jv_i', 'jv_j']
        letters = ['a', 'i', 'j']

        ### MultiIndex named (cell, jv_i, jv_j) for each triangle
        ### Those indices must be coherent with the original graph
        self.tix_aij = pd.MultiIndex.from_arrays(self.triangles.T,
                                                 names=names)
        for letter, name  in zip(letters, names):
            ### single level index on current
            ### vertex type (contains repeated values)
            idx_name = 'tix_{}'.format(letter)
            idx = self.tix_aij.get_level_values(name)
            setattr(self, idx_name, idx)

            unique = idx.unique()
            unique_name = 'uix_{}'.format(letter)
            setattr(self, unique_name, unique)

            view_name = 'fv_{}'.format(letter)
            setattr(self, view_name,
                    VertexFacesView(self.graph, idx))

            ### 2 level MultiIndex on the oriented edge opposed
            ### to the current vertex
            other_letters = list(letters)
            other_letters.remove(letter)
            other_names = list(names)
            other_names.remove(name)

            idx_name = 'tix_{}{}'.format(*other_letters)
            idx = self.tix_aij.droplevel(name)
            setattr(self, idx_name, idx)

            unique_name = 'uix_{}{}'.format(*other_letters)
            unique = idx.unique()
            setattr(self, unique_name, unique)

            view_name = 'fv_{}to{}'.format(*other_letters)
            setattr(self, view_name,
                    EdgeFacesView(self.graph, idx))

        # self.uix_active_i = np.array(
        #     list(set(self.uix_active).intersection(self.uix_i)))
        # self.uix_active_j = np.array(
        #     list(set(self.uix_active).intersection(self.uix_j)))

    def update_geometry(self):

        ### update rho
        self.update_polar()
        self.update_height()
        self.update_length()
        # self.edge_df['edge_length'] = np.linalg.norm(
        #     self.edge_df[self.dcoords], axis=1)
        # self.faces['ell_ij'] = self.tdf_itoj['edge_length'].values

        cell_columns = ['rho', 'height', 'num_sides',
                        'area', 'perimeter', 'vol']
        cell_columns.extend(self.coords)
        cell_pos = (self.fv_i[self.coords].set_index(self.tix_aij).mean(level='cell')
                    + self.fv_j[self.coords].set_index(self.tix_aij).mean(level='cell'))/2


        ### update cell pos
        for coord in self.coords:
            self.cell_graph.vp[coord].fa = cell_pos[coord].loc[self.uix_a]

        r_ak = self.fv_atoi[self.dcoords].set_index(self.tix_aij)
        r_am = self.fv_atoj[self.dcoords].set_index(self.tix_aij)

        crosses = pd.DataFrame(np.cross(r_ak, r_am), index=self.tix_a)

        sub_area = np.linalg.norm(crosses, axis=1) / 2
        self.faces['sub_area'] = sub_area
        normals = crosses / _to_3d(2 * sub_area)
        self.faces[self.normal_coords] = normals.values

        self.cell_graph['area'].fa = self.faces.sub_area.sum(
            level='cell').loc[self.uix_a]
        self.cell_graph['perimeter'].fa = self.faces.ell_ij.sum(
            level='cell').loc[self.uix_a]
        ### We're neglecting curvature here
        self.cell_graph['vol'].fa = self.cell_graph['height'].fa * self.cell_graph['area'].fa

    def update_length(self):
        for coord in self.coords:
            self.graph.ep['d'+coord].fa = (
                gt.edge_endpoint_property(self.graph,
                                          self.graph.vp[coord], 'target').fa
                - gt.edge_endpoint_property(self.graph,
                                            self.graph.vp[coord], 'source').fa)
        self.graph.ep['edge_length'].fa =  np.sqrt((self.graph.ep['dx'].fa
                                                   + self.graph.ep['dy'].fa
                                                   + self.graph.ep['dz'].fa)**2)

    def update_num_sides(self):
        num_sides = self.tix_aij.get_level_values('cell').value_counts()
        self.cell_graph['num_sides'].fa = num_sides.loc[self.uix_a]

    def set_new_pos(self, pos):
        ndim = len(self.coords)
        _pos = pos.reshape((pos.size//ndim, ndim))
        for i, coord in enumerate(self.coords):
            self.active_graph.vp[coord].fa = _pos[:, i]

    def scale(self, scaling_factor):
        '''Multiply all the distances by a factor `scaling_factor`

        Parameter
        =========
        scaling_factor: float
        '''
        for coords in self.coords:
            self.graph.vp[self.coords].fa *= scaling_factor
        self.rho_lumen *= scaling_factor
        self.update_geometry()
        self.update_pmaps()

    def update_polar(self):
        self.graph.vp['theta'].fa = np.arctan2(self.graph.vp[self.coords[1]].fa,
                                               self.graph.vp[self.coords[0]].fa)
        self.graph.vp['rho'].fa = np.hypot(self.graph.vp[self.coords[0]].fa,
                                           self.graph.vp[self.coords[1]].fa)

    def update_height(self):
        self.graph.vp['height'].fa = self.graph.vp['rho'].fa - self.rho_lumen

    def update_cartesian(self):

        rho = self.graph.vp['rho'].fa
        theta = self.graph.vp['theta'].fa
        self.graph.vp[self.coords[0]].fa = rho * np.cos(theta)
        self.graph.vp[self.coords[1]].fa = rho * np.sin(theta)

    def rotate(self, angle):
        '''Rotates the epithelium by an angle `angle` around
        the :math:`z` axis
        '''
        self.update_polar()
        self.graph.vp['theta'].fa += angle
        self.update_cartesian()

    def periodic_boundary_condition(self):
        '''
        Applies the periodic boundary condition
        to the vertices positions along the sigma axis,
        with their curent value for rho.
        '''
        self.update_polar()
        buf_theta = self.graph.vp['theta'].fa + np.pi
        buf_theta = (buf_theta % (2 * np.pi)) - np.pi
        self.graph.vp['theta'].fa = buf_theta
        self.update_cartesian()

    def proj_sigma(self):
        ''' return an array of the positions projected on the
        cylinder with average rho radius
        '''
        self.update_polar()
        rho_mean = self.graph.vp['rho'].fa.mean()
        sigmas = self.graph.new_vertex_property('float')
        sigmas.fa = self.graph.vp['theta'].fa * rho_mean
        return sigmas

    def translate(self, vector):
        raise NotImplementedError

    def closest_vert(self, position, condition=None):


        relative_pos = np.array([self.graph.vp[coord].fa - u
                                 for coord, u
                                 in zip(self.coords, position)])
        if condition is not None:
            relative_pos = relative_pos[condition]

        dist = np.linalg.norm(relative_pos)
        vi = self.graph.vertex_index.copy()

        return vi.fa[dist.argmin()]


class VertexFacesView:
    '''constructor class to get and set
    data on the columns of a dataframe with repeated values
    for each vertex of the graph'''
    def __init__(self, graph, faces_idx):
        self.graph = graph # Can be a GraphView
        self._idx = faces_idx

    def __getitem__(self, prop_name):
        if isinstance(prop_name, list):
            props = np.array([self.graph.vp[prop_n].a[self._idx]
                              for prop_n in prop_name]).T
            return pd.DataFrame(data=props, index=self._idx,
                                columns=prop_name)
        else:
            prop = self.graph.vp[prop_name]
            return pd.Series(prop.a[self._idx],
                             index=self._idx, name=prop_name)

    def __setitem__(self, prop_name, data):
        if isinstance(prop_name, list):
            for prop_n, col in zip(prop_name, data):
                self.graph.vp[prop_n].fa = col
        else:
            self.graph.vp[prop_name].fa = data

class EdgeFacesView:
    '''constructor class to get and set
    data on the columns of a dataframe with repeated values
    for each vertex of the graph'''
    def __init__(self, graph, faces_idx):

        self.graph = graph # Can be a GraphView
        self.faces_idx = faces_idx
        self._idx = [graph.edge_index[graph.edge(s, t)]
                     for s, t in faces_idx]

    def __getitem__(self, prop_name):
        if isinstance(prop_name, list):
            props = np.array([self.graph.ep[prop_n].a[self._idx]
                              for prop_n in prop_name]).T
            return pd.DataFrame(data=props, index=self._idx,
                                columns=prop_name)
        else:
            prop = self.graph.ep[prop_name]
            return pd.Series(prop.a[self._idx],
                             index=self.faces_idx,
                             name=prop_name)

    def __setitem__(self, prop_name, data):
        if isinstance(prop_name, list):
            for prop_n, col in zip(prop_name, data):
                self.graph.ep[prop_n].fa = col
        else:
            self.graph.ep[prop_name].fa = data