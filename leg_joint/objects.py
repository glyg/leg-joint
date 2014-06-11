# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

"""




"""

import pyximport
pyximport.install()


import os
import numpy as np
import scipy as sp

import graph_tool.all as gt
#from scipy import weave, spatial
from .filters import EpitheliumFilters
from .utils import to_xy, to_rhotheta
from .circumcircle import c_circumcircle
from sklearn.decomposition import PCA

import logging
log = logging.getLogger(__name__)


CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = os.path.join(ROOT_DIR, 'default', 'params.xml')
tau = 2 * np.pi



class AbstractRTZGraph(object):
    '''
    Wrapper of a (`graph_tool`)[http://projects.skewed.de/graph-tool]
    for a geometric graph in a 3D coordinate system

    Properties
    ===========
    ixs, wys, rhos, thetas, zeds, sigmas: vertex PropertyMaps
        of the vertices positions along the different axes.

    dixs, dwys, dzeds, drhos, dthetas, dsigmas : edge PropertyMaps of the
        edge components along the various axis. For exemple:
        `self.dixs[edge] = self.ixs[edge.source()] - self.ixs[edge.target()]`

    edge_lengths: edge PropertyMap giving the edge lengths

    u_dixs, u_dwys, ...: Coordinates of the unitary vector colinear to an edge


    Methods:
    ===========

    scale()

    '''

    def __init__(self):
        '''
        Create an `AbstractRTZGraph` object. This is not ment as
        a stand alone object, but should rather be sublcassed.
        '''
        self.graph.set_directed(True) #So each edge is a defined vector
        self.current_angle = 0
        if self.new :
            self._init_edge_bool_props()
            self._init_vertex_geometry()
            self._init_edge_geometry()
        else:
            self._get_edge_bool_props()
            self._get_vertex_geometry()
            self._get_edge_geometry()

        ## Properties that are not internalized
        ### edge properties from vertex properties
        self.edge_src_rhos = self.graph.new_edge_property('float')
        self.edge_trgt_rhos = self.graph.new_edge_property('float')
        self.rho_lumen = self.params['rho_lumen']

    def _init_edge_bool_props(self):
        '''
        Creates the edge boolean PropertyMaps
        '''
        self.at_boundary = self.graph.new_edge_property('bool')
        self.at_boundary.a[:] = 0
        self.graph.edge_properties["at_boundary"] = self.at_boundary
        self.is_new_edge = self.graph.new_edge_property('bool')
        self.is_new_edge.a[:] = 1
        self.graph.edge_properties["is_new_edge"] = self.is_new_edge

    def _get_edge_bool_props(self):
        self.at_boundary = self.graph.edge_properties["at_boundary"]
        self.is_new_edge = self.graph.edge_properties["is_new_edge"]

    def _init_vertex_geometry(self):
        '''Creates the vertices geometric property maps
        '''
        # Position in the rho theta zed space
        self.rhos = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["rhos"] = self.rhos
        self.thetas = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["thetas"] = self.thetas
        self.zeds = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["zeds"] = self.zeds
        self.sigmas = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["sigmas"] = self.sigmas
        self.ixs = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["ixs"] = self.ixs
        self.wys = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["wys"] = self.wys

    def _get_vertex_geometry(self):
        '''Creates attributes from the vertices geometric property maps
        '''
        # Position in the rho theta zed space
        self.rhos = self.graph.vertex_properties["rhos"]
        self.thetas = self.graph.vertex_properties["thetas"]
        self.zeds = self.graph.vertex_properties["zeds"]
        self.sigmas = self.graph.vertex_properties["sigmas"]
        self.ixs = self.graph.vertex_properties["ixs"]
        self.wys = self.graph.vertex_properties["wys"]

    def _init_edge_geometry(self):
        '''Creates the edge geometric property maps
        '''
        # deltas
        self.dthetas = self.graph.new_edge_property('float')
        self.graph.edge_properties["dthetas"] = self.dthetas
        self.dsigmas = self.graph.new_edge_property('float')
        self.graph.edge_properties["dsigmas"] = self.dsigmas
        self.dzeds = self.graph.new_edge_property('float')
        self.graph.edge_properties["dzeds"] = self.dzeds
        self.drhos = self.graph.new_edge_property('float')
        self.graph.edge_properties["drhos"] = self.drhos
        self.dixs = self.graph.new_edge_property('float')
        self.graph.edge_properties["dixs"] = self.dixs
        self.dwys = self.graph.new_edge_property('float')
        self.graph.edge_properties["dwys"] = self.dwys

        # Edge lengths
        self.edge_lengths = self.graph.new_edge_property('float')
        self.graph.edge_properties["edge_lengths"] = self.edge_lengths

        # unitary vectors
        self.u_dsigmas = self.graph.new_edge_property('float')
        self.graph.edge_properties["u_dsigmas"] = self.u_dsigmas
        self.u_dzeds = self.graph.new_edge_property('float')
        self.graph.edge_properties["u_dzeds"] = self.u_dzeds
        self.u_drhos = self.graph.new_edge_property('float')
        self.graph.edge_properties["u_drhos"] = self.u_drhos
        self.u_dixs = self.graph.new_edge_property('float')
        self.graph.edge_properties["u_dixs"] = self.u_dixs
        self.u_dwys = self.graph.new_edge_property('float')
        self.graph.edge_properties["u_dwys"] = self.u_dwys

    def _get_edge_geometry(self):
        '''
        Creates attributes from  the  geometric property maps
        '''
        # deltas
        self.dthetas = self.graph.edge_properties["dthetas"]
        self.dsigmas = self.graph.edge_properties["dsigmas"]
        self.dzeds = self.graph.edge_properties["dzeds"]
        self.drhos = self.graph.edge_properties["drhos"]
        self.dixs = self.graph.edge_properties["dixs"]
        self.dwys = self.graph.edge_properties["dwys"]

        # Edge lengths
        self.edge_lengths = self.graph.edge_properties["edge_lengths"]
        # unitary vectors
        self.u_dsigmas = self.graph.edge_properties["u_dsigmas"]
        self.u_dzeds = self.graph.edge_properties["u_dzeds"]
        self.u_drhos = self.graph.edge_properties["u_drhos"]
        self.u_dixs = self.graph.edge_properties["u_dixs"]
        self.u_dwys = self.graph.edge_properties["u_dwys"]

    def scale(self, scaling_factor):
        '''Multiply all the distances by a factor `scaling_factor`

        Parameter
        =========
        scaling_factor: float
        '''
        self.ixs.a *= scaling_factor
        self.wys.a *= scaling_factor
        self.zeds.a *= scaling_factor
        self.rho_lumen *= scaling_factor
        self.update_rhotheta()

    def rotate(self, angle):
        '''Rotates the epithelium by an angle `angle` around
        the :math:`z` axis
        '''
        self.update_rhotheta()
        buf_theta = self.thetas.a + np.pi
        buf_theta += angle
        buf_theta = (buf_theta % (2 * np.pi)) - np.pi
        self.thetas.a = buf_theta
        self.sigmas.a = self.thetas.a * self.rhos.a
        self.update_xy()
        #self.periodic_boundary_condition()

    def closest_vert(self, sigma, zed):
        '''Return the vertices closer to a position
        `sigma`,`zed`
        '''
        dist = np.hypot(self.sigmas.fa - sigma,
                        self.zeds.fa - zed)
        idx = np.argmin(dist)
        sigma, zed = self.sigmas.fa[idx], self.zeds.fa[idx]
        s_matches = gt.find_vertex(self.graph,
                                   self.sigmas, sigma)
        z_matches = gt.find_vertex(self.graph,
                                   self.zeds, zed)
        log.debug('Number of closest vertices found: %i, %i'
                  % (len(s_matches), len(z_matches)))
        return [v for v in s_matches if v in z_matches][0]

    def periodic_boundary_condition(self):
        '''
        Applies the periodic boundary condition
        to the vertices positions along the sigma axis,
        with their curent value for rho.
        '''
        self.update_rhotheta()
        buf_theta = self.thetas.a + np.pi
        buf_theta = (buf_theta % (2 * np.pi)) - np.pi
        self.thetas.a = buf_theta
        self.sigmas.a = self.thetas.a * self.rhos.a
        self.update_xy()


    def any_edge(self, v0, v1):
        '''
        Returns the edge between vertices v0 and v1 if it exists,
        whether it goes from v0 to v1 or from v1 to v0 and None otherwize
        '''
        e = self.graph.edge(v0, v1)
        if e is None:
            e = self.graph.edge(v1, v0)
        return e

    def proj_sigma(self):
        ''' return an array of the positions projected on the
        cylinder with average rho radius
        '''
        self.update_rhotheta()
        rho_mean = self.rhos.a.mean()
        sigmas = self.thetas.copy()
        sigmas.a = self.thetas.a * rho_mean
        return sigmas

    # For clarity reason, those are not properties and return copies
    def rtz_pos(self):
        """
        Returns a **copy** of the rho theta zed values
        Note that no update is run.
        """
        rhos = self.graph.vertex_properties["rhos"].copy()
        thetas = self.graph.vertex_properties["thetas"].copy()
        zeds = self.graph.vertex_properties["zeds"].copy()
        rtzs = [rhos, thetas, zeds]
        return gt.group_vector_property(rtzs, value_type='float')

    def sz_pos(self):
        """
        Returns a **copy** of the sigma zed values
        Note that no update is run.
        """
        sigmas = self.graph.vertex_properties["sigmas"].copy()
        zeds = self.graph.vertex_properties["zeds"].copy()
        sigmazs = [sigmas, zeds]
        return gt.group_vector_property(sigmazs, value_type='float')

    def update_rhotheta(self):
        '''
        Computes the values of the `rhos`, `thetas` and `sigmas` property
        maps, from the current values of the `ixs` and `xys` property maps
        '''
        self.rhos.a, self.thetas.a = to_rhotheta(self.ixs.a, self.wys.a)
        self.sigmas.a = self.rhos.a * self.thetas.a

    def update_xy(self):
        '''Computes the values of the `ixs` and `xys` property
        maps, from the current values of the `rhos` and `thetas` property maps
        '''

        self.ixs.a, self.wys.a = to_xy(self.rhos.a, self.thetas.a)

    def edge_difference(self, vprop, eprop=None):

        vtype = vprop.value_type()
        if eprop == None:
            eprop = self.graph.new_edge_property(vtype)
        for e in self.graph.edges():
            eprop[e] = vprop[e.target()] - vprop[e.source()]
        return eprop

    def update_deltas(self):
        ''' Updates the edge coordinates from their vertices posittions'''
        dzeds = self.edge_difference(self.zeds)
        dixs = self.edge_difference(self.ixs)
        dwys = self.edge_difference(self.wys)

        self.dzeds.fa = dzeds.fa
        self.dixs.fa = dixs.fa
        self.dwys.fa = dwys.fa

    def update_dsigmas(self):

        drhos = self.edge_difference(self.rhos)
        self.drhos.a = drhos.a

        for e in self.graph.edges():
            self.edge_src_rhos[e] = self.rhos[e.source()]
            self.edge_trgt_rhos[e] = self.rhos[e.target()]

        edge_src_rhos = self.edge_src_rhos.a
        edge_trgt_rhos = self.edge_trgt_rhos.a

        dthetas = self.edge_difference(self.thetas)
        dsigmas = self.edge_difference(self.sigmas)

        # Periodic boundary conditions
        self.at_boundary.a[:] = 0
        at_boundary = self.at_boundary.a

        lower_than = [dthetas.a < -tau/2.]
        dthetas.a[lower_than] = dthetas.a[lower_than] + tau
        dsigmas.a[lower_than] = dsigmas.a[lower_than] + tau * edge_src_rhos[lower_than]
        at_boundary[lower_than] = 1

        higher_than = [dthetas.a > tau/2.]
        dthetas.a[higher_than] = dthetas.a[higher_than] - tau
        dsigmas.a[higher_than] = dsigmas.a[higher_than] - tau * edge_trgt_rhos[higher_than]
        at_boundary[higher_than] = 1

        self.dthetas.a = dthetas.a
        self.dsigmas.a = dsigmas.a
        self.at_boundary.a = at_boundary

    def update_edge_lengths(self):
        edge_lengths = np.sqrt(self.dixs.a**2
                               + self.dwys.a**2
                               + self.dzeds.a**2)
        self.u_dixs.a = self.dixs.a / edge_lengths
        self.u_dwys.a = self.dwys.a / edge_lengths
        self.u_dzeds.a = self.dzeds.a / edge_lengths
        self.edge_lengths.a = edge_lengths

    def out_delta_sz(self, vertex0, vertex1 ):
        edge01 = self.graph.edge(vertex0, vertex1)
        if edge01 is not None:
            return np.array([self.dsigmas[edge01],
                             self.dzeds[edge01]])
        edge10 = self.graph.edge(vertex1, vertex0)
        if edge10 is not None:
            return np.array([-self.dsigmas[edge10],
                             -self.dzeds[edge10]])
        return

    def rtz_record_array(self):
        rtz_dtype = [('rho', np.float32),
                     ('theta', np.float32),
                     ('zed', np.float32)]
        num_vertices = self.rhos.fa.size
        rtz_record = np.zeros((num_vertices,),
                              dtype=rtz_dtype)
        rtz_record['rho'] = self.rhos.fa
        rtz_record['theta'] = self.thetas.fa
        rtz_record['zed'] = self.zeds.fa
        return rtz_record

    def sz_record_array(self):
        sz_dtype = [('sigma', np.float32),
                    ('zed', np.float32)]
        num_vertices = self.sigmas.fa.size()
        sz_record = np.zeros((num_vertices,),
                              dtype=sz_dtype)
        sz_record['sigma'] = self.sigmas.fa
        sz_record['zed'] = self.zeds.fa
        return sz_record

    def get_sigmazs(self):
        """
        deprecated
        Should be understood by `gt.geometric_graph`
        """
        return np.array([self.sigmas().fa,
                         self.zeds().fa]).T

    def ordered_neighbours(self, vertex):
        """
        in the (sigma, zed) coordinate system with it's origin
        at the vertex position, sort the neighbours counter-clockwise
        """
        phis_out = [np.arctan2(self.dsigmas[edge],
                               self.dzeds[edge])
                     for edge in vertex.out_edges()]
        phis_in = [np.arctan2(-self.dsigmas[edge],
                              -self.dzeds[edge])
                    for edge in vertex.in_edges()]
        phis = np.append(phis_out, phis_in)
        vecinos_out = [vecino for vecino
                       in vertex.out_neighbours()]
        vecinos_in = [vecino for vecino
                      in vertex.in_neighbours()]
        vecinos = np.append(vecinos_out, vecinos_in)
        indexes = np.argsort(phis)
        vecinos = vecinos.take(indexes)
        return vecinos


class Triangle(object):
    '''
    A triangle is formed by a cell and two junction vertices linked
    by a junction edge

    Attributes:
    ===========
    eptm : a :class:`Epithelium` instance containing the triangle
    cell : the `cell` vertex forming one of the triangle's corner
    j_edge : the junction edge forming the triangle side
        opposing the cell vertex
    ctoj_edges : the cell to junction edges corresponding
        to the two other sides of the triangle
    deltas : ndarray with shape (2, 3) containing the 2 vecors
        :math:`(r_{\alpha i}, r_{\alpha j})`
    rij_vect : ndarray, the :math:`r_{i j}` vector
    cross : ndarray, the cross product between the two cell to junction edges:
        :math:`(r_{\alpha i} \times r_{\alpha j})`
    area : float, the triangle area
    u_cross : ndarray, the unitary vector colinear to the cross product



    Method:
    =======

    update_geometry
    '''
    def __init__(self, eptm, cell, j_edge):

        self.eptm = eptm
        self.cell  = cell
        self.j_edge = j_edge
        self.ctoj_edges = [eptm.graph.edge(self.cell, j_edge.source()),
                           eptm.graph.edge(self.cell, j_edge.target())]
        self.update_geometry()

    def update_geometry(self):
        ctoj0, ctoj1 = self.ctoj_edges
        self.deltas = np.array([[self.eptm.dixs[ctoj0],
                                 self.eptm.dwys[ctoj0],
                                 self.eptm.dzeds[ctoj0]],
                                [self.eptm.dixs[ctoj1],
                                 self.eptm.dwys[ctoj1],
                                 self.eptm.dzeds[ctoj1]]])
        self.rij_vect = np.array([self.eptm.dixs[self.j_edge],
                                  self.eptm.dwys[self.j_edge],
                                  self.eptm.dzeds[self.j_edge]])
        self.cross = np.cross(self.deltas[0, :], self.deltas[1, :])
        self.area = np.linalg.norm(self.cross) / 2.
        self.u_cross = self.cross / (2. * self.area)
        jv0, jv1 = self.j_edge
        # self.height = ((self.eptm.rhos[jv0] + self.eptm.rhos[jv1]) / 2.
        #                - self.eptm.rho_lumen)
        self.height = self.eptm.rhos[self.cell] - self.eptm.rho_lumen
        self.vol = self.height * self.area
        self.length = self.eptm.edge_lengths[self.j_edge]

class Diamond(object):
    '''a :class:`Diamond` instance is constituted of a junction edge
    and its two adjacent cells. It is the union of two
    :class:`Triangle` instances.

    '''
    def __init__(self, eptm, j_edge, adj_cells):
        self.j_edge = j_edge
        j_verta, j_vertb = j_edge.source(), j_edge.target()
        self.j_verts = j_verta, j_vertb
        self.triangles = {}
        num_adj = len(adj_cells)
        if num_adj == 2:
            cell0, cell1 = adj_cells
            self.triangles[cell0] = Triangle(eptm, cell0, j_edge)
            self.triangles[cell1] = Triangle(eptm, cell1, j_edge)
            self.cells = cell0, cell1
        elif num_adj == 1:
            cell0 = adj_cells[0]
            self.triangles[cell0] = Triangle(eptm, cell0, j_edge)
            self.cells = cell0, None

    def update_geometry(self):
        for tr in self.triangles.values():
            tr.update_geometry()


class Cells():
    '''

    '''
    def __init__(self, eptm):
        self.eptm = eptm
        self.__verbose__ = self.eptm.__verbose__
        self.params = eptm.params

        if self.eptm.new :
            if self.eptm.generate:
                n_sigmas, n_zeds = (self.eptm.params['n_sigmas'],
                                    self.eptm.params['n_zeds'])

                rhos, sigmas, zeds = self._generate_rsz(n_sigmas, n_zeds)
                rsz = rhos, sigmas, zeds
                self.eptm.graph = self._generate_graph(rsz)
                EpitheliumFilters.__init__(self.eptm)
                AbstractRTZGraph.__init__(self.eptm)
                self.eptm.rhos.a = rhos
                self.eptm.zeds.a = zeds
                self.eptm.sigmas.a = sigmas
                self.eptm.thetas.a = sigmas/rhos
                self.eptm.update_xy()
                self.eptm.periodic_boundary_condition()
                self.eptm.is_cell_vert.a[:] = 1
            self._init_cell_geometry()
            self._init_cell_params()
            self.eptm.update_deltas()
            self.eptm.update_edge_lengths()
            self.eptm.update_dsigmas()
        else:
            self._get_cell_geometry()
            self._get_cell_params()

        self.junctions = self.eptm.graph.new_vertex_property('object')
        self.num_sides = self.eptm.graph.new_vertex_property('int')


    def __iter__(self):
        # for vertex in gt.find_vertex(self.eptm.graph,
        #                              self.eptm.is_cell_vert, 1):
        for vertex in self.eptm.graph.vertices():
            if self.eptm.is_cell_vert[vertex]:
                yield vertex

    def local_cells(self):
        for vertex in gt.find_vertex(self.eptm.graph,
                                     self.eptm.is_local_vert, 1):
            if self.eptm.is_cell_vert[vertex]:
                yield vertex

    def _init_cell_geometry(self):
        '''
        Creates the `areas`, `vols` and `perimeters` properties
        '''
        area0 = self.params['prefered_area']
        height0 = self.params['prefered_height']
        self.areas = self.eptm.graph.new_vertex_property('float')

        self.areas.a[:] = area0
        self.eptm.graph.vertex_properties["areas"] = self.areas

        self.perimeters = self.eptm.graph.new_vertex_property('float')
        self.perimeters.a[:] = 6 * self.params['lambda_0']
        self.eptm.graph.vertex_properties["perimeters"]\
            = self.perimeters
        vol0 = area0 * height0
        self.vols = self.eptm.graph.new_vertex_property('float')
        self.vols.a[:] = vol0
        self.eptm.graph.vertex_properties["vols"] = self.vols

    def _get_cell_geometry(self):
        self.areas = self.eptm.graph.vertex_properties["areas"]
        self.perimeters = self.eptm.graph.vertex_properties["perimeters"]
        self.vols = self.eptm.graph.vertex_properties["vols"]

    def _init_cell_params(self):
        '''
        Creates the parameter dependant propery maps
        '''
        area0 = self.params['prefered_area']
        height0 = self.params['prefered_height']
        vol0 = area0 * height0
        self.prefered_vol = self.eptm.graph.new_vertex_property('float')
        self.prefered_vol.a[:] = vol0
        self.eptm.graph.vertex_properties["prefered_vol"] = self.prefered_vol

        contractility0 = self.params['contractility']
        self.contractilities =self.eptm.graph.new_vertex_property('float')
        self.contractilities.a[:] = contractility0
        self.eptm.graph.vertex_properties["contractilities"]\
            = self.contractilities

        vol_elasticity0 = self.params['vol_elasticity']
        self.vol_elasticities =self.eptm.graph.new_vertex_property('float')
        self.vol_elasticities.a[:] = vol_elasticity0
        self.eptm.graph.vertex_properties["vol_elasticities"]\
            = self.vol_elasticities

        self.ages = self.eptm.graph.new_vertex_property('int')
        self.ages.a[:] = 0
        self.eptm.graph.vertex_properties["ages"]\
            = self.ages

    def _get_cell_params(self):
        self.prefered_vol = self.eptm.graph.vertex_properties["prefered_vol"]
        self.contractilities\
            = self.eptm.graph.vertex_properties["contractilities"]
        self.vol_elasticities\
            = self.eptm.graph.vertex_properties["vol_elasticities"]
        self.ages = self.eptm.graph.vertex_properties["ages"]


    def _generate_rsz(self, n_sigmas=5, n_zeds=20):

        lambda_0 = self.eptm.params['lambda_0']
        height0 = self.eptm.params['prefered_height']
        #rho_c = (n_sigmas - 1) * lambda_0 / (2 * np.pi)
        rho_c = (n_sigmas) * lambda_0 / (2 * np.pi)
        self.eptm.rho_lumen = rho_c - height0
        delta_sigma = 2 * np.pi * rho_c / n_sigmas
        delta_z = delta_sigma * np.sqrt(3)/2.

        self.n_zeds = int(n_zeds)
        self.n_sigmas = int(n_sigmas)
        log.info('''Creating a %i x %i cells lattice'''
                 % (self.n_zeds, self.n_sigmas))
        rhos = np.ones(n_sigmas * n_zeds) * rho_c
        zt_grid = np.mgrid[:n_zeds, :n_sigmas]
        sigmas = zt_grid[1].astype('float')
        sigmas[::2, ...] += 0.5
        sigmas *= delta_sigma
        zeds = zt_grid[0].astype('float')
        zeds *= delta_z
        zeds -= zeds.max() / 2
        return rhos, sigmas.T.flatten(), zeds.T.flatten()

    def _generate_graph(self, rsz):
        rhos, sigmas, zeds = rsz
        sigmazs = np.array([sigmas, zeds]).T

        radius = self.eptm.params['lambda_0'] * 1.1
        rhoc = rhos[0]
        # Boundary conditions
        s_min, s_max = 0, 2 * np.pi * rhoc
        z_min, z_max = -10 * rhoc , 10 * rhoc
        #Graph instanciation
        graph, geom_pos = gt.geometric_graph(sigmazs, radius,
                                             [(s_min, s_max),
                                              (z_min, z_max)])
        return graph

    def update_junctions(self, cell):
        self.junctions[cell] = self.get_cell_junctions(cell)
        self.num_sides[cell]\
            = self.eptm.graph.degree_property_map('out')[cell]

    def get_cell_junctions(self, cell):
        jvs = [jv for jv in cell.out_neighbours()]
        j_edges = []
        for jv0 in jvs:
            for jv1 in jvs:
                if jv1 == jv0 : continue
                j_edge = self.eptm.graph.edge(jv0, jv1)
                if j_edge is not None:
                    j_edges.append(j_edge)
        return j_edges

    def get_anisotropies(self, coords):
        anisotropies = self.areas.copy()
        anisotropies.a[:] = 0
        orientation = anisotropies.copy()
        self.eptm.update_rhotheta()

        for cell in self:
            if self.is_boundary(cell) or (not self.eptm.is_alive[cell]):
                continue
            anisotropies[cell], orientation[cell] = eval_anisotropy(cell, coords)
        return anisotropies, orientation

    def is_boundary(self, cell):
        return  any([self.eptm.at_boundary[ctoj]
                     for ctoj in cell.out_edges()])

    def polygon(self, cell, coord1, coord2):

        rel_sz = np.array([[self.eptm.dsigmas[ctoj],
                            self.eptm.dzeds[ctoj]]
                           for ctoj in cell.out_edges()])

        phis = np.arctan2(rel_sz[:, 0], rel_sz[:, 1])
        indices = np.argsort(phis)
        uv = np.array([[coord1[je],
                        coord2[je]]
                       for je in cell.out_neighbours()])
        uv = uv[indices]
        return uv, indices

    def get_neighbor_cells(self, cell):
        jes = self.junctions[cell]
        neighbors = []
        for je in jes:
            adj = self.eptm.junctions.adjacent_cells[je]
            if len(adj) == 2:
                cell0, cell1 = adj
                neighbors.append(cell0 if cell0 != cell else cell1)
            elif len(adj) == 1:
                neighbors.append(adj[0])
        return neighbors

class ApicalJunctions():

    def __init__(self, eptm):

        self.eptm = eptm
        self.__verbose__ = self.eptm.__verbose__
        self.graph = self.eptm.graph
        self.params = eptm.params
        self.adjacent_cells = self.eptm.graph.new_edge_property('object')
        if self.eptm.new :
            if self.eptm.generate:
                self._compute_voronoi()
            self._init_junction_params()
        else:
            self._get_junction_params()

    def __iter__(self):

        for edge in self.eptm.graph.edges():
            if self.eptm.is_junction_edge[edge] :
                yield edge

    def local_junctions(self):
        for edge in gt.find_edge(self.eptm.graph,
                                 self.eptm.is_local_edge, 1):
            if self.eptm.is_junction_edge[edge] :
                yield edge

    def _init_junction_params(self):

        line_tension0 = self.eptm.params['line_tension']
        self.line_tensions = self.eptm.graph.new_edge_property('float')
        self.line_tensions.a[:] = line_tension0
        self.eptm.graph.edge_properties["line_tensions"] = self.line_tensions

        radial_tension0 = self.eptm.params['radial_tension']
        self.radial_tensions = self.eptm.graph.new_vertex_property('float')
        self.radial_tensions.a[:] = radial_tension0
        self.eptm.graph.vertex_properties["radial_tensions"]\
                                 = self.radial_tensions

    def _get_junction_params(self):
        self.line_tensions\
            = self.eptm.graph.edge_properties["line_tensions"]
        self.radial_tensions\
            = self.eptm.graph.vertex_properties["radial_tensions"]


    def update_adjacent(self, j_edge):
        adj_cells = self.get_adjacent_cells(j_edge)
        self.adjacent_cells[j_edge] = adj_cells
        self.eptm.diamonds[j_edge] = Diamond(self.eptm,
                                             j_edge, adj_cells)

    def get_adjacent_cells(self, j_edge):
        jv0 = j_edge.source()
        jv1 = j_edge.target()
        cells_a = [cell for cell in jv0.in_neighbours()
                   if self.eptm.is_cell_vert[cell]]
        cells_b = [cell for cell in jv1.in_neighbours()
                   if self.eptm.is_cell_vert[cell]]
        common_cells = [cell for cell in cells_a if cell in cells_b]
        return common_cells

    def _compute_voronoi(self):
        n_dropped = 0
        eptm = self.eptm
        self.visited_cells = []

        for cell in self.eptm.cells:
            self.visited_cells.append(cell)
            new_jvs, new_ctoj_edges, ndrp = self._voronoi_nodes(cell)
            n_dropped += ndrp
        log.info("%i triangles were dropped" % n_dropped)
        # Cell to junction graph
        n_jdropped = 0
        eptm.update_xy()
        eptm.update_dsigmas()
        eptm.update_deltas()
        eptm.update_edge_lengths()
        self.visited_cells = []
        for ctoj_edge in eptm.graph.edges():
            if not eptm.is_ctoj_edge[ctoj_edge]:
                continue
            new_edge, dropped = self._voronoi_edges(ctoj_edge)
            n_jdropped += dropped
        log.info("%i junction edges were dropped" % n_dropped)
        del self.visited_cells

    def _voronoi_nodes(self, cell):

        eptm = self.eptm
        cutoff = eptm.params['pos_cutoff']
        vecinos = [v for v in eptm.ordered_neighbours(cell)
                   if eptm.is_cell_vert[v]]#that's ordered

        num_vecinos = len(vecinos)
        new_jvs = []
        new_ctoj_edges = []
        n_dropped = 0
        n_visited = 0
        for n0 in range(num_vecinos):
            # Loop over
            n1 = (n0 + 1) % num_vecinos
            vecino0, vecino1 = vecinos[n0], vecinos[n1]
            if vecino0 in self.visited_cells or vecino1 in self.visited_cells:
                n_visited += 1
                continue
            v0_sz = eptm.out_delta_sz(cell, vecino0)
            v1_sz = eptm.out_delta_sz(cell, vecino1)
            sigma, zed = c_circumcircle(np.array([0., 0.]),
                                        v0_sz, v1_sz, cutoff)
            sigma += eptm.sigmas[cell]
            zed += eptm.zeds[cell]
            if not np.isfinite(sigma) or sigma > 1e8:
                n_dropped += 1
                continue
            rho = (eptm.rhos[cell]
                   + eptm.rhos[vecino0] + eptm.rhos[vecino1]) / 3.
            try:
                theta = sigma / rho
            except ZeroDivisionError:
                log.warning('Error computing thetas')
                theta = 0
            # new junction vertex here
            # **directly** in the eptm graph
            j_vertex = eptm.graph.add_vertex()
            new_jvs.append(j_vertex)
            eptm.is_cell_vert[j_vertex] = 0
            eptm.rhos[j_vertex] = rho
            eptm.thetas[j_vertex] = theta
            eptm.zeds[j_vertex] = zed
            eptm.sigmas[j_vertex] = sigma
            new_ctoj_edges.extend([(cell, j_vertex),
                                   (vecino0, j_vertex),
                                   (vecino1, j_vertex)])
        for (cell, jv) in new_ctoj_edges:
            eptm.new_ctoj_edge(cell, jv)
        return new_jvs, new_ctoj_edges, n_dropped

    def _voronoi_edges(self, ctoj_edge):
        eptm = self.eptm
        cell0 = ctoj_edge.source()
        self.visited_cells.append(cell0)

        j_verts0 = [jv for jv in cell0.out_neighbours()
                    if not eptm.is_cell_vert[jv]]
        new_edges = []
        dropped = 0
        for cell1 in cell0.all_neighbours():
            if not eptm.is_cell_vert[cell1]: continue
            if cell1 in self.visited_cells:
                continue
            j_verts1 = [jv for jv in cell1.out_neighbours()
                        if not eptm.is_cell_vert[jv]]
            common_jvs = [jv for jv in j_verts0 if jv in j_verts1]
            if len(common_jvs) == 2:
                jv0, jv1 = common_jvs[0], common_jvs[1]
                new_edges.append(eptm.new_j_edge(jv0, jv1))
            else:
                dropped += 1
        return new_edges, dropped


def cell_polygon(cell, coords, step):
    ixs, wys = coords
    jv_ixs = np.array([ixs[jv] - ixs[cell]
                        for jv in cell.out_neighbours()])
    jv_wys = np.array([wys[jv] - wys[cell]
                         for jv in cell.out_neighbours()])
    jv_thetas = np.arctan2(jv_ixs, jv_wys)
    jv_ixs = jv_ixs[np.argsort(jv_thetas)]
    jv_wys = jv_wys[np.argsort(jv_thetas)]
    sg_xs = np.vstack([np.roll(jv_ixs, shift=1), jv_ixs]).T
    sg_ys = np.vstack([np.roll(jv_wys, shift=1), jv_wys]).T
    lengths = np.hypot(sg_xs[:,0] - sg_xs[:,1],
                       sg_ys[:,0] - sg_ys[:,1])
    num_points = np.round(lengths / step)
    num_points[num_points < 2] = 2
    line_x = np.concatenate([np.linspace(*sg_x, num=n_pts)[:-1]
                             for sg_x, n_pts in zip(sg_xs, num_points)])
    line_y = np.concatenate([np.linspace(*sg_y, num=n_pts)[:-1]
                             for sg_y, n_pts in zip(sg_ys, num_points)])
    return line_x, line_y

def ellipse(thetas, params):

    a, b, phi = params
    rho = a * b / np.sqrt((b * np.cos(thetas + phi))**2
                           + (a * np.sin(thetas + phi))**2)
    return rho

def residuals(params, data):
    x, y = data
    thetas = np.arctan2(y, x)
    rho = np.hypot(x, y)
    fit_rho = ellipse(thetas, params)
    return rho - fit_rho

def fit_ellipse(line_x, line_y):
    a = line_x.max()
    b = line_y.max()
    phi = 0
    params, r = sp.optimize.leastsq(residuals,
                                    [a, b, phi],
                                    [line_x, line_y])
    return params, r

def eval_anisotropy(cell, coords):
    line_x, line_y = cell_polygon(cell, coords, step=0.21)
    params, r = fit_ellipse(line_x, line_y)
    a, b, phi = params

    orientation = np.abs(90 - (phi % np.pi) * 180 / np.pi)
    anisotropy = b / a
    return anisotropy, orientation

