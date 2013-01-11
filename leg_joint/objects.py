#!/usr/bin/env python -*- coding: utf-8 -*-

"""This module provides the basic elements composing the leg epithelium.

The architecture of the epithelium model consists in a graph containing
two types of vertices: 

* The cells themselves
* The apical junctions vertices (It is
constructed initially as the Voronoi diagramm associated with the
cell centers triangulation, again in the ::math:(\rho, \sigma):
plane.)

and two types of edges:
* The junction edges, structuring the apical surface of the
epithelium
* The cell to junction edges linking one cell to its
neighbouring junction vertices.


This graph implemented as a class wrapping a
(`graph_tool`)[http://projects.skewed.de/graph-tool] object with the
::math:(\rho, \theta, z): coordinate system. The geometrical features
are defined in an abstract class named ::class:`AbstractRTZGraph`:,
from which ::class:`Epithelium`: and for commodity
the ::class:`Cells`: and ::class:`ApicalJunctions` are derived.

"""

import os
import numpy as np

import graph_tool.all as gt
from scipy import weave
from .filters import EpitheliumFilters
from .utils import to_xy, to_rhotheta

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = os.path.join(ROOT_DIR, 'default', 'params.xml')
tau = 2 * np.pi

class AbstractRTZGraph(object):
    '''
    Wrapper of a (`graph_tool`)[http://projects.skewed.de/graph-tool]
    object with the ::math:(\rho, \theta, z): coordinate system.
    '''

    def __init__(self):
        '''
        Create an `AbstractRTZGraph` object. This is not ment as
        a stand alone object, but should rather be sublcassed.

        Properties
        ===========
        rhos, thetas, zeds, sigmas: vertex PropertyMaps
            of the vertices positions along the different axes
        
        Methods:
        ===========
        rtz_group : Creates or updates two `GroupPropertyMap`,
            `rtz_pos` and `sz_pos` containing the ::math:(\rho, \theta, z):
            and ::math:(\sigma, z): positions of the vertices, respectively.
        
        
        
        '''
        self.graph.set_directed(True) #So each edge is a defined vector
        self.current_angle = 0
        if self.new :
            self._init_edge_bool_props()
            self._init_vertex_geometry()            
            self._init_edge_geometry()

        ## Properties that are not internalized
        ### edge properties from vertex properties
        self.edge_src_rhos = self.graph.new_edge_property('float')
        self.edge_trgt_rhos = self.graph.new_edge_property('float')

            
    def _init_edge_bool_props(self):
        '''
        '''
        at_boundary = self.graph.new_edge_property('bool')
        at_boundary.a[:] = 0
        self.graph.edge_properties["at_boundary"] = at_boundary
        is_new_edge = self.graph.new_edge_property('bool')
        is_new_edge.a[:] = 1
        self.graph.edge_properties["is_new_edge"] = is_new_edge

    @property
    def at_boundary(self):
        return self.graph.edge_properties["at_boundary"]
    @property
    def is_new_edge(self):
        return self.graph.edge_properties["is_new_edge"]
        
    def _init_vertex_geometry(self):
        # Position in the rho theta zed space
        rhos_p = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["rhos"] = rhos_p
        thetas_p = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["thetas"] = thetas_p
        zeds_p = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["zeds"] = zeds_p
        sigmas_p = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["sigmas"] = sigmas_p
        ixs_p = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["ixs"] = ixs_p
        wys_p = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["wys"] = wys_p
        
    @property
    def rhos(self):
        return self.graph.vertex_properties["rhos"]
    @property
    def ixs(self):
        return self.graph.vertex_properties["ixs"]
    @property
    def wys(self):
        return self.graph.vertex_properties["wys"]
    @property
    def thetas(self):
        return self.graph.vertex_properties["thetas"]
    @property
    def zeds(self):
        return self.graph.vertex_properties["zeds"]
    @property
    def sigmas(self):
        return self.graph.vertex_properties["sigmas"]
        
    def _init_edge_geometry(self):
        
        # deltas 
        dthetas = self.graph.new_edge_property('float')
        self.graph.edge_properties["dthetas"] = dthetas
        dsigmas = self.graph.new_edge_property('float')
        self.graph.edge_properties["dsigmas"] = dsigmas
        dzeds = self.graph.new_edge_property('float')
        self.graph.edge_properties["dzeds"] = dzeds
        drhos = self.graph.new_edge_property('float')
        self.graph.edge_properties["drhos"] = drhos
        dixs = self.graph.new_edge_property('float')
        self.graph.edge_properties["dixs"] = dixs
        dwys = self.graph.new_edge_property('float')
        self.graph.edge_properties["dwys"] = dwys

        # Edge lengths
        edge_lengths = self.graph.new_edge_property('float')
        self.graph.edge_properties["edge_lengths"] = edge_lengths

        # unitary vectors 
        u_dsigmas = self.graph.new_edge_property('float')
        self.graph.edge_properties["u_dsigmas"] = u_dsigmas
        u_dzeds = self.graph.new_edge_property('float')
        self.graph.edge_properties["u_dzeds"] = u_dzeds
        u_drhos = self.graph.new_edge_property('float')
        self.graph.edge_properties["u_drhos"] = u_drhos
        u_dixs = self.graph.new_edge_property('float')
        self.graph.edge_properties["u_dixs"] = u_dixs
        u_dwys = self.graph.new_edge_property('float')
        self.graph.edge_properties["u_dwys"] = u_dwys
        
    @property
    def dthetas(self):
        return self.graph.edge_properties["dthetas"]
    @property
    def dzeds(self):
        return self.graph.edge_properties["dzeds"]
    @property
    def drhos(self):
        return self.graph.edge_properties["drhos"]
    @property
    def dsigmas(self):
        return self.graph.edge_properties["dsigmas"]
    @property
    def dixs(self):
        return self.graph.edge_properties["dixs"]
    @property
    def dwys(self):
        return self.graph.edge_properties["dwys"]

    @property
    def u_drhos(self):
        return self.graph.edge_properties["u_drhos"]        
    @property
    def u_dixs(self):
        return self.graph.edge_properties["u_dixs"]        
    @property
    def u_dwys(self):
        return self.graph.edge_properties["u_dwys"]        
    @property
    def u_dsigmas(self):
        return self.graph.edge_properties["u_dsigmas"]
    @property
    def u_dzeds(self):
        return self.graph.edge_properties["u_dzeds"]        
    @property
    def edge_lengths(self):
        return self.graph.edge_properties["edge_lengths"]

    def scale(self, scaling_factor):
        self.ixs.a *= scaling_factor
        self.wys.a *= scaling_factor
        self.zeds.a *= scaling_factor
        self.update_rhotheta()

    def rotate(self, angle):
        self.update_rhotheta()
        buf_theta = self.thetas.a + np.pi
        buf_theta += angle
        buf_theta = (buf_theta % (2 * np.pi)) - np.pi
        self.thetas.a = buf_theta
        self.sigmas.a = self.thetas.a * self.rhos.a
        self.update_xy()
        #self.periodic_boundary_condition()

    def closest_vert(self, sigma, zed):
        dist = np.hypot(self.sigmas.fa - sigma,
                        self.zeds.fa - zed)
        idx = np.argmin(dist)
        sigma, zed = self.sigmas.fa[idx], self.zeds.fa[idx]
        s_matches = gt.find_vertex(self.graph,
                                   self.sigmas, sigma)
        z_matches = gt.find_vertex(self.graph,
                                   self.zeds, zed)
        if self.__verbose__: print len(s_matches), len(z_matches)
        return [v for v in s_matches if v in z_matches][0]
        
    def periodic_boundary_condition(self):
        '''
        Applies the periodic boundary condition
        to the vertices positions along the sigma axis,
        with their curent value for rho.
        '''
        #self.update_rhotheta()
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
        efilt = self.graph.get_edge_filter()
        self.graph.set_edge_filter(None)
        e = self.graph.edge(v0, v1)
        if e is None:
            e = self.graph.edge(v1, v0)
        self.graph.set_edge_filter(efilt[0], efilt[1])
        return e

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
        self.rhos.a, self.thetas.a = to_rhotheta(self.ixs.a, self.wys.a)
        self.sigmas.a = self.rhos.a * self.thetas.a

    def update_xy(self):
        self.ixs.a, self.wys.a = to_xy(self.rhos.a, self.thetas.a)

    def edge_difference(self, vprop, eprop=None):
        vtype = vprop.value_type()
        if eprop == None:
            eprop = self.graph.new_edge_property(vtype)
        for e in self.graph.edges():
            eprop[e] = vprop[e.target()] - vprop[e.source()]
        return eprop
        
    def update_deltas(self):
        dzeds = self.edge_difference(self.zeds)
        dixs = self.edge_difference(self.ixs)
        dwys = self.edge_difference(self.wys)

        self.dzeds.a = dzeds.a
        self.dixs.a = dixs.a
        self.dwys.a = dwys.a

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
        dthetas.a[lower_than] += tau
        dsigmas.a[lower_than] += tau * edge_src_rhos[lower_than]
        at_boundary[lower_than] = 1
        
        higher_than = [dthetas.a > tau/2.]
        dthetas.a[higher_than] -= tau
        dsigmas.a[higher_than] -= tau * edge_trgt_rhos[higher_than]
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
            return [self.dsigmas[edge01], self.dzeds[edge01]]
        edge10 = self.graph.edge(vertex1, vertex0)
        if edge10 is not None:
            return [-self.dsigmas[edge10], -self.dzeds[edge10]]
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

    def __init__(self, eptm, cell, j_edge):
        self.eptm = eptm
        self.cell  = cell
        self.j_edge = j_edge
        self.ctoj_edges = [eptm.graph.edge(self.cell, j_edge.source()),
                           eptm.graph.edge(self.cell, j_edge.target())]

    def update_geometry(self):
        ctoj0, ctoj1 = self.ctoj_edges
        self.deltas = np.array([[self.eptm.dixs[ctoj0], 
                                 self.eptm.dwys[ctoj0], 
                                 self.eptm.dzeds[ctoj0]],
                                [self.eptm.dixs[ctoj1], 
                                 self.eptm.dwys[ctoj1], 
                                 self.eptm.dzeds[ctoj1]]])
        
        self.cross = np.cross(self.deltas[0, :], self.deltas[1, :])
        self.area = np.linalg.norm(self.cross)
        self.height = (np.hypot(self.eptm.ixs[self.cell],
                                self.eptm.wys[self.cell])
                       - self.eptm.params['rho_lumen'])
        self.length = self.eptm.edge_lengths[self.j_edge]
        
        
class Diamond(object):

    def __init__(self, eptm, j_edge, adj_cells):
        self.j_edge = j_edge
        j_verta, j_vertb = j_edge.source(), j_edge.target()
        self.j_verts = j_verta, j_vertb
        self.triangles = {}
        num_adj = len(adj_cells)
        if num_adj  == 2:
            cell0, cell1 = adj_cells
            self.triangles[cell0] = Triangle(eptm, cell0, j_edge)
            self.triangles[cell1] = Triangle(eptm, cell1, j_edge)
            self.cells = cell0, cell1
        elif num_adj == 1:
            cell0 = adj_cells[0]
            self.triangles[cell0] = Triangle(eptm, cell0, j_edge)
            self.cells = cell0

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
        self.junctions = self.eptm.graph.new_vertex_property('object')
        #self.triangles = self.eptm.graph.new_vertex_property('object')
        if self.eptm.new :
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
            self._init_cell_gometry()
            self._init_cell_params()
            self.eptm.update_deltas()
            self.eptm.update_edge_lengths()
            self.eptm.update_dsigmas()
        

    def __iter__(self):
        # for vertex in gt.find_vertex(self.eptm.graph,
        #                              self.eptm.is_cell_vert, 1):
        for vertex in self.eptm.graph.vertices():
            if self.eptm.is_cell_vert[vertex]:
                yield vertex

    def _init_cell_gometry(self):
        '''
        Creates the `areas`, `vols` and `perimeters` properties 
        '''
        area0 = self.params['prefered_area']
        height0 = self.params['prefered_height']
        areas = self.eptm.graph.new_vertex_property('float')
        
        areas.a[:] = area0
        self.eptm.graph.vertex_properties["areas"] = areas

        perimeters =self.eptm.graph.new_vertex_property('float')
        perimeters.a[:] = 6 * self.params['lambda_0']
        self.eptm.graph.vertex_properties["perimeters"]\
            = perimeters
        vol0 = area0 * height0
        vols = self.eptm.graph.new_vertex_property('float')
        vols.a[:] = vol0
        self.eptm.graph.vertex_properties["vols"] = vols
        
    @property
    def vols(self):
        return self.eptm.graph.vertex_properties["vols"]
    @property
    def areas(self):
        return self.eptm.graph.vertex_properties["areas"]
    @property
    def perimeters(self):
        return self.eptm.graph.vertex_properties["perimeters"]
        
    def _init_cell_params(self):
        '''
        Creates the parameter dependant propery maps
        '''
        area0 = self.params['prefered_area']
        height0 = self.params['prefered_height']
        vol0 = area0 * height0
        vols = self.eptm.graph.new_vertex_property('float')
        vols.a[:] = vol0
        self.eptm.graph.vertex_properties["prefered_vol"] = vols
        
        contractility0 = self.params['contractility']        
        contractilities =self.eptm.graph.new_vertex_property('float')
        contractilities.a[:] = contractility0
        self.eptm.graph.vertex_properties["contractilities"]\
            = contractilities

        vol_elasticity0 = self.params['vol_elasticity']
        vol_elasticities =self.eptm.graph.new_vertex_property('float')
        vol_elasticities.a[:] = vol_elasticity0
        self.eptm.graph.vertex_properties["vol_elasticities"]\
            = vol_elasticities

        ages = self.eptm.graph.new_vertex_property('int')
        ages.a[:] = 0
        self.eptm.graph.vertex_properties["ages"]\
            = ages

    @property
    def ages(self):
        return self.eptm.graph.vertex_properties["ages"]
        
    @property
    def contractilities(self):
        return self.eptm.graph.vertex_properties["contractilities"]
    @property
    def elasticities(self):
        return self.eptm.graph.vertex_properties["elasticities"]
    @property
    def vol_elasticities(self):
        return self.eptm.graph.vertex_properties["vol_elasticities"]
    @property
    def prefered_vol(self):
        return self.eptm.graph.vertex_properties["prefered_vol"]
        
    def _generate_rsz(self, n_sigmas=5, n_zeds=20):

        lambda_0 = self.eptm.params['lambda_0']
        height0 = self.eptm.params['prefered_height']
        rho_c = (n_sigmas - 1) * lambda_0 / (2 * np.pi)
        self.eptm.params['rho_lumen'] = rho_c - height0
        delta_sigma = 2 * np.pi * rho_c / n_sigmas
        delta_z = delta_sigma * np.sqrt(3)/2.
        
        self.n_zeds = int(n_zeds)
        self.n_sigmas = int(n_sigmas)
        if self.__verbose__ :
            print ('''Creating a %i x %i cells lattice'''
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

    def update_junctions(self):
        for cell in self:
            self.junctions[cell] = self.get_cell_junctions(cell)
            
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
        
class ApicalJunctions():

    def __init__(self, eptm):

        self.eptm = eptm
        self.__verbose__ = self.eptm.__verbose__
        self.graph = self.eptm.graph
        self.params = eptm.params
        self.adjacent_cells = self.eptm.graph.new_edge_property('object')
        if self.eptm.new :
            self._compute_voronoi()
            self._init_junction_params()
        else:
            self.update_adjacent()
            
    def __iter__(self):

        # for edge in gt.find_edge(self.eptm.graph,
        #                          self.is_junction_edge, 1):

        for edge in self.eptm.graph.edges():
            if self.eptm.is_junction_edge[edge] :
                yield edge

    def _init_junction_params(self):
                                 
        line_tension0 = self.eptm.params['line_tension']
        line_tensions = self.eptm.graph.new_edge_property('float')
        line_tensions.a[:] = line_tension0
        self.eptm.graph.edge_properties["line_tensions"] = line_tensions

        radial_tension0 = self.eptm.params['radial_tension']
        radial_tensions = self.eptm.graph.new_vertex_property('float')
        radial_tensions.a[:] = radial_tension0
        self.eptm.graph.vertex_properties["radial_tensions"]\
                                 = radial_tensions

    @property
    def line_tensions(self):
        return self.eptm.graph.edge_properties["line_tensions"]

    @property
    def radial_tensions(self):
        return self.eptm.graph.vertex_properties["radial_tensions"]

    def update_adjacent(self):
        for j_edge in self:
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
        print "%i triangles were dropped" % n_dropped
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
        print "%i junction edges were dropped" % n_dropped
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
            sigma, zed = c_circumcircle([0,0], v0_sz,
                                        v1_sz, cutoff)
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
                print ('Error computing thetas')
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


        
def c_circumcircle(sz0, sz1, sz2, cutoff):

    c_code = '''
    double sigma0 = sz0[0];
    double sigma1 = sz1[0];
    double sigma2 = sz2[0];

    double zed0 = sz0[1];
    double zed1 = sz1[1];
    double zed2 = sz2[1];

    
    double x1 = sigma1 - sigma0;
    double y1 = zed1 - zed0;
    double x2 = sigma2 - sigma0;
    double y2 = zed2 - zed0;

    double xc;
    double yc;

    if (y1*y1 < cutoff*cutoff
        && y2*y2 > cutoff*cutoff) 
        {
        xc = x1 / 2.;
        yc = (x2*x2 + y2*y2 - x2*x1)/(2*y2);
        }
    else if (y2*y2 < cutoff*cutoff
             && y1*y1 > cutoff*cutoff) 
        {
        xc = x2 / 2.;
        yc = (x1*x1 + y1*y1 - x1*x2)/(2*y1);
        }
    else if (y1*y1 + y2*y2 < cutoff*cutoff)
        {
        xc = 1e12;
        yc = 1e12;
        }
    else
       {
       double a1 = -x1/y1;
       double a2 = -x2/y2;
       double b1 = (x1*x1 + y1*y1) / (2*y1);
       double b2 = (x2*x2 + y2*y2) / (2*y2);
       if ((a2 - a1) * (a2 - a1) < cutoff*cutoff)
           {
           xc = 1e12;
           yc = 1e12;
           }
       xc = (b1 - b2) / (a2 - a1);
       yc = a1 * xc + b1;
       } 
       py::tuple results(2);
       results[0] = xc + sigma0;
       results[1] = yc + zed0;

    return_val = results;
    '''
    return weave.inline(c_code,
                        arg_names=['sz0', 'sz1', 'sz2', 'cutoff'],
                        headers=['<math.h>'])

def dist_rtz(rtz0, rtz1):
    
    c_code = '''
    double r0 = rtz0[0];
    double t0 = rtz0[1];
    double z0 = rtz0[2];

    double r1 = rtz1[0];
    double t1 = rtz1[1];
    double z1 = rtz1[2];

    double dist = sqrt(r0*r0 + r1*r1 - 2*r0*r1*cos(t1 - t0) + (z1-z0)*(z1-z0));
    return_val = dist;
    '''
    return weave.inline(c_code,
                        arg_names=['rtz0', 'rtz1'],
                        headers=['<math.h>'])

