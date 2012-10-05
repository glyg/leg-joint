#!/usr/bin/env python -*- coding: utf-8 -*-

"""This module provides the basic elements composing the leg epithelium.

The architecture of the epithelium model consists in a graph containing
two types of vertices: 

* The cells themselves The appical junctions vertices (It is
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
the ::class:`Cells`: and ::class:`AppicalJunctions` are derived.

"""

import os
import numpy as np
from numpy.random import normal
import graph_tool.all as gt
from scipy import weave
from scipy.interpolate import splrep, splev

from filters import *


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

        if self.new :
            self._init_edge_bool_props()
            self._init_vertex_geometry()            
            self._init_edge_geometry()

            
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
    @property
    def rhos(self):
        return self.graph.vertex_properties["rhos"]
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
    def u_drhos(self):
        return self.graph.edge_properties["u_drhos"]        
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
        self.rhos.a *= scaling_factor
        self.sigmas.a *= scaling_factor
        self.zeds.a *= scaling_factor
        
    def periodic_boundary_condition(self):
        '''
        Applies the periodic boundary condition
        to the vertices positions along the sigma axis,
        with their curent value for rho.
        '''
        tau = 2 * np.pi
        sigmas = self.sigmas.fa
        rhos = self.rhos.fa
        # Higher than the period points are shifted back
        self.sigmas.fa[sigmas > tau * rhos] -= tau * rhos[sigmas > tau * rhos]
                
        # Lower than zeros points are shifted up
        self.sigmas.fa[sigmas < 0] += tau * rhos[sigmas < 0]
        if len(self.sigmas.fa[sigmas > tau * rhos]) > 0:
            print self.sigmas.fa[sigmas > tau * rhos]
        self.thetas.fa = self.sigmas.fa / rhos
        
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
    
    def relax_rhos(self):
        sigmas = self.sigmas.fa.copy()
        zeds = self.zeds.fa.copy()
        sorted_sigmas = sigmas[np.argsort(zeds)]
        z_bin_width = self.params['z_bin_width']
        n_zbins = np.floor(sorted_sigmas.size / z_bin_width)
        zeds.sort()
        sorted_sigmas = sorted_sigmas[:z_bin_width * n_zbins
                                      ].reshape((z_bin_width, n_zbins))
        ctrl_rhos = (sorted_sigmas.max(axis=0)
                     - sorted_sigmas.min(axis=0)) / 2 * np.pi
        rho_vs_zeds_tck = splrep(ctrl_rhos,
                                 zeds[z_bin_width / 2 :: z_bin_width],
                                 s=0, k=3)
        rhos = splev(self.zeds.fa, rho_vs_zeds_tck)
        self.rhos.fa = rhos
        
    def update_deltas(self):
        for edge in self.graph.edges():
            v0, v1 = edge.source(), edge.target()
            dzed = self.zeds[v1] - self.zeds[v0]
            self.dzeds[edge] = dzed
            drho = self.rhos[v1] - self.rhos[v0]
            self.drhos[edge] = drho

            if self.is_new_edge[edge]:
                dtheta = self.thetas[v1] - self.thetas[v0]
                # dtheta lies between -tau and tau
                if dtheta > 0.5 * tau :
                    dtheta -= tau
                    dsigma = self.rhos[v0] * dtheta
                    self.at_boundary[edge] = 1
                elif dtheta < - 0.5 * tau :
                    dtheta += tau
                    dsigma = self.rhos[v0] * dtheta
                    self.at_boundary[edge] = 1
                else :
                    dsigma = self.sigmas[v1] - self.sigmas[v0]
                    if abs(dsigma) > tau * self.rhos.a.max() :
                        print dtheta
                        print str(v0), str(v1)
                    self.at_boundary[edge] = 0
                    dtheta = dsigma / self.rhos[v0]
                    self.is_new_edge[edge] = 0
            else:
                rho0 = self.rhos[v0]
                dsigma = self.sigmas[v1] - self.sigmas[v0]
                if dsigma >= tau * rho0 / 2.:
                    dsigma -= tau * rho0
                    self.at_boundary[edge] = 1
                elif dsigma < - tau * rho0 / 2.:
                    dsigma += tau * rho0
                    self.at_boundary[edge] = 1
                dtheta = dsigma / self.rhos[v0]
            self.dthetas[edge] = dtheta
            self.dsigmas[edge] = dsigma
        
    def update_edge_lengths(self):
        edge_lengths = np.sqrt(self.dzeds.fa**2
                               + self.drhos.fa**2
                               + self.dsigmas.fa**2)
        cutoff = self.params["pos_cutoff"]
        edge_lengths = edge_lengths.clip(cutoff, edge_lengths.max())
        self.u_drhos.fa = self.drhos.fa / edge_lengths
        self.u_dsigmas.fa = self.dsigmas.fa / edge_lengths
        self.u_dzeds.fa = self.dzeds.fa / edge_lengths
        self.edge_lengths.fa = edge_lengths
        
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
        in the (\sigma, z) coordinate system with it's origin
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
    
    def sigmaz_draw(self, output='sigmaz_graph.pdf', **kwargs):
        """
        Draws the graph with `gt.graph_draw`.
        """
        output = os.path.join('drawings', output)
        sz_pos = self.sz_pos()
        pmap = gt.graph_draw(self.graph, sz_pos,
                             output=output, **kwargs)
        del pmap
        print 'graph view saved to %s' %output
    
    def sfdp_draw(self, output="lattice_3d.pdf", **kwargs):
        output = os.path.join('drawings', output)
        sfdp_pos = gt.graph_draw(self.graph,
                                 pos=gt.sfdp_layout(self.graph,
                                                    cooling_step=0.95,
                                                    epsilon=1e-3,
                                                    multilevel=True),
                                 output_size=(300,300),
                                 output=output)
        print 'graph view saved to %s' %output
        return sfdp_pos
        
    def add_position_noise(self, noise_amplitude):
        self.sigmas.fa += normal(0, noise_amplitude,
                                 self.sigmas.fa.size)
        self.zeds.fa += normal(0, noise_amplitude,
                               self.rhos.fa.size)
    
class Cells():
    '''
    
    '''
    def __init__(self, epithelium):
        self.epithelium = epithelium
        self.__verbose__ = self.epithelium.__verbose__
        self.params = epithelium.params
        if self.epithelium.new :
            #rhos, thetas, zeds = self._generate_rtz()
            rhos, sigmas, zeds = self._generate_rsz()
            rsz = rhos, sigmas, zeds
            self.epithelium.graph = self._generate_graph(rsz)
            EpitheliumFilters.__init__(self.epithelium)
            AbstractRTZGraph.__init__(self.epithelium)

            self.epithelium.rhos.a = rhos
            self.epithelium.zeds.a = zeds
            self.epithelium.sigmas.a = sigmas
            self.epithelium.thetas.a = sigmas/rhos

            self.epithelium.is_cell_vert.a[:] = 1

            self._init_cell_gometry()
            self._init_cell_params()
            self.epithelium.update_deltas()
            self.epithelium.update_edge_lengths()
            self.epithelium.periodic_boundary_condition()
        

    def __iter__(self):
        # for vertex in gt.find_vertex(self.epithelium.graph,
        #                              self.epithelium.is_cell_vert, 1):
        for vertex in self.epithelium.graph.vertices():
            if self.epithelium.is_cell_vert[vertex]:
                yield vertex

    def _init_cell_gometry(self):
        '''
        Creates the `areas` and `perimeters` properties 
        '''
        area0 = self.params['prefered_area']
        areas = self.epithelium.graph.new_vertex_property('float')
        areas.a[:] = area0
        self.epithelium.graph.vertex_properties["areas"] = areas

        perimeters =self.epithelium.graph.new_vertex_property('float')
        perimeters.a[:] = 6 * self.params['lambda_0']
        self.epithelium.graph.vertex_properties["perimeters"
                                     ] = perimeters

    @property
    def areas(self):
        return self.epithelium.graph.vertex_properties["areas"]
    @property
    def perimeters(self):
        return self.epithelium.graph.vertex_properties["perimeters"]
        
    def _init_cell_params(self):
        '''
        Creates the parameter dependant propery maps
        '''

        prefered_area0 =  self.params['prefered_area']
        prefered_area = self.epithelium.graph.new_vertex_property('float')
        prefered_area.a[:] = prefered_area0
        self.epithelium.graph.vertex_properties["prefered_area"
                                     ] = prefered_area

        contractility0 = self.params['contractility']        
        contractilities =self.epithelium.graph.new_vertex_property('float')
        contractilities.a[:] = contractility0
        self.epithelium.graph.vertex_properties["contractilities"
                                     ] = contractilities

        elasticity0 = self.params['elasticity']        
        elasticities =self.epithelium.graph.new_vertex_property('float')
        elasticities.a[:] = elasticity0
        self.epithelium.graph.vertex_properties["elasticities"
                                     ] = elasticities
            
    @property
    def contractilities(self):
        return self.epithelium.graph.vertex_properties["contractilities"]
    @property
    def elasticities(self):
        return self.epithelium.graph.vertex_properties["elasticities"]
    @property
    def prefered_area(self):
        return self.epithelium.graph.vertex_properties["prefered_area"]

    def _generate_rsz(self):
        rho_0 = self.epithelium.params['rho_0']
        lambda_0 = self.epithelium.params['lambda_0']
        n_sigmas = np.int(2 * np.pi * rho_0 / lambda_0)
        rho_c = (n_sigmas - 1) * lambda_0 / (2 * np.pi)
        delta_sigma = 2 * np.pi * rho_c / n_sigmas
        delta_z = delta_sigma * np.sqrt(3)/2.
        n_zeds = np.int(5 * rho_0 / delta_z)
        
        self.n_zeds = int(n_zeds)
        self.n_sigmas = int(n_sigmas)
        if self.__verbose__ : print ('''Creating a %i x %i cells lattice'''
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
        

    def _generate_rtz(self):
        """
        Returns hexagonaly packed vertices on a cylindre, in
        the :math:`(\rho, \theta, z)` coordinate system.
       
        
        """
        rho_0 = self.epithelium.params['rho_0']
        lambda_0 = self.epithelium.params['lambda_0']

        n_thetas = np.int(2 * np.pi * rho_0 / lambda_0)
        rho_c = (n_thetas - 1) * lambda_0 / (2 * np.pi)
        delta_theta = 2 * np.pi / n_thetas
        delta_z = delta_theta * rho_c * np.sqrt(3)/2.
        n_zeds = np.int(5 * rho_0 / delta_z)
        
        self.n_zeds = int(n_zeds)
        self.n_thetas = int(n_thetas)
        if self.__verbose__ : print ('''Creating
                              a %i x %i cells lattice'''
                              % (self.n_zeds, self.n_thetas))
        rhos = np.ones(n_thetas * n_zeds) * rho_c
        zt_grid = np.mgrid[:n_zeds, :n_thetas]
        thetas = zt_grid[1].astype('float')
        thetas[::2, ...] += 0.5
        thetas *= delta_theta
        zeds = zt_grid[0].astype('float')
        zeds *= delta_z
        zeds -= zeds.max() / 2
        return rhos, thetas.T.flatten(), zeds.T.flatten()

    def _generate_graph(self, rtz):
        rhos, sigmas, zeds = rtz
        sigmazs = np.array([sigmas, zeds]).T

        radius = self.epithelium.params['lambda_0'] * 1.1
        rhoc = rhos[0]
        # Boundary conditions
        s_min, s_max = 0, 2 * np.pi * rhoc
        z_min, z_max = -10 * rhoc , 10 * rhoc
        #Graph instanciation
        graph, geom_pos = gt.geometric_graph(sigmazs, radius,
                                             [(s_min, s_max),
                                              (z_min, z_max)])
        return graph

    def junction_edges(self, cell):
        jvs = [je.target() for je in cell.out_neighbours()]



        
class AppicalJunctions():

    def __init__(self, epithelium):

        self.epithelium = epithelium
        self.__verbose__ = self.epithelium.__verbose__
        self.graph = self.epithelium.graph
        self.params = epithelium.params
        if self.epithelium.new :
            self._compute_voronoi()
            self._init_junction_params()

    def __iter__(self):

        # for edge in gt.find_edge(self.epithelium.graph,
        #                          self.is_junction_edge, 1):

        for edge in self.epithelium.graph.edges():
            if self.epithelium.is_junction_edge[edge] :
                yield edge

    def _init_junction_params(self):
        line_tension0 = self.epithelium.params['line_tension']
        line_tensions = self.epithelium.graph.new_edge_property('float')
        line_tensions.a[:] = line_tension0
        self.epithelium.graph.edge_properties["line_tensions"] = line_tensions
        
    @property
    def line_tensions(self):
        return self.epithelium.graph.edge_properties["line_tensions"]
        
    def _compute_voronoi(self):
        n_dropped = 0
        n_visited = 0
        eptm = self.epithelium
        self.visited_cells = []

        for cell in self.epithelium.cells:
            self.visited_cells.append(cell)
            new_jvs, new_ctoj_edges, ndrp = self._voronoi_nodes(cell)
            n_dropped += ndrp
        print "%i triangles were dropped" % n_dropped
        # Cell to junction graph
        n_jdropped = 0
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

        eptm = self.epithelium
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
            # **directly** in the epithelium graph
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
        eptm = self.epithelium
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
                new_edges.append((jv0, jv1))
                j_edge = eptm.new_j_edge(jv0, jv1)
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

