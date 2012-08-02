#!/usr/bin/env python -*- coding: utf-8 -*-

"""
This module provides the basic elements composing the leg epithelium.

The architecture of the epithelium model consists in a graph devided
in two graphviews
two networks (aka graphs):

 * The first graphview is composed by the cells themselves and is obtained by
 the Delaunay triangulation of a set of points in the ::math:(\rho, \sigma):
 plane, representing the cell centers. An edge of this graph defines a
 simple neighbourhood relationship. 

 * The second graphview represents the appical junctions, and is constructed
 initially as the Voronoi diagramm associated with the cell centers
 triangulation, again in the ::math:(\rho, \sigma): plane.
 An edge of the appical junctions corresponds to the interface between
 two neighbouring cells.
 
This graph implemented as a class wrapping a
(`graph_tool`)[http://projects.skewed.de/graph-tool] object with the
::math:(\rho, \theta, z): coordinate system. The geometrical features
are defined in an abstract class named ::class:`AbstractRTZGraph`:,
from which ::class:`Epithelium`: and for commodity
the ::class:`Cells`: and ::class:`AppicalJunctions` are derived.
"""

import os, time
import numpy as np
from numpy.random import normal, random_sample
import graph_tool.all as gt
from xml_handler import ParamTree
from scipy import weave
from scipy.interpolate import splrep, splev

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = os.path.join(ROOT_DIR, 'default', 'params.xml')

class AbstractRTZGraph(object):
    '''
    Wrapper of a (`graph_tool`)[http://projects.skewed.de/graph-tool]
    object with the ::math:(\rho, \theta, z): coordinate system.
    '''

    def __init__(self):
        '''
        Create an `AbstractRTZGraph` object. This is not ment as
        a stand alone object, but should rather be sublcassed.

        Parameters:
        ===========
        rhos, thetas, zeds : arrays
            three 1D arrays containing the coordinates of the graph vertices.
            They must have the as many points as there are graph vertices.
        graph: a `graph_tool Graph` object
            The graph should contain as many vertices as points in the
            `rhos, thetas, zeds` arrays.

        Properties
        ===========
        rhos, thetas, zeds: vertex PropertyMaps of the vertices positions 
        sigmas: vertex PropertyMaps of the vertices ::math:\sigma = \rho\theta:
        vecinos: vertex PropertyMap containing a list of the vertex neighbours
            ordered counter-clockwize in the ::math:(\rho, \sigma): plane around
            the central vertex
        
        Methods:
        ===========
        rtz_group : Creates or updates two `GroupPropertyMap`,
            `rtz_pos` and `sz_pos` containing the ::math:(\rho, \theta, z):
            and ::math:(\sigma, z): positions of the vertices, respectively.
        
        
        
        '''
        self.graph.set_directed(True) #So each edge is a defined vector

        if len(self.graph.properties) == 0 :
            at_boundary = self.graph.new_edge_property('bool')
            at_boundary.a[:] = 1
            is_new_edge = self.graph.new_edge_property('bool')
            is_new_edge.a[:] = 1
            is_local_cell = self.graph.new_vertex_property('bool')
            is_local_cell.a[:] = 0
            is_local_vert = self.graph.new_vertex_property('bool')
            is_local_vert.a[:] = 0
            is_local_j_vert = self.graph.new_vertex_property('bool')
            is_local_j_vert.a[:] = 0
            is_local_both = self.graph.new_vertex_property('bool')
            is_local_both.a[:] = 0
            is_alive = self.graph.new_vertex_property('bool')
            is_alive.a[:] = 1
            is_cell_vert = self.graph.new_vertex_property('bool')
            # Internalisation
            self.graph.vertex_properties["is_local_cell"] = is_local_cell
            self.graph.vertex_properties["is_local_j_vert"] = is_local_j_vert
            self.graph.vertex_properties["is_local_vert"] = is_local_vert
            self.graph.vertex_properties["is_local_both"] = is_local_both
            self.graph.vertex_properties["is_alive"] = is_alive
            self.graph.vertex_properties["is_cell_vert"] = is_cell_vert
            self.graph.edge_properties["at_boundary"] = at_boundary
            self.graph.edge_properties["is_new_edge"] = is_new_edge

            is_local_edge = self.graph.new_edge_property('bool')
            is_local_edge.a[:] = 0
            is_local_j_edge = self.graph.new_edge_property('bool')
            is_local_j_edge.a[:] = 0
            is_junction_edge = self.graph.new_edge_property('bool')
            is_ctoj_edge = self.graph.new_edge_property('bool')
            self.graph.edge_properties["is_junction_edge"
                                       ] = is_junction_edge
            self.graph.edge_properties["is_ctoj_edge"] = is_ctoj_edge
            self.graph.edge_properties["is_local_edge"] = is_local_edge
            self.graph.edge_properties["is_local_j_edge"] = is_local_j_edge

            # Position in the rho theta zed space
            rhos_p = self.graph.new_vertex_property('float')
            zeds_p = self.graph.new_vertex_property('float')
            thetas_p = self.graph.new_vertex_property('float')
            sigmas_p = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["rhos"] = rhos_p
            self.graph.vertex_properties["thetas"] = thetas_p
            self.graph.vertex_properties["zeds"] = zeds_p
            self.graph.vertex_properties["sigmas"] = sigmas_p

            # distances and unitary vectors
            edge_lengths = self.graph.new_edge_property('float')
            dthetas = self.graph.new_edge_property('float')
            dsigmas = self.graph.new_edge_property('float')
            dzeds = self.graph.new_edge_property('float')
            drhos = self.graph.new_edge_property('float')
            u_dsigmas = self.graph.new_edge_property('float')
            u_dzeds = self.graph.new_edge_property('float')
            u_drhos = self.graph.new_edge_property('float')
            # Internalisation
            self.graph.edge_properties["edge_lengths"] = edge_lengths
            self.graph.edge_properties["dthetas"] = dthetas
            self.graph.edge_properties["dsigmas"] = dsigmas
            self.graph.edge_properties["dzeds"] = dzeds
            self.graph.edge_properties["drhos"] = drhos
            self.graph.edge_properties["u_dsigmas"] = u_dsigmas
            self.graph.edge_properties["u_dzeds"] = u_dzeds
            self.graph.edge_properties["u_drhos"] = u_drhos
            # Gradients and energy
            grad_sigma = self.graph.new_vertex_property('float')
            grad_zed = self.graph.new_vertex_property('float')
            energy_grad = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["grad_sigma"] = grad_sigma
            self.graph.vertex_properties["grad_zed"] = grad_zed
            self.graph.vertex_properties["energy_grad"] = energy_grad

    @property
    def zeds(self):
        return self.graph.vertex_properties["zeds"]
    @property
    def sigmas(self):
        return self.graph.vertex_properties["sigmas"]
    @property
    def rhos(self):
        return self.graph.vertex_properties["rhos"]
    @property
    def thetas(self):
        return self.graph.vertex_properties["thetas"]
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
    def u_dsigmas(self):
        return self.graph.edge_properties["u_dsigmas"]
    @property
    def u_dzeds(self):
        return self.graph.edge_properties["u_dzeds"]        
    @property
    def edge_lengths(self):
        return self.graph.edge_properties["edge_lengths"]
    @property
    def grad_sigma(self):
        return self.graph.vertex_properties["grad_sigma"]
    @property
    def grad_zed(self):
        return self.graph.vertex_properties["grad_zed"]
    @property
    def at_boundary(self):
        return self.graph.edge_properties["at_boundary"]
    @property
    def is_new_edge(self):
        return self.graph.edge_properties["is_new_edge"]

    @property
    def is_alive(self):
        return self.graph.edge_properties["is_alive"]
    @property
    def vecinos_indexes(self):
        return self.graph.vertex_properties["vecinos_indexes"]

    def any_edge(self, v0, v1):
        efilt = self.graph.get_edge_filter()
        self.graph.set_edge_filter(None)
        e = self.graph.edge(v0, v1)
        if e is None:
            e = self.graph.edge(v1, v0)
        self.graph.set_edge_filter(efilt[0], efilt[1])
        return e

    @property
    def energy_grad(self):
        return self.graph.vertex_properties["energy_grad"]

    # For clarity reason, those are not properties and return copies
    def rtz_pos(self, vfilt=None, inversed=False):
        """
        Returns a **copy** of the rho theta zed values
        Note that no update is run.
        """
        rhos = self.graph.vertex_properties["rhos"].copy()
        thetas = self.graph.vertex_properties["thetas"].copy()
        zeds = self.graph.vertex_properties["zeds"].copy()
        rtzs = [rhos, thetas, zeds]
        return gt.group_vector_property(rtzs, value_type='float')

    def sz_pos(self, vfilt=None, inversed=False):
        """
        Returns a **copy** of the sigma zed values
        Note that no update is run.
        """
        sigmas = self.graph.vertex_properties["sigmas"].copy()
        zeds = self.graph.vertex_properties["zeds"].copy()
        sigmazs = [sigmas, zeds]
        return gt.group_vector_property(sigmazs, value_type='float')
    
    def relax_rhos(self, vfilt=None, inversed=False):
        self.graph.set_vertex_filter(vfilt, inversed)
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
        self.graph.vertex_properties["rhos"].fa = rhos
        self.graph.set_vertex_filter(None)
        
    def update_deltas(self, efilt=None):
        tau = 2 * np.pi
        self.graph.set_edge_filter(efilt)
        for edge in self.graph.edges():
            v0, v1 = edge.source(), edge.target()
            dzed = self.zeds[v1] - self.zeds[v0]
            self.graph.edge_properties["dzeds"][edge] = dzed
            drho = self.rhos[v1] - self.rhos[v0]
            self.graph.edge_properties["drhos"][edge] = drho
            if self.is_new_edge[edge] or self.at_boundary[edge] :
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
                    if abs(dsigma) > 20. :
                        print dtheta
                        print str(v0), str(v1)
                    if self.is_new_edge[edge]:
                        self.at_boundary[edge] = 0
                    dtheta = dsigma / self.rhos[v0]
            else:
                dsigma = self.sigmas[v1] - self.sigmas[v0]
                dtheta = dsigma / self.rhos[v0]
                
            self.is_new_edge[edge] = 0
            self.graph.edge_properties["dthetas"][edge] = dtheta
            self.graph.edge_properties["dsigmas"][edge] = dsigma
        self.graph.set_edge_filter(None)
        
    def update_edge_lengths(self, efilt=None):
        self.graph.set_edge_filter(efilt)
        edge_lengths = np.sqrt(self.dzeds.fa**2
                               + self.drhos.fa**2
                               + self.dsigmas.fa**2)
        cutoff = self.params["pos_cutoff"]
        edge_lengths = edge_lengths.clip(cutoff, edge_lengths.max())
        self.graph.edge_properties["u_drhos"
                                   ].fa = self.drhos.fa / edge_lengths
        self.graph.edge_properties["u_dsigmas"
                                   ].fa = self.dsigmas.fa / edge_lengths
        self.graph.edge_properties["u_dzeds"
                                   ].fa = self.dzeds.fa / edge_lengths
        self.graph.edge_properties["edge_lengths"].fa = edge_lengths
        self.graph.set_edge_filter(None)
        
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

    def ordered_neighbours(self, vertex, vfilt=None):
        """
        in the (\sigma, z) coordinate system with it's origin
        at the vertex position, sort the neighbours counter-clockwise
        """
        self.graph.set_vertex_filter(vfilt)
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
        self.graph.set_vertex_filter(None)
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

    def set_new_pos(self, new_sz_pos, vfilt=None):
        new_sz_pos = new_sz_pos.flatten()
        if vfilt == None:
            self.graph.set_vertex_filter(self.is_cell_vert, inverted=True)
        else:
            vfilt = vfilt.copy()
            vfilt.a *= (1 - self.is_cell_vert.a)
            self.graph.set_vertex_filter(vfilt)
        assert len(new_sz_pos) / 2 == self.graph.num_vertices()
        self.sigmas.fa = new_sz_pos[::2]
        self.zeds.fa = new_sz_pos[1::2]
        self.graph.set_vertex_filter(None)

    def add_position_noise(self, noise_amplitude):
        self.sigmas.a += normal(0, noise_amplitude,
                                self.sigmas.a.size)
        self.zeds.a += normal(0, noise_amplitude,
                              self.rhos.a.size)
    
class Cells(AbstractRTZGraph):
    '''
    
    '''
    def __init__(self, epithelium):
        
        self.epithelium = epithelium
        self.params = epithelium.params
        self.graph = self.epithelium.graph
        rtz = self.generate_rtz()
        self.generate_graph(rtz)
        
        contractility0 = self.params['contractility']        
        prefered_area0 =  self.params['prefered_area']
        elasticity0 = self.params['elasticity']        

        areas =self.epithelium.graph.new_vertex_property('float')
        areas.a[:] = prefered_area0
        self.epithelium.graph.vertex_properties["areas"] = areas
            
        contractilities =self.epithelium.graph.new_vertex_property('float')
        contractilities.a[:] = contractility0
        self.epithelium.graph.vertex_properties["contractilities"
                                     ] = contractilities
        elasticities =self.epithelium.graph.new_vertex_property('float')
        elasticities.a[:] = elasticity0
        self.epithelium.graph.vertex_properties["elasticities"
                                     ] = elasticities

        perimeters =self.epithelium.graph.new_vertex_property('float')
        perimeters.a[:] = 6 * self.params['lambda_0']
        self.epithelium.graph.vertex_properties["perimeters"
                                     ] = perimeters

        prefered_area = self.epithelium.graph.new_vertex_property('float')
        prefered_area.a[:] = prefered_area0
        self.epithelium.graph.vertex_properties["prefered_area"
                                     ] = prefered_area
        self.epithelium.graph.vertex_properties["is_cell_vert"].a[:] = 1

        
        AbstractRTZGraph.__init__(self)

    def __iter__(self):
        for vertex in self.epithelium.graph.vertices():
            if self.epithelium.is_cell_vert[vertex]:
                yield vertex

    @property
    def areas(self):
        return self.epithelium.graph.vertex_properties["areas"]
    @property
    def contractilities(self):
        return self.epithelium.graph.vertex_properties["contractilities"]
    @property
    def elasticities(self):
        return self.epithelium.graph.vertex_properties["elasticities"]
    @property
    def perimeters(self):
        return self.epithelium.graph.vertex_properties["perimeters"]
    @property
    def prefered_area(self):
        return self.epithelium.graph.vertex_properties["prefered_area"]

    def generate_graph(self, rtz):
        rhos, thetas, zeds = rtz
        sigmas = rhos * (thetas % (2*np.pi))
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
        self.epithelium.graph = graph
        AbstractRTZGraph.__init__(self.epithelium)
        self.epithelium.graph.vertex_properties["rhos"].a = rhos
        self.epithelium.graph.vertex_properties["thetas"].a = thetas
        self.epithelium.graph.vertex_properties["zeds"].a = zeds
        self.epithelium.graph.vertex_properties["sigmas"].a = sigmas
            
    def generate_rtz(self):
        """
        Returns hexagonaly packed vertices on a cylindre, in
        the :math:`(\rho, \theta, z)` coordinate system.
       
        
        """
        rho_0 = self.epithelium.params['rho_0']
        lambda_0 = self.epithelium.params['lambda_0']

        delta_z = lambda_0 * np.sqrt(3) / 2.
        n_zeds = np.int(4 * rho_0 / delta_z)
        delta_theta = 2 * np.arcsin(lambda_0 / (2 * rho_0))

        n_thetas = np.int(2 * np.pi / delta_theta)
        rho_c = (n_thetas - 1) * lambda_0 / (2 * np.pi)
        delta_theta_c = 2 * np.pi / n_thetas

        self.n_zeds = int(n_zeds)
        self.n_thetas = int(n_thetas)

        rhos = np.ones(n_thetas * n_zeds) * rho_c
        zt_grid = np.mgrid[:n_zeds, :n_thetas]
        thetas = zt_grid[1].astype('float')
        thetas[::2, ...] += 0.5
        thetas *= delta_theta_c
        zeds = zt_grid[0].astype('float')
        zeds *= delta_z
        zeds -= zeds.max() / 2
        return rhos, thetas.T.flatten(), zeds.T.flatten()

class AppicalJunctions(AbstractRTZGraph):

    def __init__(self, epithelium):

        self.epithelium = epithelium
        self.graph = self.epithelium.graph
        self.params = epithelium.params
        self.compute_voronoi()
        line_tension0 = epithelium.params['line_tension']
        line_tensions = self.epithelium.graph.new_edge_property('float')
        line_tensions.a[:] = line_tension0
        self.epithelium.graph.edge_properties["line_tensions"] = line_tensions
        AbstractRTZGraph.__init__(self)

    def __iter__(self):
        for edge in self.epithelium.graph.edges():
            if self.epithelium.is_junction_edge[edge]:
                yield edge
    @property
    def line_tensions(self):
        return self.epithelium.graph.edge_properties["line_tensions"]

    def graphview(self):
        not_cell_vert = self.epithelium.is_cell_vert.copy()
        not_cell_vert.a = 1 - self.epithelium.is_cell_vert.a
        return gt.GraphView(self.epithelium.graph,
                            vfilt=not_cell_vert,
                            efilt=self.epithelium.is_junction_edge)

    def compute_voronoi(self):
        n_dropped = 0
        n_visited = 0
        #cells = self.epithelium.cells
        eptm = self.epithelium
        cutoff = eptm.params['pos_cutoff']
        visited_cells = []
        # Cell to cell graph to compute the junction vertices positions
        cells_graph = gt.GraphView(eptm.graph,
                                   vfilt=eptm.is_cell_vert)
        for cell in cells_graph.vertices():
            visited_cells.append(cell)
            vecinos = eptm.ordered_neighbours(cell) #that's ordered
            num_vecinos = len(vecinos)
            for n0 in range(num_vecinos):
                n1 = (n0 + 1) % num_vecinos
                vecino0 = vecinos[n0]
                vecino1 = vecinos[n1]
                if vecino0 in visited_cells or vecino1 in visited_cells:
                    n_visited += 1
                    continue
                v0_sz = eptm.out_delta_sz(cell, vecino0)
                v1_sz = eptm.out_delta_sz(cell, vecino1)
                sigma, zed = c_circumcircle([0,0], v0_sz,
                                            v1_sz, cutoff)
                sigma += eptm.sigmas[cell]
                zed += eptm.zeds[cell]
                if not np.isfinite(sigma) or sigma > 1e8:
                    # print 'defective triangle %s, %s, %s' % (str(cell),
                    #                                          str(vecino0),
                    #                                          str(vecino1))
                    n_dropped += 1
                    continue
                rho = (eptm.rhos[cell]
                       + eptm.rhos[vecino0] + eptm.rhos[vecino1]) / 3.
                theta = sigma / rho

                # new junction vertex here
                # **directly** in the epithelium graph
                j_vertex = eptm.graph.add_vertex()
                eptm.is_cell_vert[j_vertex] = 0

                eptm.rhos[j_vertex] = rho
                eptm.thetas[j_vertex] = theta
                eptm.zeds[j_vertex] = zed
                eptm.sigmas[j_vertex] = sigma

                # new cell to junction edges here
                cj = eptm.graph.add_edge(cell, j_vertex)
                eptm.is_ctoj_edge[cj] = 1
                eptm.is_junction_edge[cj] = 0
                eptm.at_boundary[cj] = 1
                eptm.is_new_edge[cj] = 1

                v0j = eptm.graph.add_edge(vecino0, j_vertex)
                eptm.is_ctoj_edge[v0j] = 1
                eptm.is_junction_edge[v0j] = 0
                eptm.at_boundary[v0j] = 1
                eptm.is_new_edge[v0j] = 1

                v1j = eptm.graph.add_edge(vecino1, j_vertex)
                eptm.is_ctoj_edge[v1j] = 1
                eptm.is_junction_edge[v1j] = 0
                eptm.at_boundary[v1j] = 1
                eptm.is_new_edge[v1j] = 1

        # Cell to junction graph
        n_jdropped = 0
        c2jgraph = gt.GraphView(eptm.graph,
                                   efilt=eptm.is_ctoj_edge)
        eptm.update_deltas()
        eptm.update_edge_lengths()
        print "%i triangles were dropped" % n_dropped
        visited_cells = []
        for link in c2jgraph.edges():
            cell0_link = link.source()
            j_verts0 = [jv for jv in cell0_link.out_neighbours()]
            assert link.target() in j_verts0
            cell0 = cells_graph.vertex(cell0_link)
            visited_cells.append(cell0)
            for cell1 in cell0.all_neighbours():
                if cell1 in visited_cells:
                    continue
                cell1_link = c2jgraph.vertex(cell1)
                j_verts1 = [jv for jv in cell1_link.out_neighbours()]
                common_jvs = [jv for jv in j_verts0 if jv in j_verts1]
                if len(common_jvs) == 2:
                    j_v0, j_v1 = common_jvs[0], common_jvs[1]
                    j_edge_out = eptm.graph.edge(j_v0, j_v1)
                    j_edge_in = eptm.graph.edge(j_v1, j_v0)
                    if j_edge_in is None and j_edge_out is None:
                        j_edge = eptm.graph.add_edge(j_v0, j_v1)
                        eptm.is_ctoj_edge[j_edge] = 0
                        eptm.is_junction_edge[j_edge] = 1
                        eptm.at_boundary[j_edge] = 1
                        eptm.is_new_edge[j_edge] = 1
                else:
                    n_jdropped += 1
        print "%i junction edges were dropped" % n_dropped


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

