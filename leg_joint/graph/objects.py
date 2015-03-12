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

import logging
log = logging.getLogger(__name__)


import graph_tool.all as gt
#from scipy import weave, spatial
from .filters import EpitheliumFilters
from .utils import to_xy, to_rhotheta
try:
    from .circumcircle import c_circumcircle
except ImportError:
    log.warning('''Compiling circumcircle failed - tissue generation won't work''')
from .data import default_params
from sklearn.decomposition import PCA



CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = default_params()
tau = 2 * np.pi


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
        return _triangles
    triangles = np.array([tri.a for tri in _triangles], dtype=np.int)
    return triangles

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
    ========

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
        else:
            self._get_edge_bool_props()

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

    def new_edge(self, vertex0, vertex1, source_edge):
        '''Adds an edge between vertex0 and vertex1 and copies the properties
        of source_edge to this new edge
        '''
        if self.any_edge(vertex0, vertex1) is None:
            new_edge = self.graph.add_edge(vertex0, vertex1)
            for prop in self.graph.edge_properties.values():
                prop[new_edge] = prop[source_edge]
            return new_edge
        else:
            return self.any_edge(vertex0, vertex1)

    def new_vertex(self, source_vertex):
        '''Adds a vertex and copies the properties
        of source_vertex to this new vertex
        '''
        new_v = self.graph.add_vertex()
        for prop in self.graph.vertex_properties.values():
            prop[new_v] = prop[source_vertex]
        return new_v

    def any_edge(self, v0, v1):
        '''
        Returns the edge between vertices v0 and v1 if it exists,
        whether it goes from v0 to v1 or from v1 to v0 and None otherwize
        '''
        e = self.graph.edge(v0, v1)
        if e is None:
            e = self.graph.edge(v1, v0)
        return e

    def edge_difference(self, vprop, eprop=None):
        eprop = gt.edge_difference(self.graph, vprop)
        return eprop

class Cells():
    '''

    '''
    def __init__(self, eptm):
        self.eptm = eptm
        self.__verbose__ = self.eptm.__verbose__
        self.params = eptm.params

        if self.eptm.new :
            raise NotImplementedError
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

    def update_junctions(self, cell):

        self.junctions[cell] = self.get_cell_junctions(cell)
        self.num_sides[cell] = self.eptm.graph.degree_property_map('out')[cell]

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
