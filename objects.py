#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from numpy.random import normal, random_sample
import graph_tool.all as gt
from xml_handler import ParamTree
from scipy import weave

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = os.path.join(ROOT_DIR, 'default', 'params.xml')


class Epithelium():
    
    def __init__(self, paramtree=None, paramfile=PARAMFILE):
        if paramtree == None:
            self.paramtree = ParamTree(paramfile)
        else:
            self.paramtree = paramtree
        self.params = self.paramtree.absolute_dic
        self.cells = CellGraph(self)
        self.junctions = AppicalJunctions(self)

    def calc_apical_geometry(self, cell):
        """
        The area is approximated as the sum
        of the areas of the triangles
        formed by the cell position and each junction
        """
        area = 0.
        perimeter = 0.
        self.cells.rtz_group()
        self.junctions.rtz_group()
        for edge in self.cells.junctions_edges[cell]:
            rtz0 = self.cells.rtz_pos[cell]
            rtz1 = self.junctions.rtz_pos[edge.source()]
            rtz2 = self.junctions.rtz_pos[edge.target()]
            d01 = dist_rtz(rtz0, rtz1)
            d02 = dist_rtz(rtz0, rtz2)
            d12 = dist_rtz(rtz1, rtz2)
            p = d01 + d02 + d12
            area += np.sqrt(p * (p - d01)
                            * (p - d02)
                            * (p - d12))
            self.junctions.edge_lengths[edge] = d12
            perimeter += d12
                            
        self.cells.areas[cell] = area
        self.cells.perimeters[cell] = perimeter
    
    def total_energy(self):
        elasticity


class AbstractRTZGraph(object):
    '''
    For both the cells and the junctions (hence abstract)
    Wraps `graph-tool` in the rho, theta, zed
    coordinate system.

    '''

    def __init__(self, rhos, thetas, zeds,
                 graph, pos_cutoff=1e-3):
        #Position in the rho theta zed space
        #Instanciation
        self.rhos = self.graph.new_vertex_property('float')
        self.thetas = self.graph.new_vertex_property('float')
        self.zeds = self.graph.new_vertex_property('float')
        #sigmas = rhos * thetas
        self.sigmas = self.graph.new_vertex_property('float')
        self.calc_sigmas()
        #Assignation
        self.rhos.a = rhos
        self.thetas.a = thetas % (2 * np.pi)
        self.zeds.a = zeds
        self.sigmas.a = rhos * thetas
        #Grouping
        self.rtz_group()
        self.vecinos_indexes = self.graph.new_vertex_property('vector<int>')
        self.at_boundary = self.graph.new_vertex_property('bool')
        self.order_neighbours()
        #Custom data type for record arrays
        self.rtz_dtype = np.dtype([('rho', np.float32),
                                   ('theta', np.float32),
                                   ('zed', np.float32)])
        self.sz_dtype = np.dtype([('sigma', np.float32),
                                  ('zed', np.float32)])

    def rtz_group(self):
        
        rtzs = [self.rhos, self.thetas, self.zeds]
        self.rtz_pos = gt.group_vector_property(rtzs, value_type='float')
        del rtzs
        sigmazs = [self.sigmas, self.zeds]
        self.sz_pos = gt.group_vector_property(sigmazs, value_type='float')
        del sigmazs

        
    def rtz_record_array(self):
        num_vertices = self.rhos.a.size
        rtz_record = np.zeros((num_vertices,),
                              dtype=self.rtz_dtype)
        rtz_record['rho'] = self.rhos.a
        rtz_record['theta'] = self.thetas.a
        rtz_record['zed'] = self.zeds.a
        return rtz_record

    def sz_record_array(self):
        self.calc_sigmas()
        num_vertices = self.rhos.a.size
        sz_record = np.zeros((num_vertices,),
                              dtype=self.sz_dtype)
        sz_record['sigma'] = self.sigmas.a
        rtz_record['zed'] = self.zeds.a
        return sz_record
        
    def calc_sigmas(self):
        self.sigmas.a = self.rhos.a * (self.thetas.a % (2 * np.pi))

    def get_sigmazs(self):
        """Should be understood by `gt.geometric_graph`
        """
        return np.array([self.sigmas.a,
                         self.zeds.a]).T

    def inv_sigma(self, sigmas):
        """
        to calculate theta from sigma. 
        cuts off rho values under `self.params['pos_cutoff']`
        """
        rho = self.rho.a.copy()
        cutoff = self.params['pos_cutoff']
        rho = rho.clip(cutoff, rho.max())
        self.theta.a = (sigmas / rho) % (2 * np.pi)

    def periodic_theta(self, vertex, ref_vertex):

        dtheta = self.thetas[vertex] - self.thetas[ref_vertex]
        if -np.pi <= dtheta <= np.pi:
            return self.thetas[vertex]
        if dtheta > np.pi :
            return self.thetas[vertex] - 2 * np.pi
        return self.thetas[vertex] + 2 * np.pi


        

    def order_neighbours(self, ndim=3):
        """
        
        """

        #We work in the (sigma, z) plane

        # in the (z, \sigma) coordinate system with it's origin
        # at the cell position, sort the neighbours counter-clockwise

        self.calc_sigmas()
        for vertex in self.graph.vertices():
            zetas = []
            self.at_boundary[vertex]
            for vecino in vertex.all_neighbours():
                theta = self.periodic_theta(vecino, vertex)
                if theta != self.thetas[vecino] : self.at_boundary[vertex] = 1
                zeta = np.arctan2(self.zeds[vecino] - self.zeds[vertex],
                                  theta - self.thetas[vertex])
                zetas.append(zeta)

            zetas = np.array(zetas)
            indexes = np.argsort(zetas)
            self.vecinos_indexes[vertex] = indexes

    def all_vecinos(self, vertex):
        """
        Returns a array of neighbours ordered counter-clockwise
        """
        vecinos = np.array([vecino for vecino in vertex.all_neighbours()])
        indexes = self.vecinos_indexes[vertex].a
        return vecinos.take(indexes)

    def degree(self, vertex):
        return vertex.out_degree()

    def sigmaz_draw(self, output='sigmaz_graph.pdf', **kwargs):
        """
        Draws the graph with `gt.graph_draw`.
        """
        pos = self.sz_pos
        pmap = gt.graph_draw(self.graph, self.sz_pos,
                             output=output, **kwargs)
        print 'graph view saved to %s' %output
    
    def sfdp_draw(self, output="lattice_3d.pdf"):
        
        sfdp_pos = gt.graph_draw(self.graph,
                                 pos=gt.sfdp_layout(self.graph,
                                                    cooling_step=0.95,
                                                    epsilon=1e-3,
                                                    multilevel=True),
                                 output_size=(300,300),
                                 output=output)
        print 'graph view saved to %s' %output
        return sfdp_pos
        
    
class CellGraph(AbstractRTZGraph):
    '''
    
    '''
    def __init__(self, epithelium):
        
        self.epithelium = epithelium
        cutoff = self.epithelium.params['pos_cutoff']
        contractility0 = self.epithelium.params['contractility']        
        elasticity0 = self.epithelium.params['elasticity']        

        rhos, thetas, zeds = self.generate_rtz()
        num_vertices = rhos.shape[0]
        sigmas = rhos * thetas
        sigmazs = np.array([sigmas, zeds]).T
        radius = self.epithelium.params['lambda_0'] + cutoff
        rho_0 = self.epithelium.params['rho_0']
        rho_c = rhos[0]
        
        # Boundary conditions
        s_min, s_max = 0, 2 * np.pi * rho_c
        z_min, z_max = -100 * rho_0, 100 * rho_0
        self.boundaries = [(s_min, s_max), (z_min, z_max)]
        #Graph instanciation
        self.graph, self.geom_pos = gt.geometric_graph(sigmazs, radius, 
                                                       [(s_min, s_max),
                                                        (z_min, z_max)])
        AbstractRTZGraph.__init__(self, rhos, thetas, zeds,
                                  self.graph, cutoff)

        self.junctions_edges = self.graph.new_vertex_property('object')
        self.junctions_vertices = self.graph.new_vertex_property('object')

        self.areas = self.graph.new_vertex_property('float')
        self.perimeters = self.graph.new_vertex_property('float')

        self.contractilities = self.graph.new_vertex_property('float')
        self.contractilities.a[:] = contractility0

        self.elasticities = self.graph.new_vertex_property('float')
        self.elasticities.a[:] = elasticity0

        for cell in self.graph.vertices():
            self.junctions_vertices[cell] = []
            self.junctions_edges[cell] = []


    def generate_rtz(self):
        """
        Returns hexagonaly packed vertices on a cylindre, in
        the :math:`(\rho, \theta, z)` coordinate system
        """
        rho_0 = self.epithelium.params['rho_0']
        lambda_0 = self.epithelium.params['lambda_0']
        pos_noise = self.epithelium.params['pos_noise']

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
        return rhos, thetas.flatten(), zeds.flatten()



class AppicalJunctions(AbstractRTZGraph):

    def __init__(self, epithelium):
        self.epithelium = epithelium
        self.graph = gt.Graph(directed=False)
        line_tension0 = self.epithelium.params['line_tension']

        cutoff = self.epithelium.params['pos_cutoff']

        self.compute_voronoi()
        AbstractRTZGraph.__init__(self, self.raw_rtzs[:, 0],
                                  self.raw_rtzs[:, 1],
                                  self.raw_rtzs[:, 2],
                                  self.graph, cutoff)
        self.edge_lengths = self.graph.new_edge_property('float')
        self.line_tensions = self.graph.new_edge_property('float')
        self.line_tensions.a[:] = line_tension0
        

    def compute_voronoi(self):
        n_dropped = 0
        cutoff = self.epithelium.params['pos_cutoff']
        rtzs = []
        cells = self.epithelium.cells
        visited_cells = []
        for cell in cells.graph.vertices():
            visited_cells.append(cell)
            vecinos = cells.all_vecinos(cell) #that's ordered
            num_vecinos = len(vecinos)
            
            cell_sigma = cells.sigmas[cell]
            cell_theta = cells.thetas[cell]
            cell_rho = cells.rhos[cell]
            cell_sz = cells.sz_pos[cell].a
            
            for n0 in range(num_vecinos):
                n1 = (n0 + 1) % num_vecinos
                vecino0 = vecinos[n0]
                vecino1 = vecinos[n1]
                if vecino0 in visited_cells or vecino1 in visited_cells:
                    continue

                v0_rho = cells.rhos[vecino0]
                v0_theta = cells.periodic_theta(vecino0, cell)
                v0_zed = cells.zeds[vecino0]

                v1_rho = cells.rhos[vecino1]
                v1_theta = cells.periodic_theta(vecino1, cell)
                v1_zed = cells.zeds[vecino1]
                    
                v0_sz = [v0_rho * v0_theta, v0_zed]
                v1_sz = [v1_rho * v1_theta, v1_zed]
                sigma, zed = circumcircle(cell_sz, v0_sz,
                                          v1_sz, cutoff)
                if not np.isfinite(sigma):
                    n_dropped += 1
                    # raise ValueError
                    continue
                rho = (cell_rho + v0_rho + v1_rho) / 3.
                theta = (sigma / rho) % (2 * np.pi)

                j_vertex = self.graph.add_vertex()
                rtzs.append([rho, theta, zed])
                cells.junctions_vertices[cell].append(j_vertex)
                cells.junctions_vertices[vecino0].append(j_vertex)
                cells.junctions_vertices[vecino1].append(j_vertex)

            for vecino in vecinos:
                if vecino in visited_cells:
                    continue
                j_vert0 = None
                for jv in cells.junctions_vertices[cell]:
                    if jv in cells.junctions_vertices[vecino]:
                        if j_vert0 == None:
                            j_vert0 = jv
                        else:
                            junction = self.graph.add_edge(j_vert0, jv)
                            cells.junctions_edges[cell].append(junction)
                            cells.junctions_edges[vecino].append(junction)
        print n_dropped
        self.raw_rtzs = np.array(rtzs)


def circumcircle(u0, u1, u2, cutoff):
    u1_loc = u1 - u0
    u2_loc = u2 - u0
    centers = med_intersect(u1_loc[0], u1_loc[1],
                            u2_loc[0], u2_loc[1], cutoff)
    uc = centers + u0
    return uc

def med_intersect(x1, y1, x2, y2, cutoff):
    if y1**2 < cutoff**2 and y2**2 > cutoff**2:
        xc = x1 / 2.
        yc = (x2**2 + y2**2 - 2 * x2 * xc) / (2 * y2)  
        return xc, yc
    if y2**2 < cutoff**2 and y1**2 > cutoff**2:
        xc = x2 / 2.
        yc = (x1**2 + y1**2 - 2 * x1 * xc) / (2 * y1)  
        return xc, yc
    if y1**2 + y2**2 < cutoff**2:
        if x1**2 + x2**2 < cutoff**2:
            print 'points are superimposed'
            return x1 / 2., y1 / 2.
        else:
            return np.nan, np.nan
    #Equation des mediatrices
    a1,  a2 = - x1/y1, - x2/y2
    if (a1 - a2)**2 < cutoff**2:
        print 'points are aligned'
        return np.inf, np.inf
    b1 = y1 / 2. - a1 * x1 / 2.
    b2 = y2 / 2. - a2 * x2 / 2.
    xc = (b2 - b1) / (a1 - a2)
    yc = a1 * xc + b1
    
    return xc, yc

def dist_rtz(rtz0, rtz1):
    
    r0, t0, z0 = rtz0
    r1, t1, z1 = rtz1
    d01 = 0.
    c_code = '''
    d01 = sqrt(pow(r0, 2) + pow(r1, 2) - 2 * r0 * r1 * cos(t1 - t0) + pow((z1 - z0), 2));
    '''
    weave.inline(c_code,
                 arg_names=['r0', 'r1', 't0', 't1', 'z0', 'z1', 'd01'],
                 headers=['<math.h>'])
    return d01
