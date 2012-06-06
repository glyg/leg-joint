#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from numpy.random import normal, random_sample
import graph_tool.all as gt
from xml_handler import ParamTree



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
        self.apical_junctions = AppicalJunctions(self)

        
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
        
        self.vecinos_indexes = self.graph.new_vertex_property('vector<int>')
        self.vecinos_pos = self.graph.new_vertex_property('vector<double>')

        #Assignation
        self.rhos.a = rhos
        self.thetas.a = thetas % (2 * np.pi)
        self.zeds.a = zeds
        self.sigmas.a = rhos * thetas

        #Grouping
        rtzs = [self.rhos, self.thetas, self.zeds]
        self.rtz_pos = gt.group_vector_property(rtzs, value_type='float')
        del rtzs
        sigmazs = [self.sigmas, self.zeds]
        self.sz_pos = gt.group_vector_property(sigmazs, value_type='float')
        del sigmazs

        #Custom data type for record arrays
        self.rtz_dtype = np.dtype([('rho', np.float32),
                                   ('theta', np.float32),
                                   ('zed', np.float32)])
        
    def rtz_record_array(self):

        num_vertices = self.rhos.a.size
        rtz_record = np.zeros((num_vertices,),
                              dtype=self.rtz_dtype)
        rtz_record['rho'] = self.rhos.a
        rtz_record['theta'] = self.thetas.a
        rtz_record['zed'] = self.zeds.a
        return rtz_record
        
        
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
        self.theta.a = sigmas / rho % (2 * np.pi)


    def order_neighbours(self, ndim=3):
        """
        for each cell, retrives the positions of
        the neighbours by turning counter-clockwise
        around the cell vertex
        
        """

        #We work in the (sigma, z) plane

        # in the (z, \sigma) coordinate system with it's origin
        # at the cell position, sort the neighbours counter-clockwise

        self.calc_sigmas()
        for vertex in self.graph.vertices():
            zetas = [np.arctan2(self.zeds[vecino] - self.zeds[vertex],
                                self.sigmas[vecino] - self.sigmas[vertex])
                     for vecino in vertex.all_neighbours()]
            zetas = np.array(zetas)
            indexes = np.argsort(zetas)
            self.vecinos_indexes[vertex] = indexes

            vecinos = np.array([vecino for vecino in vertex.all_neighbours()])
            vecinos = vecinos.take(indexes)
            
            vecinos_pos = np.array([self.rtz_pos[vecino] for vecino in vecinos])
            self.vecinos_pos[vertex] =  vecinos_pos.flatten()

    def all_vecinos(self, vertex):
        """
        Returns a array of neighbours with ordered counter-clockwise
        """
        vecinos = np.array([vecino for vecino in vertex.all_neighbours()])
        indexes = self.vecinos_indexes[vertex].a
        return vecinos.take(indexes)
        
    def _calc_zetas(self, vertex):
        self.zetas[vertex].a = zetas.flatten()

    def degree(self, vertex):
        return vertex.in_degree() + vertex.out_degree()


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

        #Each vertex of `cell.graph` represents a cell
        
        rhos, thetas, zeds = self.generate_rtz()
        num_vertices = rhos.shape[0]
        sigmas = rhos * thetas
        sigmazs = np.array([sigmas, zeds]).T
        radius = self.epithelium.params['lambda_0'] + cutoff
        rho_0 = self.epithelium.params['rho_0']
        rho_c = rhos[0]
        
        # Boundary conditions
        s_min, s_max = 0, 2 * np.pi * rho_c - radius
        z_min, z_max = -100 * rho_0, 100 * rho_0
        self.boundary = [(s_min, s_max), (z_min, z_max)]
        #Graph instanciation
        self.graph, self.geom_pos = gt.geometric_graph(sigmazs, radius, 
                                                       [(s_min, s_max),
                                                        (z_min, z_max)])

        AbstractRTZGraph.__init__(self, rhos, thetas, zeds,
                                  self.graph, cutoff)

        self.order_neighbours()
        self.rhos_voronoi = self.graph.new_vertex_property('vector<double>')
        self.thetas_voronoi = self.graph.new_vertex_property('vector<double>')
        self.zeds_voronoi = self.graph.new_vertex_property('vector<double>')
        self.sigmas_vornoi = self.graph.new_vertex_property('vector<double>')
        self.voronoi_edges = self.graph.new_edge_property('object')
        self.voronoi_vertices = self.graph.new_vertex_property('object')
        #self.compute_voronoi()
        for cell in self.graph.vertices():
            self.voronoi_vertices[cell] = []
        

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
        #n_zeds -= n_zeds % 3
        delta_theta = 2 * np.arcsin(lambda_0 / (2 * rho_0))
        n_thetas = np.int(2 * np.pi / delta_theta)

        rho_c = (n_thetas - 1) * lambda_0 / (2 * np.pi)

        #n_thetas -= n_thetas % 3

        self.n_zeds = int(n_zeds)
        self.n_thetas = int(n_thetas)

        rhos = np.ones(n_thetas * n_zeds) * rho_c
        zt_grid = np.mgrid[:n_zeds, :n_thetas]
        thetas = zt_grid[1].astype('float')
        thetas[::2, ...] += 0.5
        thetas *= delta_theta
        zeds = zt_grid[0].astype('float')
        zeds *= delta_z
        zeds -= zeds.max() / 2
        return rhos, thetas.flatten(), zeds.flatten()

            
    def compute_voronoi(self):

        cutoff = self.epithelium.params['pos_cutoff']
        #We work in the sigma, zed plane
        
        self.calc_sigmas()
        self.order_neighbours()
        for cell in self.graph.vertices():
            degree = self.degree(cell)
            pos = self.rtz_pos[cell].a
            
            rho = (self.rhos[cell],) * np.ones(degree)

            sigmazs = np.ones((degree, 2))
            sigmazs[:, 0] *= self.sigmas[cell]
            sigmazs[:, 1] *= self.zeds[cell]
            
            rhos_vecinos = np.array([self.rhos[vecino]
                                     for vecino in self.all_vecinos(cell)])
            sigmazs_vecinos = np.array([(self.sigmas[vecino], self.zeds[vecino])
                                        for vecino in self.all_vecinos(cell)])

            sigmazs_voronoi = np.zeros((2, degree))
            sigmazs_voronoi[:, :-1] = circumcircle(sigmazs[:-1].T, 
                                                   sigmazs_vecinos[:-1].T,
                                                   sigmazs_vecinos[1:].T,
                                                   cutoff)
            sigmazs_voronoi[:, -1] =  circumcircle(sigmazs[-1], 
                                                   sigmazs_vecinos[-1],
                                                   sigmazs_vecinos[0],
                                                   cutoff)
            rhos_voronoi = np.zeros(degree)
            rhos_voronoi[:-1] = np.array([rho[:-1],
                                          rhos_vecinos[:-1],
                                          rhos_vecinos[1:]]).mean(axis=0)
            rhos_voronoi[-1] = np.mean([rho[-1],
                                        rhos_vecinos[-1],
                                        rhos_vecinos[0]])

            self.zeds_voronoi[cell] = sigmazs_voronoi[1].ravel()
            self.sigmas_vornoi[cell] = sigmazs_voronoi[0].ravel()
            self.rhos_voronoi[cell] = rhos_voronoi.ravel()
            rhos = rhos_voronoi.clip(cutoff, rhos_voronoi.max())
            thetas_voronoi = sigmazs_voronoi[0] / rhos  % (2 * np.pi)
            self.thetas_voronoi[cell] = thetas_voronoi.ravel()
            self.voronoi_vertices[cell] = []





class AppicalJunctions(AbstractRTZGraph):

    def __init__(self, epithelium):
        self.epithelium = epithelium
        self.graph = gt.Graph(directed=False)
        cutoff = self.epithelium.params['pos_cutoff']
        n_dropped = 0
        cells = epithelium.cells
        rtzs = []
        visited_cells = []
        s_min, s_max = cells.boundary[0]
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
                v0_theta = cells.thetas[vecino0]
                v0_zed = cells.zeds[vecino0]

                v1_rho = cells.rhos[vecino1]
                v1_theta = cells.thetas[vecino1]
                v1_zed = cells.zeds[vecino1]

                shift0 = period_shift(v0_theta - cell_theta)
                shift1  = period_shift(v1_theta - cell_theta)
                    
                v0_sz = [v0_rho * (v0_theta + shift0), v0_zed]
                v1_sz = [v1_rho * (v1_theta + shift1), v1_zed]
                sigma, zed = circumcircle(cell_sz, v0_sz,
                                          v1_sz, cutoff)
                
                if not np.isfinite(sigma):
                    n_dropped += 1
                    raise ValueError
                    continue
                rho = (cell_rho + v0_rho + v1_rho) / 3.
                theta = (sigma / rho)#  % (2 * np.pi)

                j_vertex = self.graph.add_vertex()
                rtzs.append([rho, theta, zed])
                cells.voronoi_vertices[cell].append(j_vertex)
                if not j_vertex in cells.voronoi_vertices[vecino0]:
                    cells.voronoi_vertices[vecino0].append(j_vertex)
                    if not j_vertex in cells.voronoi_vertices[vecino1]:
                        cells.voronoi_vertices[vecino1].append(j_vertex)
            

            for vecino in vecinos:
                if vecino in visited_cells:
                    continue
                j_vert0 = None
                for jv in cells.voronoi_vertices[cell]:
                    if jv in cells.voronoi_vertices[vecino]:
                        if j_vert0 == None:
                            j_vert0 = jv
                        elif jv != j_vert0:
                            self.graph.add_edge(j_vert0, jv)
                    
        rtzs = np.array(rtzs)
        AbstractRTZGraph.__init__(self, rtzs[:, 0], rtzs[:, 1], rtzs[:, 2],
                                  self.graph, cutoff)
        print n_dropped


    def draw(self, output='junction_graph.pdf', **kwargs):
        """
        Draws the graph with `gt.graph_draw`.
        """
        pos = self.sz_pos
        return gt.graph_draw(self.graph, pos,
                             output=output)

def period_shift(dtheta):
    if - np.pi < dtheta < np.pi:
        return 0.
    elif dtheta > np.pi :
        return -2 * np.pi
    return 2 * np.pi
    

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
