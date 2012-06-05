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
        self.thetas.a = thetas
        self.zeds.a = zeds
        self.sigmas.a = rhos * thetas

        #Grouping
        rtzs = [self.rhos, self.thetas, self.zeds]
        self.rtz_pos = gt.group_vector_property(rtzs, value_type='float')
        del rtzs

    def calc_sigmas(self):
        self.sigmas.a = self.rhos.a * self.thetas.a

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


    
class CellGraph(AbstractRTZGraph):
    '''
    
    '''
    def __init__(self, epithelium):
        
        self.epithelium = epithelium
        #Each vertex of `cell.graph` represents a cell
        
        rhos, thetas, zeds = self.generate_rtz()
        num_vertices = rhos.shape[0]

        sigmazs = np.array([rhos * thetas, zeds]).T

        radius = self.epithelium.params['lambda_0'] * 1.4
        rho_0 = self.epithelium.params['rho_0']
        
        # Boundary conditions
        s_min, s_max = 0, 2 * np.pi * rho_0
        z_min, z_max = -2 * rho_0, 2 * rho_0

        #Graph instanciation
        self.graph, geom_pos = gt.geometric_graph(sigmazs, radius, 
                                                  [(s_min, s_max),
                                                   (z_min, z_max)])
        del geom_pos 
        cutoff = self.epithelium.params['pos_cutoff']

        AbstractRTZGraph.__init__(self, rhos, thetas, zeds,
                                  self.graph, cutoff)

        self.order_neighbours()
        self.rhos_voronoi = self.graph.new_vertex_property('vector<double>')
        self.thetas_voronoi = self.graph.new_vertex_property('vector<double>')
        self.zeds_voronoi = self.graph.new_vertex_property('vector<double>')
        

    def generate_rtz(self):
        """
        Returns hexagonaly packed vertices on a cylindre, in
        the ::math:(\rho, \theta, z): coordinate system
        """
        rho_0 = self.epithelium.params['rho_0']
        lambda_0 = self.epithelium.params['lambda_0']
        pos_noise = self.epithelium.params['pos_noise']

        n_zeds = np.int(4 * rho_0 / lambda_0)
        #n_zeds -= n_zeds % 3
        delta_theta = 2 * np.arcsin(lambda_0 / (2 * rho_0))
        n_thetas = np.int(2 * np.pi / delta_theta)
        #n_thetas -= n_thetas % 3

        self.n_zeds = int(n_zeds)
        self.n_thetas = int(n_thetas)

        rhos = np.ones(n_thetas * n_zeds
                       ) * (n_thetas * delta_theta)
        zt_grid = np.mgrid[:n_zeds, :n_thetas]
        thetas = zt_grid[1].astype('float')
        thetas[::2, ...] += 0.5
        thetas *= delta_theta
        zeds = zt_grid[0].astype('float')
        zeds *= lambda_0
        zeds -= 2 * rho_0
        return rhos, thetas.flatten(), zeds.flatten()

            
    def compute_voronoi(self):

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
            sigmazs_voronoi = np.zeros((degree, 2))
            sigmazs_voronoi[:-1] = circumcircle(sigmazs[:-1], 
                                                   sigmazs_vecinos[:-1],
                                                   sigmazs_vecinos[1:])
            sigmazs_voronoi[-1] =  circumcircle(sigmazs[-1], 
                                                sigmazs_vecinos[-1],
                                                sigmazs_vecinos[0])
            rhos_voronoi = np.zeros(degree)
            rhos_voronoi[:-1] = np.array([rho[:-1],
                                          rhos_vecinos[:-1],
                                          rhos_vecinos[1:]]).mean(axis=0)
            rhos_voronoi[-1] = np.mean([rho[-1],
                                        rhos_vecinos[-1],
                                        rhos_vecinos[0]])

            thetas_voronoi = sigmazs_voronoi[:, 0]/rhos_voronoi
            self.rhos_voronoi[cell] = rhos_voronoi.ravel()
            self.thetas_voronoi[cell] = thetas_voronoi.ravel()
            self.zeds_voronoi[cell] = sigmazs_voronoi[:, 1].ravel()

    def draw(self, output='cell_graph.pdf', **kwargs):
        """
        Draws the graph with `gt.graph_draw`.
        """
        print 'graph view saved to %s' %output
        
        return gt.graph_draw(self.graph, self.sigmaz_pos,
                             output=output, **kwargs)
    
    def sfdp_draw(self):
        self.sfdp_pos = gt.graph_draw(self.graph,
                                      pos=gt.sfdp_layout(self.graph,
                                                         cooling_step=0.95,
                                                         epsilon=1e-3,
                                                         multilevel=True),
                                      output_size=(300,300),
                                      output="lattice_3d.pdf")


class AppicalJunctions():

    def __init__(self, epithelium):
        self.epithelium = epithelium
        cells = epithelium.cells
        self.graph = gt.Graph(directed=False)
        
        



    def draw(self, output='junction_graph.pdf', **kwargs):
        """
        Draws the graph with `gt.graph_draw`.
        """
        return gt.graph_draw(self.graph, self.pos,
                             output=output)



def circumcircle(u0, u1, u2):
    u1d = u1 - u0
    u2d = u2 - u0
    if len(u0.shape) == 2:
        centers = np.array([med_intersect(u1d[k,0], u1d[k,1],
                                          u2d[k,0], u2d[k,1])
                            for k in range(u0.shape[0])])
    else:
        centers = med_intersect(u1d[0], u1d[1],
                                u2d[0], u2d[1])

    uc = centers + u0
    return uc

def med_intersect(x1, y1, x2, y2):
    if y1**2 < 1e-12 and y2**2 > 1e-12:
        xc = x1 / 2.
        yc = (x2**2 + y2**2 - 2 * x2 * xc) / (2 * y2)  
        return xc, yc
    if y2**2 < 1e-12 and y1**2 > 1e-12:
        xc = x2 / 2.
        yc = (x1**2 + y1**2 - 2 * x1 * xc) / (2 * y1)  
        return xc, yc
    if y1**2 < 1e-12 and y2**2 < 1e-12:
        return np.inf, np.inf
    #Equation des mediatrices
    a1,  a2 = - x1/y1, - x2/y2
    if (a1 - a2)**2 < 1e-12:
        return np.inf, np.inf
    b1 = y1 / 2 - a1 * x1 / 2
    b2 = y2 / 2 - a2 * x2 / 2
    xc = (b2 - b1) / (a1 - a2)
    yc = a1 * xc + b1
    return xc, yc
