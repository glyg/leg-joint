#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from numpy.random import normal, random_sample
import graph_tool.all as gt
from xml_handler import ParamTree


FLOAT = np.dtype('float64')
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
        rhos, thetas, zeds = self.generate_rtz_cells()
        self.num_cells = rhos.shape[0]
        self.rtz_dtype = np.dtype([('rho', FLOAT),
                                   ('theta', FLOAT),
                                   ('zed', FLOAT)])
        self.rtz_cells = np.zeros((self.num_cells,),
                                  dtype=self.rtz_dtype)
        self.rtz_cells['rho'] = rhos
        self.rtz_cells['theta'] = thetas
        self.rtz_cells['zed'] = zeds

        self.cell_network = CellGraph(self)
        self.apical_junctions = AppicalJunctions(self)

    def generate_rtz_cells(self):

        """
        Returns hexagonaly packed vertices on a cylindre, in
        the ::math:(\rho, \theta, z): coordinate system
        """
        rho_0 = self.params['rho_0']
        lambda_0 = self.params['lambda_0']
        pos_noise = self.params['pos_noise']

        n_zeds = np.floor(4 * rho_0 / lambda_0)
        #n_zeds -= n_zeds % 3
        delta_theta = 2 * np.arcsin(lambda_0 / (2 * rho_0))
        n_thetas = np.floor(2 * np.pi / delta_theta)
        #n_thetas -= n_thetas % 3

        self.n_zeds = int(n_zeds)
        self.n_thetas = int(n_thetas)

        rhos = np.ones(n_thetas * n_zeds) * rho_0 
        zt_grid = np.mgrid[:n_zeds, :n_thetas]
        thetas = zt_grid[1].astype(FLOAT)
        thetas[::2, ...] += 0.5
        thetas *= delta_theta
        zeds = zt_grid[0].astype(FLOAT)
        zeds *= lambda_0
        zeds -= 2 * rho_0
        return rhos, thetas.flatten(), zeds.flatten()
        

class CellGraph():

    def __init__(self, epithelium):
        
        self.epithelium = epithelium

        sigma = epithelium.rtz_cells['rho'] * epithelium.rtz_cells['theta']
        sigmaz = np.array([sigma, epithelium.rtz_cells['zed']]).T
        radius = self.epithelium.params['lambda_0'] * 1.4
        rho_0 = self.epithelium.params['rho_0']

        # Boundary conditions
        s_min, s_max = 0, 2 * np.pi * rho_0
        z_min, z_max = -2 * rho_0, 2 * rho_0

        #Each vertex of `cell.graph` represents a cell
        self.graph, self.sigmaz_pos = gt.geometric_graph(sigmaz, radius, 
                                                         [(s_min, s_max),
                                                          (z_min, z_max)])

        #Position in the rho theta zed space
        
        self.rhos = self.graph.new_vertex_property('float')
        self.thetas = self.graph.new_vertex_property('float')
        self.zeds = self.graph.new_vertex_property('float')

        self.rhos.a = epithelium.rtz_cells['rho']
        self.thetas.a = epithelium.rtz_cells['theta']
        self.zeds.a = epithelium.rtz_cells['zed']

        rtzs = [self.rhos, self.thetas, self.zeds]
        self.rtz_pos = gt.group_vector_property(rtzs, value_type='float')

        self.vecinos = self.graph.new_vertex_property('vector<double>')
        self.rhos_faces = self.graph.new_vertex_property('vector<double>')
        self.thetas_faces = self.graph.new_vertex_property('vector<double>')
        self.zeds_faces = self.graph.new_vertex_property('vector<double>')


        self.compute_faces()

        self.rhos_voronoi = self.graph.new_vertex_property('vector<double>')
        self.thetas_voronoi = self.graph.new_vertex_property('vector<double>')
        self.zeds_voronoi = self.graph.new_vertex_property('vector<double>')


        

    def compute_faces(self, ndim=3):
        """
        for each cell, retrives the positions of
        the three summits of all the triangles
        with the current cell at one summit

        it goes through the neighbours by turning counter-clockwise
        around the vertex
        
        """

        for cell in self.graph.vertices():

            pos=self.rtz_pos[cell].a
            degree = self.degree(cell)

            assert pos.shape[0] == ndim

            #neighbours
            vecinos = np.array([self.rtz_pos[vecino].a
                                for vecino in cell.all_neighbours()])

            # in the (z, \sigma) coordinate system with it's origin
            # at the cell position, sort the neighbours counter-clockwise

            vecinos_local = vecinos - pos
            vecinos_zangle = np.arctan2(vecinos_local[:, 0] * vecinos_local[:, 1],
                                        vecinos_local[:, 2])
            indexes = np.argsort(vecinos_zangle)

            vecinos = vecinos.take(indexes, axis=0)
            self.vecinos[cell] = vecinos.flatten()

            faces = np.zeros((3, degree, ndim))
            faces[0, :, :] = pos
            faces[1, :, :] = vecinos
            faces[2, :-1, :] = vecinos[:-1]
            faces[2, -1, :] = vecinos[0]
            faces = faces.reshape((degree * 3, ndim))

            self.rhos_faces[cell] = faces[..., 0].ravel()
            self.thetas_faces[cell] = faces[..., 1].ravel()
            self.zeds_faces[cell] = faces[..., 2].ravel()

    def degree(self, cell):
        return cell.in_degree() + cell.out_degree()
            
    def compute_voronoi(self):
        for cell in self.graph.vertices():
            degree = self.degree(cell)

            sigmas = self.rhos_faces[cell].a * self.thetas_faces[cell].a
            faces_sigmas = sigmas.reshape((degree, 3))
            faces_zeds = self.zeds_faces[cell].a.reshape((degree, 3))
            

            faces_sigmas -= self.rhos[cell] * self.thetas[cell]
            faces_zeds  -= self.zeds[cell]

            transfo = lambda x, y : (x**2 + y**2) / 2 * (x + y)  
            sigmas_voronoi =  transfo(faces_sigmas[:, 1], faces_sigmas[:, 2]
                                      ) + self.rhos[cell] * self.thetas[cell]
            zeds_voronoi =  transfo(faces_zeds[:, 1], faces_zeds[:, 2]
                                    ) + self.zeds[cell]

            self.zeds_voronoi[cell] = zeds_voronoi.ravel()

            rhos = self.rhos_faces[cell].a.reshape((degree, 3)).mean(axis=1)
            self.rhos_voronoi[cell] = rhos.ravel()
            self.thetas_voronoi[cell] = sigmas_voronoi.ravel()/rhos.ravel()
        

    def draw(self, output='cell_graph.pdf', **kwargs):
        """
        Draws the graph with `gt.graph_draw`.
        """
        print 'graph view saved to %s' %output
        
        return gt.graph_draw(self.graph, self.sigmaz_pos,
                             output=output, **kwargs)
    
    def sfdp_draw(self):

        self.sfdp_pos = gt.graph_draw(self.graph,
                                      # self.rtz_pos,
                                      pos=gt.sfdp_layout(self.graph,
                                                         cooling_step=0.95,
                                                         epsilon=1e-3,
                                                         multilevel=True),
                                      output_size=(300,300),
                                      output="lattice_3d.pdf")



class AppicalJunctions():

    def __init__(self, epithelium):
        
        self.epithelium = epithelium
        
        

    def draw(self, output='junction_graph.pdf', **kwargs):
        """
        Draws the graph with `gt.graph_draw`.
        """
        return gt.graph_draw(self.graph, self.pos,
                             output=output)


    
    # for vecino0, vecino1, vecino2 in zip(cell0_vecinos:
    #     vrn = vornoi.add_vertex()  
    #     pos0 = cells.pos[cell0].a


    # for seed in input_graph.vertices():
    #     for neighbour_seed in v.all_neighbours():
            
        
