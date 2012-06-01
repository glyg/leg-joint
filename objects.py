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

        n_zeds = int(4 * rho_0 / lambda_0)
        n_zeds -= n_zeds % 3
        delta_theta = 2 * np.arcsin(lambda_0 / (2 * rho_0))
        n_thetas = int(2 * np.pi / delta_theta)
        n_thetas -= n_thetas % 3

        self.n_zeds = n_zeds
        self.n_thetas = n_thetas

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
        z_min, z_max = -2 * rho_0, rho_0

        #Each vertex of `cell.graph` represents a vertex
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
        

    def compute_triangles(self):
        """
        computes all the triangles for each cell
        the triangle
        the associated array 
        """
        self.triangles = self.graph.new_vertex_property('vector<double>')

        self.rhos_voronoi = self.graph.new_vertex_property('float')
        self.thetas_voronoi = self.graph.new_vertex_property('float')
        self.sigmas_voronoi = self.graph.new_vertex_property('float')
        self.zeds_voronoi = self.graph.new_vertex_property('float')

        for cell in self.graph.vertices():
            pos = self.rtz_pos[cell].a
            vecino_pos = [pos]
            triangle_list = []
            for vecino in cell.all_neighbours():
                vecino_pos.append(self.rtz_pos[vecino].a)
            for vp0, vp1 in zip(vecino_pos[:-1], vecino_pos[1:]):
                triangle_list.append([pos, vp0, vp1])


            triangle = np.array(triangle_list)
            degree = cell.in_degree() + cell.out_degree()
            if triangle.shape == (degree, 3, pos.shape[0]):

                self.triangles[cell] = triangle.flatten()

                triangle_sigma = triangle[:, :, 0] * triangle[:, :, 1]
                self.sigmas_voronoi[cell] =  triangle_sigma.mean(axis=1)[0]

                zeds_voronoi = triangle[:, :, 2]
                self.zeds_voronoi[cell] = zeds_voronoi.mean(axis=1)[0]
            else:
                # TODO : invalidate the cell
                print 'gloup'

            
            
        
    def draw(self, output='cell_graph.pdf', **kwargs):
        """
        Draws the graph with `gt.graph_draw`.
        """
        print 'graph view saved to %s' %output
        
        return gt.graph_draw(self.graph, self.sigmaz_pos,
                             output=output, **kwargs)
    
    def sfdp_draw(self):

        self.sfdp_pos = gt.graph_draw(self.graph, 
                                      pos=gt.sfdp_layout(apical_junction_geom,
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


def voronoi_diagram(input_graph, input_posmap):
    
    input_shape = input_posmap.a.shape
    input_size = input_graph.num_vertices()
   
    voronoi_graph = gt.Graph(directed='False')
    voronoi_pos = voronoi.new_vertex_property('vector<double>')
    vrn0 = vornoi.add_vertex()
    
    rand_idx = np.random.randint()
    cell0 = input_graph.vertex(rand_idx)

    
    
    cell0_vecinos = [cells.pos[vecino].a
                     for vecino in cell0.all_neighbours()]
    cell0_triangles = [(t0, t1, t2)
                       for t0, t1, t2 in zip((cell0_vecinos[: -2],
                                              cell0_vecinos[1: -1],
                                              cell0_vecinos[2: ]))]



    vornoi_triangle0 = np.array([cells.pos[cell0].a,
                                 cell0_vecinos[0].a,
                                 cell0_vecinos[1].a])
    voronoi_pos[vrn0] = vornoi_triangle0.sum(axis=1)
    
    # for vecino0, vecino1, vecino2 in zip(cell0_vecinos:
    #     vrn = vornoi.add_vertex()  
    #     pos0 = cells.pos[cell0].a


    # for seed in input_graph.vertices():
    #     for neighbour_seed in v.all_neighbours():
            
        
