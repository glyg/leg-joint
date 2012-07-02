#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, time
import numpy as np
from numpy.random import normal, random_sample
import graph_tool.all as gt
from xml_handler import ParamTree
from scipy import weave

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = os.path.join(ROOT_DIR, 'default', 'params.xml')

        
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

        self.is_local = self.graph.new_vertex_property('bool')
        self.is_local.a[:] = 0

    def rtz_group(self):
        
        rtzs = [self.rhos, self.thetas, self.zeds]
        self.rtz_pos = gt.group_vector_property(rtzs, value_type='float')
        del rtzs
        self.calc_sigmas()
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
        sz_record['zed'] = self.zeds.a
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

    def periodic_theta(self, vertex, ref_theta):
        dtheta = self.thetas[vertex] - ref_theta
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
        # at the vertex position, sort the neighbours counter-clockwise

        self.calc_sigmas()
        for vertex in self.graph.vertices():
            zetas = []
            self.at_boundary[vertex]
            for vecino in vertex.all_neighbours():
                theta = self.periodic_theta(vecino, self.thetas[vertex])
                if theta != self.thetas[vecino] :
                    self.at_boundary[vertex] = 1
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
        output = os.path.join('drawings', output)
        pos = self.sz_pos
        pmap = gt.graph_draw(self.graph, self.sz_pos,
                             output=output, **kwargs)
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
        
        self.rhos.a += normal(0, noise_amplitude,
                              self.rhos.a.size)
        self.zeds.a += normal(0, noise_amplitude,
                              self.rhos.a.size)
        theta_noise = 2 * np.pi * noise_amplitude / self.rhos.a.mean()
        self.thetas.a += normal(0, theta_noise,
                                self.rhos.a.size)
        self.rtz_group()
        return self.rhos.a, self.thetas.a, self.zeds.a

    
class CellGraph(AbstractRTZGraph):
    '''
    
    '''
    def __init__(self, epithelium):
        
        self.epithelium = epithelium
        cutoff = self.epithelium.params['pos_cutoff']
        contractility0 = self.epithelium.params['contractility']        
        elasticity0 = self.epithelium.params['elasticity']        
        prefered_area0 =  self.epithelium.params['prefered_area']

        rtz = self.generate_rtz()
        self.generate_graph(rtz)
        rhos, thetas, zeds = rtz
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

        self.prefered_area =  self.graph.new_vertex_property('float')
        self.prefered_area.a[:] = prefered_area0

        for cell in self.graph.vertices():
            self.junctions_vertices[cell] = []
            self.junctions_edges[cell] = []



    def generate_graph(self, rtz):

        rhos, thetas, zeds = rtz
        
        num_vertices = rhos.shape[0]
        sigmas = rhos * thetas
        sigmazs = np.array([sigmas, zeds]).T
        sigmas_shift = rhos * ((thetas + np.pi) % (2 * np.pi))
        sigmazs_shift = np.array([sigmas_shift, zeds]).T

        #Graph instanciation
        t0 = time.time()
        graph1, pos1 = gt.triangulation(sigmazs, type='delaunay')
        graph2, pos2 = gt.triangulation(sigmazs_shift, type='delaunay')
        if not hasattr(self, 'graph'):
            self.graph = graph1.copy()
        else:
            self.rhos.a = rhos
            self.thetas.a = thetas
            self.zeds.a = zeds
        self.graph.clear_edges()
        for cell1, cell2 in zip(graph1.vertices(), graph2.vertices()):
            if cell1.out_degree() >= cell2.out_degree():
                for n1 in cell1.all_neighbours():
                    if self.graph.edge(cell1, n1) is None:
                        self.graph.add_edge(cell1, n1) 
            else:
                for n2 in cell2.all_neighbours():
                    if self.graph.edge(cell2, n2) is None:
                        self.graph.add_edge(cell2, n2)
        

        
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

        self.graph = gt.Graph(directed=False)
        line_tension0 = self.epithelium.params['line_tension']

        cutoff = self.epithelium.params['pos_cutoff']
        
        self.cells_vertices = self.graph.new_vertex_property('object')
        self.adjacent_cells = self.graph.new_edge_property('object')
        self.compute_voronoi()
        
        AbstractRTZGraph.__init__(self, self.raw_rtzs[:, 0],
                                  self.raw_rtzs[:, 1],
                                  self.raw_rtzs[:, 2],
                                  self.graph, cutoff)
        self.edge_lengths = self.graph.new_edge_property('float')
        self.line_tensions = self.graph.new_edge_property('float')
        self.line_tensions.a[:] = line_tension0
        epithelium.junctions = self

    def compute_voronoi(self):
        n_dropped = 0
        cutoff = self.epithelium.params['pos_cutoff']
        rtzs = []
        cells = self.epithelium.cells
        cells.order_neighbours()

        visited_cells = []
        self.graph.clear()
        for cell in cells.graph.vertices():
            cells.junctions_vertices[cell] = []
            cells.junctions_edges[cell] = []
        
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
                v0_theta = cells.periodic_theta(vecino0, cells.thetas[cell])
                v0_zed = cells.zeds[vecino0]
                
                v1_rho = cells.rhos[vecino1]
                v1_theta = cells.periodic_theta(vecino1, cells.thetas[cell])
                v1_zed = cells.zeds[vecino1]
                    
                v0_sz = [v0_rho * v0_theta, v0_zed]
                v1_sz = [v1_rho * v1_theta, v1_zed]
                sigma, zed = c_circumcircle(cell_sz, v0_sz,
                                            v1_sz, cutoff)
                if not np.isfinite(sigma):
                    n_dropped += 1
                    continue
                rho = (cell_rho + v0_rho + v1_rho) / 3.
                theta = (sigma / rho) % (2 * np.pi)

                j_vertex = self.graph.add_vertex()
                rtzs.append([rho, theta, zed])
                cells.junctions_vertices[cell].append(j_vertex)
                cells.junctions_vertices[vecino0].append(j_vertex)
                cells.junctions_vertices[vecino1].append(j_vertex)
                self.cells_vertices[j_vertex] = [cell, vecino0, vecino1]
            for vecino in vecinos:
                if vecino in visited_cells:
                    continue
                j_vert0 = None
                for jv in cells.junctions_vertices[cell]:
                    if jv in cells.junctions_vertices[vecino]:
                        if j_vert0 == None:
                            j_vert0 = jv
                        else:
                            j_edge = self.graph.add_edge(j_vert0, jv)
                            cells.junctions_edges[cell].append(j_edge)
                            cells.junctions_edges[vecino].append(j_edge)
                            self.adjacent_cells[j_edge] = (cell, vecino)
                            
        print n_dropped
        self.raw_rtzs = np.array(rtzs)

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
        yc = (x2*x2 + y2*y2 - 2*x2*xc)/(2*y2);
        }
    else if (y2*y2 < cutoff*cutoff
             && y1*y1 > cutoff*cutoff) 
        {
        xc = x2 / 2.;
        yc = (x1*x1 + y1*y1 - 2*x1*xc)/(2*y1);
        }
    else if (y1*y1 + y2*y2 < cutoff*cutoff)
        {
        if (x1*x1 + x2*x2 < cutoff*cutoff)
            {
            xc = x1 / 2.;
            yc = y1 / 2.;
            }
        else
            {
            xc = 0.;
            yc = 0.;
            }
        }
    else
       {
       double a1 = -x1/y1;
       double a2 = -x2/y2;
       double b1 = y1/2.;
       double b2 = y2/2.;
            
       xc = (b2 - b1) / (a1 - a2);
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
            return np.inf, np.inf
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

