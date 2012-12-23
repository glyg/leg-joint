#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import graph_tool.all as gt
import numpy as np
from scipy import optimize, weave


from objects import  AbstractRTZGraph, Cells, AppicalJunctions
from xml_handler import ParamTree
from energies import EpitheliumEnergies
import filters

EpitheliumFilters = filters.EpitheliumFilters

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = os.path.join(ROOT_DIR, 'default', 'params.xml')

# See [the tau manifesto](http://tauday.com/tau-manifesto)
tau = 2. * np.pi


class Epithelium(EpitheliumFilters,
                 AbstractRTZGraph,
                 EpitheliumEnergies):
    """
    The ::class::`Epithelium` is the container for all the simulation.
    
    """
    def __init__(self, graphXMLfile=None, paramtree=None, paramfile=PARAMFILE):
        """
        
        """
        # Parametrisation
        if paramtree == None:
            self.paramtree = ParamTree(paramfile)
        else:
            self.paramtree = paramtree
        self.params = self.paramtree.absolute_dic

        # Graph instanciation
        if graphXMLfile is None:
            self.graph = gt.Graph(directed=True)
            self.new = True
        else:
            self.graph = gt.load_graph(graphXMLfile)
            self.new = False
            self.xmlfname = graphXMLfile

        self.__verbose__ = False
        EpitheliumFilters.__init__(self)
        # All the geometrical properties are packed here
        AbstractRTZGraph.__init__(self)
        # Cells and Junctions subgraphs initialisation
        print 'Initial cells'
        self.cells = Cells(self)
        print 'Initial junctions'
        self.junctions = AppicalJunctions(self)
        if self.new:
            self.is_alive.a = 1
            # Remove cell to cell edges after graph construction
            efilt = self.is_ctoj_edge.copy()
            efilt.a += self.is_junction_edge.a
            self.graph.set_edge_filter(efilt)
            self.graph.purge_edges()
            self.set_vertex_state()
            self.set_edge_state()
        self.cells.update_junctions()
        self.junctions.update_adjacent()
        
        EpitheliumEnergies.__init__(self)    
        if self.__verbose__: print 'isotropic relaxation'
        self.isotropic_relax()
        if self.__verbose__: print 'Periodic boundary'
        self.periodic_boundary_condition()
        if self.__verbose__: print 'Update appical'
        self.update_apical_geom()
        if self.__verbose__: print 'Update gradient'
        self.update_gradient()

    def __str__(self):
        num_cells = self.is_cell_vert.a.sum()
        num_edges = self.is_junction_edge.a.sum()
        str1 = ['Epithelium with %i cells and %i junction edges' % (num_cells,
                                                                    num_edges)]
        str1.append('Vertex Properties:\n'
                    '==================')
        for key in sorted(self.graph.vertex_properties.keys()):
            str1.append('    * %s' % key)
        str1.append('Edge Properties:\n'
                    '================')
        for key in sorted(self.graph.edge_properties.keys()):
            str1.append('    * %s' % key)
        return '\n'.join(str1)
        
        
    @filters.active
    def precondition(self):
        if self.at_boundary.fa.sum() > 0:
            self.rotate(np.pi)
            self.current_angle = np.pi
            print 'rotated'
            
        pos0 = np.vstack([self.sigmas.fa,
                          self.zeds.fa]).T.flatten()
        max_dsigma =  2 * self.edge_lengths.fa.mean()
        max_dzed = max_dsigma
        if self.__verbose__ : print ('Initial postion `pos0` has shape %s'
                              %str(pos0.shape))
        if self.__verbose__ : print ('Max displacement amplitude for : %.3f'
                             % max_dsigma)
        bounds = np.zeros((pos0.shape[0],),
                          dtype=[('min', np.float32),
                                 ('max', np.float32)])
        if self.__verbose__: print ('bounds array has shape: %s'
                             % str(bounds.shape))
        s_bounds = np.vstack((self.sigmas.fa - max_dsigma,
                             self.sigmas.fa + max_dsigma)).T
        z_bounds = np.vstack((self.zeds.fa - max_dzed,
                             self.zeds.fa + max_dzed)).T
        for n in range(pos0.shape[0]/2):
            bounds[2 * n] = (s_bounds[n, 0], s_bounds[n, 1])
            bounds[2 * n + 1] = (z_bounds[n, 0], z_bounds[n, 1])
        return pos0, bounds

    @filters.local
    def find_energy_min(self, method='fmin_l_bfgs_b', tol=1e-8):
        '''
        Performs the energy minimisation
        '''
        pos0, bounds = self.precondition()
        output = 0
        if method == 'fmin_l_bfgs_b':
            # On my box, machine precision is 10^-19, so
            ## I set factr to 1e14 to avoid too long computation
            output = optimize.fmin_l_bfgs_b(self.opt_energy,
                                            pos0.flatten(),
                                            fprime=self.opt_gradient,
                                            approx_grad=1,
                                            bounds=bounds.flatten(),
                                            factr=1e14,
                                            m=10,
                                            pgtol=tol,
                                            epsilon=1e-8,
                                            iprint=1,
                                            maxfun=150,
                                            disp=None)
        elif method=='fmin':
            output = optimize.fmin(self.opt_energy,
                                   pos0.flatten(),
                                   ftol=tol, xtol=0.01,
                                   callback=self.opt_callback)
        elif method=='fmin_ncg':
            try:
                output = optimize.fmin_ncg(self.opt_energy,
                                           pos0.flatten(),
                                           fprime=self.opt_gradient,
                                           avextol=tol,
                                           retall=True,
                                           maxiter=100,
                                           callback=self.opt_callback)
            except:
                self.set_new_pos(pos0)
                self.graph.set_vertex_filter(None)
                output = 0
        elif method=='fmin_tnc':
            output = optimize.fmin_tnc(self.opt_energy,
                                       pos0.flatten(),
                                       fprime=self.opt_gradient,
                                       pgtol=tol,
                                       bounds=bounds,
                                       maxCGit=0,
                                       disp=5)
        elif method=='fmin_bfgs':
            output = optimize.fmin_bfgs(self.opt_energy,
                                        pos0.flatten(),
                                        fprime=self.opt_gradient,
                                        gtol=tol,
                                        norm=np.inf,
                                        retall=1,
                                        callback=self.opt_callback)
        if not -1e-8 < self.current_angle < 1e-8:
            self.rotate(-self.current_angle)
            self.current_angle = 0.
        return pos0, output
        
    def update_apical_geom(self):
        # Edges
        if self.__verbose__: print ('Geometry update on %i edges'
                                    % self.graph.num_edges())
        self.update_deltas()
        self.update_edge_lengths()
        # Cells
        if self.__verbose__: print ('Cells geometry update on %i vertices'
                                    % self.graph.num_vertices())
        rho_lumen = self.params['rho_lumen']
        for cell in self.cells:
            self._one_cell_geom(cell)
        self.thetas.fa = self.sigmas.fa / self.rhos.fa
        self.cells.vols.fa = self.cells.areas.fa * (self.rhos.fa - rho_lumen)
            
    def _one_cell_geom(self, cell):
        """
        The area is approximated as the sum
        of the areas of the triangles
        formed by the cell position and each junction
        """
        if not self.is_alive[cell]: return
        area = 0.
        perimeter = 0.
        j_edges = self.cells.junctions[cell]
        if self.__verbose__:
            print(''' Cell %s has %i junction edges'''
                  % (cell, len(j_edges)))
        if len(j_edges) < 3 and self.is_alive[cell]:
            if self.__verbose__:
                print('''Two edges ain't enough to compute
                      area for cell %s''' % cell)
            self.cells.areas[cell] = self.cells.prefered_area[cell]
            self.cells.perimeters[cell] = 0.
            self.is_alive[cell] = 0
            return
        for j_edge in j_edges:
            perimeter += self.edge_lengths[j_edge]
            ctoj0 = self.graph.edge(cell, j_edge.source())
            ctoj1 = self.graph.edge(cell, j_edge.target())
            area += np.abs(self.dsigmas[ctoj0] * self.dzeds[ctoj1]
                           - self.dsigmas[ctoj1] * self.dzeds[ctoj0])/2.
            self.zeds[cell] += self.zeds[j_edge.target()]
            self.rhos[cell] += self.rhos[j_edge.target()]

        self.cells.areas[cell] = area
        self.cells.perimeters[cell] = perimeter

        ##  Update cell position 
        j_rsz = np.array([[self.rhos[jv], self.sigmas[jv], self.zeds[jv]]
                          for jv in cell.out_neighbours()])
        ### set z and rho
        self.zeds[cell] = j_rsz[:,2].mean()
        self.rhos[cell] = j_rsz[:,0].mean()
        
        ### set periodic sigma
        raw_dsigma = j_rsz[:,1] - self.sigmas[cell]
        period = tau * self.rhos[cell] 
        pbc_sigma = j_rsz[:,1]
        pbc_sigma[raw_dsigma <= -period/2] += period
        pbc_sigma[raw_dsigma > period/2] -= period
        self.sigmas[cell] = pbc_sigma.mean()
        
    @filters.active
    def set_new_pos(self, new_sz_pos):
        new_sz_pos = new_sz_pos.flatten()
        assert len(new_sz_pos) / 2 == self.graph.num_vertices()
        self.sigmas.fa = new_sz_pos[::2]
        self.zeds.fa = new_sz_pos[1::2]
        self.thetas.fa = self.sigmas.fa / self.rhos.fa

    @filters.active
    def set_new_rhos(self, rhos):
        self.rhos.fa = rhos
        
    def reset_topology(self):
        self.junctions.update_adjacent()
        self.cells.update_junctions()
        self.update_apical_geom()
        self.update_gradient()
        
    def outward_uvect(self, cell, j_edge):
        """
        Returns the (sigma, zed) coordinates of the unitary vector
        perpendicular to the junction edge `j_edge` and pointing
        outward the cell `cell`
        """
        edge_usigma = self.u_dsigmas[j_edge]
        edge_uzed = self.u_dzeds[j_edge]
        ctoj = self.graph.edge(cell, j_edge.source())
        cj_sigma = self.u_dsigmas[ctoj]
        cj_zed = self.u_dzeds[ctoj]
        cross_prod = cj_sigma * edge_uzed - edge_usigma * cj_zed
        if cross_prod < 0:
            return -edge_uzed, edge_usigma
        return edge_uzed, -edge_usigma

    def check_phase_space(self, gamma, lbda):
        # See the energies.pynb notebook for the derivation of this:
        mu = 6 * np.sqrt(2. / (3 * np.sqrt(3)))
        if (gamma < - lbda / (2 * mu)):
            report= ("Contractility is too low,"
                     "Soft network not supported")
            return False, report
        if 2 * gamma * mu**2 > 4:
            lambda_max = 0.
        else:
            lambda_max = ((4 - 2 * gamma * mu**2) / 3.)**(3./2.) / mu
        if lbda > lambda_max:
            report = ("Invalid value for the line tension: "
                      "it should be lower than %.2f "
                      "for a contractility of %.2f "
                    % (lambda_max, gamma))
            return False, report
        return True, 'ok!'

    def add_junction(self, j_verta, j_vertb, cell0, cell1):
        ##### TODO: This block should go in a decorator
        valid = np.array([obj.is_valid() for obj in
                          (cell0, j_verta, j_vertb)])
        if not valid.all():
            raise ValueError("invalid elements in the argument list"
                             "with cell0 => %s"
                             "     vertex a => %s "
                             "     vertex b => %s " % (str(v) for v in valid)
                             )
        ####
        j_edgeab = self.graph.edge(j_verta, j_vertb)
        if j_edgeab is not None:
            if self.__verbose__:
                print ('''Warning: previous %s to %s 
                       edge is re-created.'''
                       % (str(j_verta), str(j_vertb)))
            self.graph.remove_edge(j_edgeab)
        j_edgeab = self.graph.add_edge(j_verta, j_vertb)
        self.is_junction_edge[j_edgeab] = 1
        self.is_new_edge[j_edgeab] = 1
        self.is_ctoj_edge[j_edgeab] = 0
        line_tension0 = self.params['line_tension']
        self.junctions.line_tensions[j_edgeab] = line_tension0
        self.junctions.adjacent_cells[j_edgeab] = cell0, cell1
        if self.cells.junctions[cell0] is None:
            self.cells.junctions[cell0] = [j_edgeab,]
        else:
            self.cells.junctions[cell0].append(j_edgeab)

        if self.cells.junctions[cell1] is None:
            self.cells.junctions[cell1] = [j_edgeab,]
        else:
            self.cells.junctions[cell1].append(j_edgeab)

        ctoj_0a = self.graph.edge(cell0, j_verta)
        if ctoj_0a is not None:
            # print ("Warning: previous cell %s "
            #        "to junction vertex %s edge is re-created."
            #        ) % (str(cell0), str(j_verta))
            self.graph.remove_edge(ctoj_0a)
        ctoj_0a = self.graph.add_edge(cell0, j_verta)
        self.is_junction_edge[ctoj_0a] = 0
        self.is_ctoj_edge[ctoj_0a] = 1
        self.is_new_edge[ctoj_0a] = 1
        
        ctoj_0b = self.graph.edge(cell0, j_vertb)
        if ctoj_0b is not None:
            if self.__verbose__:
                print ('''Warning: previous cell %s 
                        to junction vertex %s edge is re-created.'''
                       % (str(cell0), str(j_vertb)))
            self.graph.remove_edge(ctoj_0b)
        ctoj_0b = self.graph.add_edge(cell0, j_vertb)
        self.is_junction_edge[ctoj_0b] = 0
        self.is_ctoj_edge[ctoj_0b] = 1
        self.is_new_edge[ctoj_0b] = 1

        if cell1 is not None:
            ctoj_1a = self.graph.edge(cell1, j_verta)
            if ctoj_1a is not None:
                # print ("Warning: previous cell %s "
                #        "to junction vertex %s edge is re-created."
                #        ) % (str(cell1), str(j_verta))
                self.graph.remove_edge(ctoj_1a)
            ctoj_1a = self.graph.add_edge(cell1, j_verta)
            self.is_junction_edge[ctoj_1a] = 0
            self.is_ctoj_edge[ctoj_1a] = 1
            self.is_new_edge[ctoj_1a] = 1

            ctoj_1b = self.graph.edge(cell1, j_vertb)
            if ctoj_1b is not None:
                # print ("Warning: previous cell %s "
                #        "to junction vertex %s edge is re-created."
                #        ) % (str(cell1), str(j_vertb))
                self.graph.remove_edge(ctoj_1b)
            ctoj_1b = self.graph.add_edge(cell1, j_vertb)
            self.is_junction_edge[ctoj_1b] = 0
            self.is_ctoj_edge[ctoj_1b] = 1
            self.is_new_edge[ctoj_1b] = 1
        return j_verta, j_vertb, cell0, cell1

    def remove_junction(self, j_verta, j_vertb, cell0, cell1):
        #This block should go in a decorator
        valid = np.array([element.is_valid() for element in
                          (cell0, j_verta, j_vertb)])
        if not valid.all():
            raise ValueError("invalid elements in the argument list"
                             "with cell0 => %s"
                             "     vertex a => %s "
                             "     vertex b => %s " % (str(v) for v in valid))
        ####
        j_edgeab = self.graph.edge(j_verta, j_vertb)
        if j_edgeab is None:
            j_edgeab = self.graph.edge(j_vertb, j_verta)
        if j_edgeab is None:
            print "Warning: junction from %s to %s doesn't exist" % (
                str(j_edgeab.source()), str(j_edgeab.target()))
            return
        self.cells.junctions[cell0].remove(j_edgeab)
        self.cells.junctions[cell1].remove(j_edgeab)
        self.junctions.adjacent_cells[j_edgeab] = []
        self.graph.remove_edge(j_edgeab)
        ctoj_0a = self.graph.edge(cell0, j_verta)
        if ctoj_0a is not None:
            self.graph.remove_edge(ctoj_0a)
        ctoj_0b = self.graph.edge(cell0, j_vertb)
        if ctoj_0b is not None:
            self.graph.remove_edge(ctoj_0b)
        if cell1 is not None:
            ctoj_1a = self.graph.edge(cell1, j_verta)
            if ctoj_1a is not None:
                self.graph.remove_edge(ctoj_1a)
            ctoj_1b = self.graph.edge(cell1, j_vertb)
            if ctoj_1b is not None:
                self.graph.remove_edge(ctoj_1b)

    @filters.j_edges_in
    def merge_j_verts(self, jv0, jv1):
        vertex_trash = []
        edge_trash = []
        je = self.any_edge(jv0, jv1)
        if je is None:
            raise ValueError('Can only merge connected edges')
            
        edge_trash.append(je)
        for vert in jv1.out_neighbours():
            old_edge = self.graph.edge(jv1, vert)
            if vert != jv0:
                new_edge = self.graph.add_edge(jv0, vert)
            for prop in self.graph.edge_properties.values():
                prop[new_edge] = prop[old_edge]
            edge_trash.append(old_edge)
        for vert in jv1.in_neighbours():
            old_edge = self.graph.edge(vert, jv1)
            if vert != jv0:
                new_edge = self.graph.add_edge(vert, jv0)
            for prop in self.graph.edge_properties.values():
                prop[new_edge] = prop[old_edge]
            edge_trash.append(old_edge)
        vertex_trash.append(jv1)
        return vertex_trash, edge_trash
        
        
def triangle_geometry(sz0, sz1, sz2):
    c_code = """
    double s0 = sz0[0];
    double z0 = sz0[1];
    double s1 = sz1[0];
    double z1 = sz1[1];
    double s2 = sz2[0];
    double z2 = sz2[1];


    double d01 = sqrt((s0-s1) * (s0-s1) + (z1-z0) * (z1-z0));
    double d02 = sqrt((s0-s2) * (s0-s2) + (z2-z0) * (z2-z0));
    double d12 = sqrt((s1-s2) * (s1-s2) + (z2-z1) * (z2-z1));
    double area012 = fabs((s1-s0) * (z2-z0) - (s2-s0) * (z1-z0));

    py::tuple results(4);
    results[0] = d01;
    results[1] = d02;
    results[2] = d12;
    results[3] = area012;
    return_val = results;
    
    """
    return weave.inline(c_code,
                        arg_names=['sz0', 'sz1', 'sz2'],
                        headers=['<math.h>'])

