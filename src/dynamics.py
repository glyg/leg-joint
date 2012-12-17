#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import graph_tool.all as gt
import numpy as np
from scipy import optimize, weave
from scipy.interpolate import splrep, splev


from objects import  AbstractRTZGraph, Cells, AppicalJunctions
from xml_handler import ParamTree
from utils import compute_distribution
import filters

EpitheliumFilters = filters.EpitheliumFilters

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = os.path.join(ROOT_DIR, 'default', 'params.xml')

# See [the tau manifesto](http://tauday.com/tau-manifesto)
tau = 2. * np.pi


class Epithelium(EpitheliumFilters, AbstractRTZGraph):
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
            self._init_grad_and_energy()
            # self.compute_bending()
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
        
    def _init_grad_and_energy(self):
        # Gradients amplitudes
        elastic_grad = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["elastic_grad"] = elastic_grad
        contractile_grad = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["contractile_grad"] = contractile_grad
        grad_radial = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["grad_radial"] = grad_radial

        # Gradients along sigma and zed
        grad_sigma = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["grad_sigma"] = grad_sigma
        grad_zed = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["grad_zed"] = grad_zed

        # Energy per cell
        energy_grad = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["energy_grad"] = energy_grad
        
    @property
    def grad_sigma(self):
        return self.graph.vertex_properties["grad_sigma"]
    @property
    def grad_zed(self):
        return self.graph.vertex_properties["grad_zed"]
    @property
    def elastic_grad(self):
        return self.graph.vertex_properties["elastic_grad"]
    @property
    def contractile_grad(self):
        return self.graph.vertex_properties["contractile_grad"]
    @property
    def grad_radial(self):
        return self.graph.vertex_properties["grad_radial"]
    @property
    def energy_grad(self):
        return self.graph.vertex_properties["energy_grad"]

    def compute_bending(self):
        pass
        
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
        if method == 'fmin_l_bfgs_b':
            output = optimize.fmin_l_bfgs_b(self.opt_energy,
                                            pos0.flatten(),
                                            fprime=self.opt_gradient,
                                            approx_grad=1,
                                            bounds=bounds.flatten(),
                                            factr=1e10,
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
                                           callback=self.opt_callback)
            except:
                self.graph.set_vertex_filter(vfilt)
                self.set_new_pos(pos0)
                self.graph.set_vertex_filter(None)
                output = 0
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

    @filters.local
    def check_local_grad(self, pos0):
        if self.__verbose__: print "Checking gradient"
        chk_out = optimize.check_grad(self.opt_energy,
                                      self.opt_gradient,
                                      pos0.flatten())
        return chk_out

        
    def opt_energy(self, sz_pos):
        """
        After setting the sigma, zed position to sz_pos,
        computes the energy over the graph as filtered
        by vfilt and efilt vertex and edge filters (respectively)
        """
        # Position setting
        self.set_new_pos(sz_pos)
        self.update_apical_geom()
        energy = self.calc_energy()
        return energy

    def opt_gradient(self, sz_pos):
        """
        After setting the sigma, zed position to sz_pos,
        computes the gradient over the graph as filtered
        by vfilt and efilt vertex and edge filters (respectively)
        """
        # # position setting
        self.set_new_pos(sz_pos)
        self.update_apical_geom()
        self.update_gradient()
        gradient = self.gradient_array()
        return gradient

    def opt_callback(self, sz_pos):
        """ Call back for the optimization """
        self.update_apical_geom()
        self.update_gradient()
        self.periodic_boundary_condition()

    def calc_energy(self):
        """ Computes the apical energy on the filtered epithelium """
        cells_term, denominator = self.calc_cells_energy()
        junction_term = self.calc_junctions_energy()
        total_energy = cells_term + junction_term
        return total_energy / denominator
        
    @filters.j_edges_in
    def calc_junctions_energy(self):
        junctions_energy = self.junctions.line_tensions.fa\
                           * self.edge_lengths.fa
        return junctions_energy.sum()
        
    @filters.cells_in
    def calc_cells_energy(self):
        elastic_term = 0.5 * self.cells.elasticities.fa * (
            self.cells.areas.fa - self.cells.prefered_area.fa)**2
        contractile_term = 0.5 * self.cells.contractilities.fa * \
                           self.cells.perimeters.fa**2
        num_cells = self.graph.num_vertices() #elastic_term.size
        prefered_area0 =  self.params['prefered_area']
        elasticity0 = self.params['elasticity']
        denominator = num_cells * elasticity0 * prefered_area0**2
        cells_energy = elastic_term + contractile_term
        return cells_energy.sum(), denominator
        
    @filters.active
    def gradient_array(self):
        prefered_area0 =  self.params['prefered_area']
        elasticity0 = self.params['elasticity']
        norm_factor = prefered_area0 * elasticity0
        gradient = np.zeros(self.graph.num_vertices() * 2)
        if self.__verbose__ : print 'Gradient shape: %s' % gradient.shape
        gradient[::2] = self.grad_sigma.fa / norm_factor
        gradient[1::2] = self.grad_zed.fa / norm_factor
        return gradient

    def update_gradient(self):
        self.update_cells_grad()
        self.update_junctions_grad()

    def update_junctions_grad(self):
        # Junction edges
        if self.__verbose__ :
            num_cells = self.is_cell_vert.fa.sum()
            num_jverts = self.graph.num_vertices() - num_cells
            num_edges = self.is_junction_edge.fa.sum()
            print ('''Updating gradient for %i cells,'''
                   '''%i junctions vertices and  %i junctions edges'''
                   % (num_cells, num_jverts, num_edges))
        self.grad_sigma.fa = 0.
        self.grad_zed.fa = 0.
        self.grad_radial.fa = 0.
        u_rhosigma = self.dsigmas.copy()
        u_rhosigma.fa = self.u_dsigmas.fa * np.sin(self.dthetas.fa / 2)
        for edge in self.junctions:
            tension = self.junctions.line_tensions[edge]
            j_src, j_trgt = edge.source(), edge.target()
            u_sg = self.u_dsigmas[edge]
            u_zg = self.u_dzeds[edge]
            u_rs = u_rhosigma[edge]
                            
            for cell in self.adjacent_cells(edge):
                perp_sigma, perp_zed =  self.outward_uvect(cell, edge)
                el_grad = self.elastic_grad[cell]
                ctr_grad = self.contractile_grad[cell]
                self.grad_sigma[j_src] += el_grad * perp_sigma\
                                          - ctr_grad * u_sg   
                self.grad_sigma[j_trgt] += el_grad * perp_sigma\
                                           + ctr_grad * u_sg
                self.grad_zed[j_src] += el_grad * perp_zed\
                                        - ctr_grad * u_zg
                self.grad_zed[j_trgt] += el_grad * perp_zed\
                                        + ctr_grad * u_zg

                self.grad_radial[j_src] += (el_grad  + ctr_grad ) * u_rs

            self.grad_sigma[j_src] += - tension * u_sg
            self.grad_zed[j_src] += - tension * u_zg
            self.grad_sigma[j_trgt] += tension * u_sg
            self.grad_zed[j_trgt] += tension * u_zg
            # self.grad_radial[j_src] += tension[edge] * u_rhosigma[edge]
            # self.grad_radial[j_trgt] += tension[edge] * u_rhosigma[edge]

    @filters.cells_in
    def update_cells_grad(self):
        # Cell vertices
        self.elastic_grad.fa =  self.cells.elasticities.fa \
                                * (self.cells.areas.fa -
                                   self.cells.prefered_area.fa )
        self.contractile_grad.fa =  self.cells.contractilities.fa \
                                    * self.cells.perimeters.fa
        
    def update_apical_geom(self):
        # Edges
        if self.__verbose__: print ('Geometry update on %i edges'
                                    % self.graph.num_edges())
        self.update_deltas()
        self.update_edge_lengths()
        # Cells
        if self.__verbose__: print ('Cells geometry update on %i vertices'
                                    % self.graph.num_vertices())
        for cell in self.cells:
            j_edges = self.cell_junctions(cell)
            self._one_cell_geom(cell, j_edges)
    
    def _one_cell_geom(self, cell, j_edges):
        """
        The area is approximated as the sum
        of the areas of the triangles
        formed by the cell position and each junction
        """
        area = 0.
        perimeter = 0.
        if len(j_edges) < 3 and self.is_alive[cell]:
            if self.__verbose__: print ('''Two edges ain't enough to compute '''
                                 '''area for cell %s''' % cell)
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
        self.cells.areas[cell] = area
        self.cells.perimeters[cell] = perimeter

        ##  Update cell position 
        j_sz = np.array([[self.sigmas[jv], self.zeds[jv]]
                         for jv in cell.out_neighbours()])
        ### set z
        self.zeds[cell] = j_sz[:,1].mean()

        ### set periodic sigma
        raw_dsigma = j_sz[:,0] - self.sigmas[cell]
        period = tau * self.rhos[cell] 
        pbc_sigma = j_sz[:,0]
        pbc_sigma[raw_dsigma <=- period/2] += period
        pbc_sigma[raw_dsigma >= period/2] -= period
        self.sigmas[cell] = pbc_sigma.mean()

    @filters.active
    def set_new_pos(self, new_sz_pos):
        new_sz_pos = new_sz_pos.flatten()
        assert len(new_sz_pos) / 2 == self.graph.num_vertices()
        self.sigmas.fa = new_sz_pos[::2]
        self.zeds.fa = new_sz_pos[1::2]

    @filters.cells_in
    def anisotropic_relax(self):
        eq_area = self.cells.areas.fa.mean()
        nz = self.params['n_zeds']
        ns = self.params['n_sigmas']
        
        tck_area_vs_zed, H = compute_distribution(self.zeds.fa,
                                                  self.cells.areas.fa,
                                                  bins=(ns, nz), smth=3)
        z_dilation = np.sqrt(splev(self.zeds.a, tck_area_vs_zed) / eq_area)
        self.rhos.a /= z_dilation
        self.zeds.a /= z_dilation
        self.sigmas.a = self.thetas.a * self.rhos.a

        
    def isotropic_relax(self):
        gamma = self.paramtree.relative_dic['contractility']
        lbda = self.paramtree.relative_dic['line_tension']
        good, report = self.check_phase_space(gamma, lbda)
        if not good:
            raise ValueError("Invalid values for "
                             "the average contractility and elasticity \n"
                             +report)
        delta0 = 0.9
        delta_o,  = optimize.fsolve(self.isotropic_grad, delta0,
                                    args=(gamma, lbda))

        self.set_vertex_state([(self.is_cell_vert, False),
                               (self.is_alive, False)])
        area = self.cells.areas.fa.mean()
        area0 = self.params['prefered_area']
        area_opt = delta_o**2 * area0
        correction = np.sqrt(area_opt / area)
        print "Scaling all the distances by a factor %.3f" %correction
        self.delta_o = delta_o
        self.ground_energy = self.isotropic_energy(delta_o, gamma, lbda)
        self.scale(correction)
        self.set_vertex_state()

    def isotropic_grad(self, delta, gamma, lbda):
        mu = 6 * np.sqrt(2. / (3 * np.sqrt(3)))
        grad = 4 * delta**3 + (2 * gamma * mu**2 - 4) * delta + lbda * mu
        return grad

    def isotropic_energy(self, delta, gamma, lbda):
        """
        Computes the theoritical energy per cell for the given
        parameters.
        """
        mu = 6 * np.sqrt(2. / (3 * np.sqrt(3)))
        elasticity = (delta**2 - 1**2)**2 / 2.
        contractility = gamma * mu**2 * delta**2 / 2.
        tension = lbda * mu * delta / 2.
        energy = elasticity + contractility + tension
        return energy

    def outward_uvect(self, cell, j_edge):
        """
        Returns the (sigma, zed) coordinates of the unitary vector
        perpendicular to the junction edge `j_edge` and pointing
        outward the cell `cell`
        """
        if not cell in self.adjacent_cells(j_edge):
            raise AttributeError("Cell not adjacent to junction")
        perp_sigma = - self.u_dzeds[j_edge]
        perp_zed = self.u_dsigmas[j_edge]
        ctoj1 = self.graph.edge(cell, j_edge.source())
        ctoj2 = self.graph.edge(cell, j_edge.target())
        median_dsigma = (self.dsigmas[ctoj1] + self.dsigmas[ctoj2])
        median_dzed = (self.dzeds[ctoj1] + self.dzeds[ctoj2])
        if perp_sigma * median_dsigma < 0:
            perp_sigma *= -1
        if perp_zed * median_dzed < 0:
            perp_zed *= -1
        return perp_sigma, perp_zed

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

    def resolve_small_edges(self, threshold=5e-2, vfilt=None, efilt=None):
        # Collapse 3 sided cells
        if vfilt == None:
            vfilt_3sides = self.is_cell_vert.copy()
        else:
            vfilt_3sides = vfilt.copy()
        self.graph.set_edge_filter(self.is_ctoj_edge,
                                   inverted=True)
        degree_pm =  self.graph.degree_property_map('out').a
        vfilt_3sides.a *= [degree_pm == 3][0] * self.is_alive.a
        self.graph.set_vertex_filter(vfilt_3sides)
        cells = [cell for cell in self.graph.vertices()]
        self.graph.set_vertex_filter(None)
        self.graph.set_edge_filter(None)
        new_jvs = [self.type3_transition(cell, threshold)
                   for cell in cells]
        self.update_apical_geom(vfilt)
        # Type 1 transitions
        if efilt == None:
            efilt_je = self.is_junction_edge.copy()
        else:
            efilt_je = efilt.copy()
            efilt_je.a *= self.is_junction_edge.a
        efilt_je.a *= [self.edge_lengths.a < threshold][0]
        self.graph.set_edge_filter(efilt_je)
        visited_cells = []
        old_local_edge = self.is_local_edge.copy()
        old_local_vert = self.is_local_vert.copy()
        for edge in self.graph.edges():
            self.set_local_mask(None)
            cell0, cell1 = self.adjacent_cells(edge)
            if cell0 in visited_cells or cell1 in visited_cells:
                continue
            visited_cells.extend([cell0, cell1])
            if (len(self.cell_junctions[cell0]) < 4) or (
                len(self.cell_junctions[cell1]) < 4):
                continue
            print 'Type 1 transition'
            self.type1_transition((cell0, cell1))
            
            pos0, pos1 = self.find_energy_min()
        self.graph.set_edge_filter(None)
        
    

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
            if self.__verbose__: print ("Warning: previous %s to %s "
                                 "edge is re-created." % (str(j_verta),
                                                          str(j_vertb)))
            self.graph.remove_edge(j_edgeab)
        j_edgeab = self.graph.add_edge(j_verta, j_vertb)
        self.is_junction_edge[j_edgeab] = 1
        self.is_new_edge[j_edgeab] = 1
        self.is_ctoj_edge[j_edgeab] = 0
        line_tension0 = self.params['line_tension']
        self.junctions.line_tensions[j_edgeab] = line_tension0

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
                print ("Warning: previous cell %s \n"
                       "to junction vertex %s edge is re-created.",
                       ) % (str(cell0), str(j_vertb))
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

        
    ## Junction edges related  functions 
    def adjacent_cells(self, j_edge):
        jv0 = j_edge.source()
        jv1 = j_edge.target()
        cells_a = [cell for cell in jv0.in_neighbours()
                   if self.is_cell_vert[cell]]
        cells_b = [cell for cell in jv1.in_neighbours()
                   if self.is_cell_vert[cell]]
        common_cells = [cell for cell in cells_a if cell in cells_b]
        return common_cells

    ## Cell vertices topological functions
    def cell_junctions(self, cell):
        jvs = [jv for jv in cell.out_neighbours()]
        j_edges = []
        for jv0 in jvs:
            for jv1 in jvs:
                if jv1 == jv0 : continue
                e = self.graph.edge(jv0, jv1)
                if e is not None: j_edges.append(e)
        return j_edges
        
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

