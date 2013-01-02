#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

#from utils import compute_distribution
#from scipy.interpolate import splev
from scipy import optimize
import filters

class Dynamics(object):
    '''
    Container for the various gradients and energy
    aspects of the model
    '''
    def __init__(self):
        if self.new:
            # Gradients amplitudes
            elastic_grad = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["elastic_grad"] = elastic_grad
            contractile_grad = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["contractile_grad"] = contractile_grad
            volume_grad = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["volume_grad"] = volume_grad
            # Gradients along sigma, zed and rho
            grad_sigma = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["grad_sigma"] = grad_sigma
            grad_zed = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["grad_zed"] = grad_zed
            grad_radial = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["grad_radial"] = grad_radial

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
    def volume_grad(self):
        return self.graph.vertex_properties["volume_grad"]
    
    @filters.local
    def check_local_grad(self, pos0):
        if self.__verbose__: print "Checking gradient"
        chk_out = optimize.check_grad(self.opt_energy,
                                      self.opt_gradient,
                                      pos0.flatten())
        return chk_out

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
            num_ctoj = self.is_ctoj_edge.fa.sum()
            print (
                '''Updating gradient for %i cells,
                %i junctions vertices, %i junctions edges
                and %i cell to junction edges'''
                % (num_cells, num_jverts, num_edges, num_ctoj))
        self.grad_sigma.fa = 0.
        self.grad_zed.fa = 0.
        for edge in self.junctions:
            tension = self.junctions.line_tensions[edge]
            j_src, j_trgt = edge.source(), edge.target()
            u_sg = self.u_dsigmas[edge]
            u_zg = self.u_dzeds[edge]
            for cell in self.junctions.adjacent_cells[edge]:
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
            self.grad_sigma[j_src] += - tension * u_sg
            self.grad_zed[j_src] += - tension * u_zg
            self.grad_sigma[j_trgt] += tension * u_sg
            self.grad_zed[j_trgt] += tension * u_zg

    @filters.cells_in
    def update_cells_grad(self):
        # Cell vertices
        self.elastic_grad.fa =  self.cells.elasticities.fa \
                                * (self.cells.areas.fa -
                                   self.cells.prefered_area.fa )
        self.contractile_grad.fa =  self.cells.contractilities.fa \
                                    * self.cells.perimeters.fa
        self.volume_grad.fa = self.cells.vol_elasticities.fa \
                              * (self.cells.vols.fa
                                 - self.cells.prefered_vol.fa)
        
    @filters.active
    def radial_grad_array(self):

        rho_lumen = self.params['rho_lumen']
        rho0 = self.params['rho0']
        rho_lumen = self.params['rho_lumen']
        rho0 = self.params['rho0']
        area0 = self.params['prefered_area']
        num_cells = self.graph.num_vertices() #elastic_term.size
        prefered_vol0 = area0 * (rho0 - rho_lumen)
        norm_factor = num_cells * prefered_vol0 * prefered_vol0
        gradient = np.zeros(self.graph.num_vertices())

        if self.__verbose__ : print 'Gradient shape: %s' % gradient.shape
        gradient = self.grad_radial.fa / norm_factor
        return gradient

    def calc_radial_energy(self):
        rho_lumen = self.params['rho_lumen']
        rho0 = self.params['rho0']
        area0 = self.params['prefered_area']
        num_cells = self.graph.num_vertices() #elastic_term.size
        prefered_vol0 = area0 * (rho0 - rho_lumen)
        norm_factor = num_cells * prefered_vol0 * prefered_vol0**2
        radial_tension_energy = self.junctions.radial_tensions.fa \
                                * self.rhos.fa
        radial_tension_energy *= self.is_cell_vert.fa * self.is_alive.fa

        volume_energy = self.cells.vol_elasticities.fa \
                        * (self.cells.vols.fa - self.cells.prefered_vol.fa)**2
        total_energy = radial_tension_energy.sum() + volume_energy.sum()
        return total_energy / norm_factor
        

    @filters.ctoj_in
    def update_radial_grad(self):
        self.grad_radial.fa = self.junctions.radial_tensions.fa
        if self.__verbose__:
            print('''
                  Updating radial gradient
                  for %i cell to junction edges
                  ''' % self.graph.num_edges)
        for e in self.graph.edges():
            if not self.is_alive[e.source()]: continue
            self.grad_radial[e.target()] += self.u_drhos[e] \
                                            * self.volume_grad[e.source()]
            
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
        self.set_vertex_state()

        area0 = self.params['prefered_area']
        area_opt = delta_o**2 * area0
        correction = np.sqrt(area_opt / area)

        print "Scaling all the distances by a factor %.3f" %correction
        self.delta_o = delta_o
        self.ground_energy = self.isotropic_energy(delta_o, gamma, lbda)
        self.scale(correction)
        self.update_geometry()
        self.set_vertex_state([(self.is_cell_vert, False),
                               (self.is_alive, False)])
        area = self.cells.areas.fa.mean()
        vol = self.cells.vols.fa.mean()
        self.set_vertex_state()
        height = vol/area
        rho_avg = self.rhos.a.mean()
        self.params['rho_lumen'] = rho_avg - height
        
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
