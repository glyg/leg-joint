#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize

from utils import compute_distribution
from scipy.interpolate import splev

import filters


class EpitheliumEnergies(object):

    def __init__(self):
        if self.new:
            # Gradients amplitudes
            elastic_grad = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["elastic_grad"] = elastic_grad
            contractile_grad = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["contractile_grad"] = contractile_grad
            volume_grad = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["volume_grad"] = volume_grad

            grad_radial = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["grad_radial"] = grad_radial

            # Gradients along sigma and zed
            grad_sigma = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["grad_sigma"] = grad_sigma
            grad_zed = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["grad_zed"] = grad_zed

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

    def opt_radial_energy(self, rhos):
        self.set_rhos(rhos)
        self.update_apical_geom()
        energy = self.calc_radial_energy()
        return energy
        
    def opt_gradient(self, sz_pos):
        """
        After setting the sigma, zed position to sz_pos,
        computes the gradient over the filtered graph
        """
        # # position setting
        self.set_new_pos(sz_pos)
        self.update_apical_geom()
        self.update_gradient()
        gradient = self.gradient_array()
        return gradient

    def opt_radial_grad(self, rhos):
        """
        After setting the sigma, zed position to sz_pos,
        computes the gradient over the filtered graph
        """
        # # position setting
        self.set_rhos(rhos)
        self.update_apical_geom()
        self.update_gradient()
        gradient = self.radial_grad_array()
        return gradient

        
    @filters.active
    def radial_grad_array(self):

        rho_lumen = self.params['rho_lumen']
        rho0 = self.params['rho0']
        prefered_vol =  self.params['prefered_area'] * (rho0 - rho_lumen)
        vol_elasticity0 = self.params['vol_elasticity']
        norm_factor = prefered_vol * vol_elasticity0
        gradient = np.zeros(self.graph.num_vertices())
        if self.__verbose__ : print 'Gradient shape: %s' % gradient.shape
        gradient = self.grad_radial.fa / norm_factor
        return gradient
        
    def opt_callback(self, sz_pos):
        """ Call back for the optimization """
        self.periodic_boundary_condition()
        self.update_apical_geom()
        self.update_gradient()

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

    def calc_radial_energy(self):

        rho_lumen = self.params['rho_lumen']
        rho0 = self.params['rho0']
        prefered_vol =  self.params['prefered_area'] * (rho0 - rho_lumen)
        vol_elasticity0 = self.params['vol_elasticity']
        num_cells = self.graph.num_vertices() #elastic_term.size
        norm_factor = num_cells * prefered_vol * vol_elasticity0**2
        radial_tension_energy = self.junctions.radial_tensions.fa \
                                * self.rhos.fa
        radial_tension_energy *= self.is_cell_vert.fa * self.is_alive.fa
        prefered_vol = self.cells.prefered_area.fa \
                       * self.cells.prefered_height.fa
        volume_energy = self.cells.vol_elasticities.fa \
                        * (self.cells.vols.fa - prefered_vol)**2
        total_energy = radial_tension_energy.sum() + volume_energy.sum()
        return total_energy / norm_factor
        
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
        prefered_vol = self.cells.prefered_area.fa \
                       * self.cells.prefered_height.fa
        self.volume_grad.fa = self.cells.vol_elasticities.fa \
                              * (self.cells.vols.fa - prefered_vol)
    @filters.ctoj_in
    def calc_radial_grad(self):
        self.grad_radial.fa = self.junctions.radial_tensions.fa
        if self.__verbose__:
            print('''Updating radial gradient
                  for %i cell to junction edges
                  ''' % self.graph.num_edges)
        for e in self.graph.edges():
            if not self.is_alive[e.source()]: continue
            self.grad_radial[e.target()] += self.u_drhos[e] \
                                            * self.volume_grad[e.source()]
        
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
        #self.zeds.a /= z_dilation
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
