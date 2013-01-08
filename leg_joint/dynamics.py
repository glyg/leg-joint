#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

#from utils import compute_distribution
#from scipy.interpolate import splev
from scipy import optimize
import graph_tool.all as gt
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
            volume_grad_radial = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["volume_grad_radial"]\
                = volume_grad_radial
            volume_grad_apical = self.graph.new_vertex_property('float')
            self.graph.vertex_properties["volume_grad_apical"]\
                = volume_grad_apical

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
    @property
    def volume_grad_radial(self):
        return self.graph.vertex_properties["volume_grad_radial"]
    @property
    def volume_grad_apical(self):
        return self.graph.vertex_properties["volume_grad_apical"]
    
    def calc_energy(self):
        """ Computes the apical energy on the filtered epithelium """
        cells_term, denominator = self.calc_cells_energy()
        junction_term = self.calc_junctions_energy()
        
        total_energy = cells_term.sum() + junction_term.sum()
        return total_energy / denominator

    @filters.cells_in
    def calc_cells_energy(self):
        # elastic_term = 0.5 * self.cells.elasticities.fa * (
        #     self.cells.areas.fa - self.cells.prefered_area.fa)**2
        contractile_term = 0.5 * self.cells.contractilities.fa * \
                           self.cells.perimeters.fa**2
        volume_term = 0.5 * self.cells.vol_elasticities.fa \
                      * (self.cells.vols.fa - self.cells.prefered_vol.fa)**2
        prefered_area0 =  self.params['prefered_area']
        elasticity0 = self.params['elasticity']
        denominator = elasticity0 * prefered_area0**2 #* num_cells 
        cells_energy = contractile_term + volume_term# + elastic_term
        #num_cells = self.graph.num_vertices()#elastic_term.size
        return cells_energy, denominator
        
    @filters.j_edges_in
    def calc_junctions_energy(self):
        junctions_energy = self.junctions.line_tensions.fa\
                           * self.edge_lengths.fa
        return junctions_energy
        
    @filters.active
    def gradient_array(self):
        prefered_area0 =  self.params['prefered_area']
        elasticity0 = self.params['elasticity']
        norm_factor = elasticity0 * prefered_area0**2

        gradient = np.zeros(self.graph.num_vertices() * 3)
        if self.__verbose__ : print 'Gradient shape: %s' % gradient.shape
        gradient[::3] = self.grad_radial.fa
        gradient[1::3] = self.grad_sigma.fa / self.rhos.fa
        gradient[2::3] = self.grad_zed.fa 
        return gradient / norm_factor

    def update_gradient(self):
        self.update_cells_grad()
        self.update_junctions_grad()
        
    @filters.cells_in
    def update_cells_grad(self):
        # Cell vertices
        self.contractile_grad.fa =  self.cells.contractilities.fa \
                                    * self.cells.perimeters.fa
        delta_v = self.cells.vols.fa - self.cells.prefered_vol.fa
        k_deltav = self.cells.vol_elasticities.fa * delta_v
        self.volume_grad_radial.fa = k_deltav * self.cells.areas.fa
        self.volume_grad_apical.fa = k_deltav * (self.rhos.fa
                                                 - self.params['rho_lumen'])
        
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
        self.grad_sigma.a[:] = 0.
        self.grad_zed.a[:] = 0.
        self.grad_radial.a[:] = 0.
        for j_edge in self.junctions:
            self.update_edge_grad(j_edge)
        for ctoj_edge in gt.find_edge(self.graph, self.is_ctoj_edge, True):
            cell, j_vert = ctoj_edge
            self.volume_grad_radial[j_vert] += self.volume_grad_radial[cell]\
                                               / self.num_sides[cell]
            
    def update_edge_grad(self, j_edge):

        u_sg = self.u_dsigmas[j_edge]
        u_zg = self.u_dzeds[j_edge]
        u_rg = self.u_drhos[j_edge]
        tension = self.junctions.line_tensions[j_edge]
        jv0, jv1 = j_edge
        self.grad_radial[jv0] -= tension * u_rg
        self.grad_sigma[jv0] -= tension * u_sg
        self.grad_zed[jv0] -= tension * u_zg
        self.grad_radial[jv1] += tension * u_rg
        self.grad_sigma[jv1] += tension * u_sg
        self.grad_zed[jv1] += tension * u_zg
        dmnd = self.diamonds[j_edge]
        for triangle in dmnd.triangles.values():
            ctoj0, ctoj1 = triangle.ctoj_edges
            v_grad_a = triangle.sign * self.volume_grad_apical[triangle.cell] 

            theta1 = self.thetas[jv0] + self.dthetas[j_edge]
            self.grad_radial[jv0] += v_grad_a * self.dzeds[ctoj1] * theta1
            self.grad_sigma[jv0] += v_grad_a * self.dzeds[ctoj1]
            self.grad_zed[jv0] -= v_grad_a * self.dsigmas[ctoj1]

            theta0 = self.thetas[jv1] - self.dthetas[j_edge]
            self.grad_radial[jv1] -= v_grad_a * self.dzeds[ctoj0] * theta0
            self.grad_sigma[jv1] -= v_grad_a * self.dzeds[ctoj0]
            self.grad_zed[jv1] += v_grad_a * self.dsigmas[ctoj0]

            ctr_grad = self.contractile_grad[triangle.cell]
            self.grad_radial[jv0] -= ctr_grad * u_rg
            self.grad_sigma[jv0] -= ctr_grad * u_sg
            self.grad_zed[jv0] -= ctr_grad * u_zg

            self.grad_radial[jv1] += ctr_grad * u_rg
            self.grad_sigma[jv1] += ctr_grad * u_sg
            self.grad_zed[jv1] += ctr_grad * u_zg

                        
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
        height = self.cells.prefered_vol.fa.mean() / area
        rho_avg = self.rhos.fa.mean()
        self.set_vertex_state()
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
