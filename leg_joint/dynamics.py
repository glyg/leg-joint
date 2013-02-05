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
            self._init_gradients()
        else:
            self._get_gradients()

        vol_elasticity0 = self.params['vol_elasticity']
        prefered_area = self.params["prefered_area"]
        height = self.params["prefered_height"]
        prefered_vol = prefered_area * height
        self.norm_factor = vol_elasticity0 * prefered_vol**2

    def _init_gradients(self):
        # Gradients amplitudes
        self.elastic_grad = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["elastic_grad"] = self.elastic_grad
        self.contractile_grad = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["contractile_grad"]\
            = self.contractile_grad
        self.volume_grad = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["volume_grad"] = self.volume_grad
        # Gradients along sigma, zed and rho
        self.grad_sigma = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["grad_sigma"] = self.grad_sigma
        self.grad_ix = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["grad_ix"] = self.grad_ix
        self.grad_wy = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["grad_wy"] = self.grad_wy
        self.grad_zed = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["grad_zed"] = self.grad_zed
        self.grad_radial = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["grad_radial"] = self.grad_radial
        self.volume_grad_radial = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["volume_grad_radial"]\
            = self.volume_grad_radial
        self.volume_grad_apical = self.graph.new_vertex_property('float')
        self.graph.vertex_properties["volume_grad_apical"]\
            = self.volume_grad_apical

        self.volume_grad_cell = self.graph.new_vertex_property('vector<double>')
        self.graph.vertex_properties["volume_grad_cell"]\
            = self.volume_grad_cell

        
    def _get_gradients(self):
        # Gradients amplitudes
        self.elastic_grad = self.graph.vertex_properties["elastic_grad"]
        self.contractile_grad\
            = self.graph.vertex_properties["contractile_grad"]
        self.volume_grad = self.graph.vertex_properties["volume_grad"]
        # Gradients along sigma, zed and rho
        self.grad_sigma = self.graph.vertex_properties["grad_sigma"]
        self.grad_ix = self.graph.vertex_properties["grad_ix"]
        self.grad_wy = self.graph.vertex_properties["grad_wy"]
        self.grad_zed = self.graph.vertex_properties["grad_zed"]
        self.grad_radial = self.graph.vertex_properties["grad_radial"]
        self.volume_grad_radial\
            = self.graph.vertex_properties["volume_grad_radial"]
        self.volume_grad_apical\
            = self.graph.vertex_properties["volume_grad_apical"]

        self.volume_grad_cell = self.graph.new_vertex_property('vector<double>')
        self.graph.vertex_properties["volume_grad_cell"]\
            = self.volume_grad_cell
        
    def calc_energy(self):
        """ Computes the apical energy on the filtered epithelium """
        cells_term = self.calc_cells_energy()
        junction_term, radial_term = self.calc_junctions_energy()
        
        total_energy = (cells_term.sum()
                        + junction_term.sum()
                        + radial_term.sum())
        return total_energy / self.norm_factor

    @filters.cells_in
    def calc_cells_energy(self):
        contractile_term = 0.5 * self.cells.contractilities.fa * \
                           self.cells.perimeters.fa**2
        volume_term = 0.5 * self.cells.vol_elasticities.fa \
                      * (self.cells.vols.fa - self.cells.prefered_vol.fa)**2
        cells_energy = contractile_term + volume_term
        return cells_energy

    
        
    @filters.j_edges_in
    def calc_junctions_energy(self):
        junctions_energy = self.junctions.line_tensions.fa\
                           * self.edge_lengths.fa
        radial_energy = self.junctions.radial_tensions.fa\
                        * (self.rhos.fa - self.params['rho_lumen'])
        return junctions_energy, radial_energy
        
    @filters.active
    def gradient_array(self, gtol=1e-8):
        gradient = np.zeros(self.graph.num_vertices() * 3)
        if self.__verbose__ : print 'Gradient shape: %s' % gradient.shape
        gradient[::3] = self.grad_ix.fa
        gradient[1::3] = self.grad_wy.fa
        gradient[2::3] = self.grad_zed.fa 
        gradient[np.abs(gradient) < gtol] = 0.
        return gradient / self.norm_factor

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
        self.volume_grad_radial.fa = k_deltav# * self.cells.areas.fa
        self.volume_grad_apical.fa = k_deltav
        for cell in self.cells:
            self.calc_vol_grad_cell(cell)

    def calc_vol_grad_cell(self, cell):
        vol_grad = 0
        for j_edge in self.cells.junctions[cell]:
            triangle = self.diamonds[j_edge].triangles[cell]
            vol_grad += triangle.height * np.cross(triangle.u_cross,
                                                   triangle.rij_vect) / 2.
        vol_grad *= self.volume_grad_radial[cell] / self.cells.num_sides[cell]
        self.volume_grad_cell[cell] = vol_grad

    def update_junctions_grad(self):
        # Junction edges
        if self.__verbose__ :
            num_cells = self.is_cell_vert.a.sum()
            num_jverts = self.graph.num_vertices() - num_cells
            num_edges = self.is_junction_edge.a.sum()
            num_ctoj = self.is_ctoj_edge.a.sum()
            print (
                '''Updating gradient for %i cells,
                %i junctions vertices, %i junctions edges
                and %i cell to junction edges'''
                % (num_cells, num_jverts, num_edges, num_ctoj))
        self.grad_wy.a = 0.
        self.grad_ix.a = 0.
        self.grad_zed.a = 0.
        
        radial = self.junctions.radial_tensions.a
        self.grad_ix.a += radial * self.ixs.a / self.rhos.a
        self.grad_wy.a += radial * self.wys.a / self.rhos.a
        for j_edge in self.junctions:
            self.update_edge_grad(j_edge)
        for ctoj_edge in gt.find_edge(self.graph, self.is_ctoj_edge, True):
            cell, j_vert = ctoj_edge
            gc_x, gc_y, gc_z = self.volume_grad_cell[cell]
            self.grad_ix[j_vert] += gc_x
            self.grad_wy[j_vert] += gc_y
            self.grad_zed[j_vert] += gc_z
            
    def update_edge_grad(self, j_edge):

        tension = self.junctions.line_tensions[j_edge]
        jv0, jv1 = j_edge
        u_xg = self.u_dixs[j_edge]
        u_yg = self.u_dwys[j_edge]
        u_zg = self.u_dzeds[j_edge]
        self.grad_ix[jv0] -= tension * u_xg
        self.grad_wy[jv0] -= tension * u_yg
        self.grad_zed[jv0] -= tension * u_zg
        
        self.grad_ix[jv1] += tension * u_xg
        self.grad_wy[jv1] += tension * u_yg
        self.grad_zed[jv1] += tension * u_zg

        dmnd = self.diamonds[j_edge]
        rij_vect = np.array([self.dixs[j_edge],
                             self.dwys[j_edge],
                             self.dzeds[j_edge]])
        for triangle in dmnd.triangles.values():
            # K_alpha DeltaV h_alpha / 2
            v_grad_a = 0.5 * (self.volume_grad_apical[triangle.cell]
                              * triangle.height)
            r_a0, r_a1 = triangle.deltas
            area_grad0 = - v_grad_a * np.cross(triangle.u_cross, r_a1)
            self.grad_ix[jv0] += area_grad0[0]
            self.grad_wy[jv0] += area_grad0[1]
            self.grad_zed[jv0] += area_grad0[2]

            area_grad1 = + v_grad_a * np.cross(triangle.u_cross, r_a0)
            self.grad_ix[jv1] += area_grad1[0]
            self.grad_wy[jv1] += area_grad1[1]
            self.grad_zed[jv1] += area_grad1[2]

            #K_alpha DeltaV Aaij / 2
            v_grad_r = (self.volume_grad_radial[triangle.cell]
                        * triangle.area) / 2.

            self.grad_ix[jv0] += v_grad_r * self.ixs[jv0] / self.rhos[jv0]
            self.grad_wy[jv0] += v_grad_r * self.wys[jv0] / self.rhos[jv0]
            self.grad_ix[jv1] += v_grad_r * self.ixs[jv1] / self.rhos[jv1]
            self.grad_wy[jv1] += v_grad_r * self.wys[jv1] / self.rhos[jv1]

            
            ctr_grad = self.contractile_grad[triangle.cell]
            self.grad_ix[jv0] -= ctr_grad * u_xg
            self.grad_wy[jv0] -= ctr_grad * u_yg
            self.grad_zed[jv0] -= ctr_grad * u_zg

            self.grad_ix[jv1] += ctr_grad * u_xg
            self.grad_wy[jv1] += ctr_grad * u_yg
            self.grad_zed[jv1] += ctr_grad * u_zg
            

    def isotropic_relax(self):
        
        gamma = self.paramtree.relative_dic['contractility']
        lbda = self.paramtree.relative_dic['line_tension']
        # good, report = self.check_phase_space(gamma, lbda)
        # if not good:
        #     raise ValueError("Invalid values for "
        #                      "the average contractility and elasticity \n"
        #                      +report)
        ### Computing the optimal dilation for the parameters
        delta0 = 0.9
        delta_o,  = optimize.fsolve(self.isotropic_grad, delta0,
                                    args=(gamma, lbda))
        self.delta_o = delta_o
        self.ground_energy = self.isotropic_energy(delta_o, gamma, lbda)

        ### Relaxing tissue average height
        height0 = self.params['prefered_height']
        rho_avg = self.rhos.a.mean()
        self.params['rho_lumen'] = rho_avg - height0
        self.update_cells_pos()
        self.update_geometry()

        ### Cells only area
        self.set_vertex_state([(self.is_cell_vert, False),
                               (self.is_alive, False)])
        area = self.cells.areas.fa.mean()
        vol =  self.cells.vols.fa.mean()
        self.set_vertex_state()
        ### Searching for the correct scaling (see the doc)
        area0 = self.params['prefered_area']
        R = self.params['rho_lumen']
        cst = - (delta_o**2) * area0 * height0 / area
        scaler_poly = [rho_avg, -R, 0., cst]
        roots = np.roots(scaler_poly)
        good_roots = np.real([r for r in roots if np.abs(r) == r])
        if len(good_roots) == 1:
            correction = good_roots[0]
        else:
            print 'More than one scaling solution, which is strange'
            correction = good_roots.min()
        self.scale(correction)
        if self.__verbose__:
            print "Scaled all the distances by a factor %.3f" %correction
        self.update_cells_pos()
        self.update_geometry()
        self.update_gradient()
        
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
