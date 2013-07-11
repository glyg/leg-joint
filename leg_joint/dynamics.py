#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

#from utils import compute_distribution
#from scipy.interpolate import splev
from scipy import optimize
import graph_tool.all as gt
import filters

mu = 6 * np.sqrt(2. / (3 * np.sqrt(3)))


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
                        * (self.rhos.fa - self.rho_lumen)
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
        '''
        Updates the components of the gradient exerted on the junction
        vertices

        
        '''
        
        self._update_cells_grad()
        self._update_junctions_grad()
        
    def _update_cells_grad(self):
        # Cell vertices
        self.contractile_grad.fa =  self.cells.contractilities.fa \
                                    * self.cells.perimeters.fa
        delta_v = self.cells.vols.fa - self.cells.prefered_vol.fa
        k_deltav = self.cells.vol_elasticities.fa * delta_v
        self.volume_grad_radial.fa = k_deltav# * self.cells.areas.fa
        self.volume_grad_apical.fa = k_deltav
        for cell in self.cells:
            self._calc_vol_grad_cell(cell)

    def _calc_vol_grad_cell(self, cell):
        
        vol_grad = [0, 0, 0]
        if  self.is_alive[cell]:
            for j_edge in self.cells.junctions[cell]:
                try:
                    triangle = self.diamonds[j_edge].triangles[cell]
                    vol_grad += (triangle.height
                                 * np.cross(triangle.u_cross,
                                            triangle.rij_vect) / 2.)
                except KeyError:
                    pass
            vol_grad *= self.volume_grad_radial[cell]\
                        / self.cells.num_sides[cell]
        self.volume_grad_cell[cell] = vol_grad
            
    def _update_junctions_grad(self):
        
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

        # Radial tension
        radial = self.junctions.radial_tensions.a
        self.grad_ix.a += radial * self.ixs.a / self.rhos.a
        self.grad_wy.a += radial * self.wys.a / self.rhos.a

        # Contribution from junction edges
        for j_edge in self.junctions:
            self._update_edge_grad(j_edge)

        # Contribution from neighboring cells
        for ctoj_edge in gt.find_edge(self.graph, self.is_ctoj_edge, True):
            cell, j_vert = ctoj_edge
            gc_x, gc_y, gc_z = self.volume_grad_cell[cell]
            self.grad_ix[j_vert] += gc_x
            self.grad_wy[j_vert] += gc_y
            self.grad_zed[j_vert] += gc_z
            
    def _update_edge_grad(self, j_edge):
        ''' Computes the components of the gradient for the junction edge
        `j_edge` vertices 

        Parameter
        ---------
        j_edge : a junction edge
        
        '''

        
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
            
    def update_tensions(self, phi, delta_phi, factor=2.):
        '''
        Multiplies tension by `factor` for junctions that verify:
        .. math::

           - \Delta\phi/2 < \phi <  \Delta\phi/2
        with
        ..math::

           \phi = \tan^{-1}\frac{\sqrt{\delta x^2 + \delta y^2}}{\delta z}
        
        Parameters
        ----------
        phi : graph_tool :class:`EdgePropertyMap` with `float` data type
            used to store the values for the angle :math:`phi`

        delta_phi : float
            Angular range for which the line tension is changed
        
        '''

        lt0 = self.params['line_tension']
        phi.a = np.arctan2(np.sqrt(self.dixs.a**2
                                   + self.dwys.a**2),
                           self.dzeds.a)
        lower = - delta_phi
        upper = delta_phi
        for je in self.junctions:
            if (lower < phi[je] < upper):
                self.junctions.line_tensions[je] = factor * lt0
            else:
                self.junctions.line_tensions[je] = lt0

    def isotropic_relax(self):

        self.update_rhotheta()
        area0 = self.params['prefered_area']
        ### Cells only area and height
        self.set_vertex_state([(self.is_cell_vert, False),
                               (self.is_alive, False)])
        area_avg = self.cells.areas.fa.mean()
        rho_avg = self.rhos.fa.mean()
        self.set_vertex_state()
        
        ### Current value for delta
        delta_i = np.sqrt(area_avg / area0)

        ### Optimal value for delta
        delta_o = self.find_grad_roots()
        if not np.isfinite(delta_o):
            raise ValueError('invalid parameters values')
        self.delta_o = delta_o
        self.ground_energy = self.isotropic_energy(delta_o)
        ### Scaling
        correction = delta_o / delta_i
        self.scale(correction)
        self.update_geometry()

        if self.__verbose__:
            print "Scaled all the distances by a factor %.3f" %correction
        

    def isotropic_energy(self, delta):
        """
        Computes the theoritical energy per cell for the given
        parameters.
        """
        lbda = self.paramtree.relative_dic['line_tension']
        gamma = self.paramtree.relative_dic['contractility']

        elasticity = (delta**3 - 1 )**2 / 2.
        contractility = gamma * mu**2 * delta**2 / 2.
        tension = lbda * mu * delta / 2.
        energy = elasticity + contractility + tension
        return energy
        

    def isotropic_grad_poly(self):
        
        lbda = self.paramtree.relative_dic['line_tension']
        gamma = self.paramtree.relative_dic['contractility']
        grad_poly = [3, 0, 0,
                     -3,
                     mu**2 * gamma,
                     mu * lbda / 2.]
        return grad_poly

    def isotropic_grad(self, delta):
        grad_poly = self.isotropic_grad_poly()
        return np.polyval(grad_poly, delta)

    def find_grad_roots(self):
        p = self.isotropic_grad_poly()
        roots = np.roots(p)            
        good_roots = np.real([r for r in roots if np.abs(r) == r])
        np.sort(good_roots)
        if len(good_roots) == 1:
            return good_roots
        elif len(good_roots) > 1:
            return good_roots[0]
        else:
            return np.nan

        
    def check_phase_space(self):
        '''
        Checks wether parameter values `gamma` and `lbda` yields a
        correct phase space.
        '''
        lbda = self.paramtree.relative_dic['line_tension']
        gamma = self.paramtree.relative_dic['contractility']

        # See the energies.pynb notebook for the derivation of this:
        if (gamma < - lbda / (2 * mu)):
            report= ("Contractility is too low,"
                     "Soft network not supported")
            return False, report

        lbda_max = 6 * (2 / 5.)**(2/3.) * 0.6 / mu
        gamma_max = 3 * 4**(-1 / 3.) * 0.75 / mu**2


        if  gamma  > gamma_max:
            report = ("Invalid value for the contractility")
            return False, report

        if lbda > lbda_max:
            report = ("Invalid value for the line tension")
            return False, report

        delta_o = self.find_grad_roots()
        if not np.isfinite(delta_o):
            report = ('Invalid values for line tension and contractility')
            return False, report
        return True, 'ok!'


