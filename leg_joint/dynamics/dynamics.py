# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import logging
log = logging.getLogger(__name__)

import numpy as np

#from utils import compute_distribution
#from scipy.interpolate import splev
from scipy import optimize
import graph_tool.all as gt

mu = 6 * np.sqrt(2. / (3 * np.sqrt(3)))

from .pandas_dynamics import compute_energy, compute_gradient


class Dynamics(object):
    '''
    Container for the various gradients and energy
    aspects of the model
    '''
    def __init__(self):

        vol_elasticity0 = self.params['vol_elasticity']
        prefered_area = self.params["prefered_area"]
        height = self.params["prefered_height"]
        prefered_vol = prefered_area * height
        self.norm_factor = vol_elasticity0 * prefered_vol**2

    def calc_energy(self, full_output=False):
        return compute_energy(self, full_output)

    def update_gradient(self, components=False):
        '''
        Updates the components of the gradient exerted on the junction
        vertices
        '''
        return compute_gradient(self, components)

    def isotropic_relax(self):

        self.update_polar()
        area0 = self.params['prefered_area']
        h_0 = self.params['prefered_height']
        ### Cells only area and height
        live_cells = self.is_cell_vert.copy()
        live_cells.a = self.is_cell_vert.a * self.is_alive.a
        self.graph.set_vertex_filter(live_cells)
        area_avg = self.areas.fa.mean()
        rho_avg = self.rhos.fa.mean()
        self.graph.set_vertex_filter(None)

        ### Set height and area to height0 and area0
        scale = (area0 / area_avg)**0.5
        self.scale(scale)
        self.rho_lumen = rho_avg * scale - h_0
        self.update_geometry()

        ### Optimal value for delta
        delta_o = self.find_grad_roots()
        if not np.isfinite(delta_o):
            raise ValueError('invalid parameters values')
        self.delta_o = delta_o
        self.ground_energy = self.isotropic_energy(delta_o)
        ### Scaling

        self.scale(delta_o)
        self.update_geometry()

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
