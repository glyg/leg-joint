# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import numpy as np

from scipy import optimize

# from ..utils import local_subgraph
# from ..epithelium import hdf_snapshot


import logging
log = logging.getLogger(__name__)

def find_energy_min(eptm):


    eptm.update_geometry()
    pos0 = eptm.vertex_df.loc[eptm.uix_active,
                              eptm.coords].values.flatten()

    p_out = optimize.minimize(opt_energy,
                              pos0, method='L-BFGS-B',
                              jac=opt_gradient,
                              args=(eptm,),
                              options={'gtol': 1e-4, 'disp': False})
    return p_out

# @active
# def precondition(eptm):
#     '''Grabs the positions and computes the maximum displacements
#     before optimisation.
#     '''
#     pos0 = np.vstack([eptm.x.fa,
#                       eptm.y.fa,
#                       eptm.z.fa]).T.flatten()
#     max_disp = 2 * eptm.edge_lengths.fa.mean()
#     if eptm.__verbose__ : print('Initial postion  has shape %s'
#                                 % str(pos0.shape))
#     if eptm.__verbose__ : print('Max displacement amplitude  : %.3f'
#                                 % max_disp)
#     bounds = np.zeros((pos0.shape[0],),
#                       dtype=[('min', np.float32),
#                              ('max', np.float32)])
#     if eptm.__verbose__: print('bounds array has shape: %s'
#                                % str(bounds.shape))
#     x_bounds = np.vstack((eptm.x.fa - max_disp,
#                           eptm.x.fa + max_disp)).T
#     y_bounds = np.vstack((eptm.y.fa - max_disp,
#                           eptm.y.fa + max_disp)).T
#     z_bounds = np.vstack((eptm.z.fa - max_disp,
#                           eptm.z.fa + max_disp)).T

#     for n in range(pos0.shape[0]//3):
#         bounds[3 * n] = (x_bounds[n, 0], x_bounds[n, 1])
#         bounds[3 * n + 1] = (y_bounds[n, 0], y_bounds[n, 1])
#         bounds[3 * n + 2] = (z_bounds[n, 0], z_bounds[n, 1])
#     return pos0, bounds

#@hdf_snapshot
#@local_subgraph
def approx_grad(eptm):
    pos0 = eptm.vertex_df.loc[eptm.uix_active,
                              eptm.coords].values.flatten()
    grad = optimize.approx_fprime(pos0,
                                  opt_energy,
                                  1e-9, eptm)
    return grad

#@local_subgraph
def check_local_grad(eptm):
    pos0 = eptm.vertex_df.loc[eptm.uix_active,
                              eptm.coords].values.flatten()
    log.info("Checking gradient")

    # grad_err = np.linalg.norm(approx_grad(eptm)
    #                           - eptm.gradient_array())
    grad_err = optimize.check_grad(opt_energy,
                                   opt_gradient,
                                   pos0,
                                   eptm)
    return grad_err

## For consistency, the first argument must be the postion
def opt_energy(pos, eptm):
    """
    """
    # Position setting
    eptm.set_new_pos(pos)
    eptm.update_geometry()
    eptm.update_gradient()
    energy = eptm.calc_energy()
    return energy/eptm.norm_factor

def opt_gradient(pos, eptm):
    """
    """
    grad = eptm.update_gradient().values.flatten()
    return grad / eptm.norm_factor

def opt_callback(pos, eptm):
    """ Call back for the optimization """
    eptm.reset_topology()

def isotropic_optimum(eptm, tol):
    """ Recursively apply `isotropic_relax`, until
    the total energy is stable (with relative tolerance `tol`)
    """
    energy0 = eptm.calc_energy()
    eptm.isotropic_relax()
    energy1 = eptm.calc_energy()
    criterium = np.abs(energy0 - energy1) / energy0
    while criterium > tol:
        energy0 = energy1
        eptm.isotropic_relax()
        energy1 = eptm.calc_energy()
        criterium = np.abs(energy0 - energy1) / energy0


def running_local_optimum(eptm, tol, pola=False, save_to=None, ):
    '''
    Computes the local energy minimum for each cell on the filtered epithelium
    in a random order
    '''
    cells = [cell for cell in eptm.cells if eptm.is_alive[cell]]
    np.random.shuffle(cells)
    if pola:
        phi = eptm.dsigmas.copy()
        eptm.update_tensions(phi, np.pi/3)
    for cell in cells:
        if not eptm.is_alive[cell]: continue
        eptm.set_local_mask(None)
        eptm.set_local_mask(cell, wider=True)
        find_energy_min(eptm, tol=1e-3, approx_grad=0)
        if pola:
            eptm.update_tensions(phi, np.pi/3)
    if save_to is not None:
        eptm.graph.save(save_to)
