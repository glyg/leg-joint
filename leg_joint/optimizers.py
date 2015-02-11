# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import numpy as np

from scipy import optimize

from .filters import active
from .utils import local_subgraph
from .epithelium import hdf_snapshot

from .pandas_geometry import opt_gradient as opt_gradient_pd
from .pandas_geometry import opt_energy as opt_energy_pd
from .pandas_geometry import parse_graph, update_graph
from .pandas_geometry import Triangles

from graph_tool import GraphView


import logging
log = logging.getLogger(__name__)

def find_energy_min_pd(eptm):

    local_graph = GraphView(eptm.graph,
                            vfilt=eptm.is_local_vert,
                            efilt=eptm.is_local_edge)
    vertex_df, edges_df, triangles = parse_graph(local_graph)
    coords = ['ixs', 'wys', 'zeds']

    trgles = Triangles(vertex_df, edges_df, triangles, coords)
    pos0 = trgles.vertex_df.loc[trgles.uix_active, coords].values.flatten()
    p_out = optimize.minimize(opt_energy_pd,
                              pos0, method='BFGS',
                              jac=opt_gradient_pd,
                              args=(trgles, eptm.norm_factor, eptm.rho_lumen),
                              options={'gtol': 1e-4, 'disp': False})
    eptm.graph.set_vertex_filter(eptm.is_local_vert)
    eptm.graph.set_edge_filter(eptm.is_local_edge)
    update_graph(trgles, eptm.graph)
    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)


@active
def precondition(eptm):
    '''Grabs the positions and computes the maximum displacements
    before optimisation.
    '''
    pos0 = np.vstack([eptm.ixs.fa,
                      eptm.wys.fa,
                      eptm.zeds.fa]).T.flatten()
    max_disp = 2 * eptm.edge_lengths.fa.mean()
    if eptm.__verbose__ : print('Initial postion  has shape %s'
                                % str(pos0.shape))
    if eptm.__verbose__ : print('Max displacement amplitude  : %.3f'
                                % max_disp)
    bounds = np.zeros((pos0.shape[0],),
                      dtype=[('min', np.float32),
                             ('max', np.float32)])
    if eptm.__verbose__: print('bounds array has shape: %s'
                               % str(bounds.shape))
    x_bounds = np.vstack((eptm.ixs.fa - max_disp,
                          eptm.ixs.fa + max_disp)).T
    y_bounds = np.vstack((eptm.wys.fa - max_disp,
                          eptm.wys.fa + max_disp)).T
    z_bounds = np.vstack((eptm.zeds.fa - max_disp,
                          eptm.zeds.fa + max_disp)).T

    for n in range(pos0.shape[0]//3):
        bounds[3 * n] = (x_bounds[n, 0], x_bounds[n, 1])
        bounds[3 * n + 1] = (y_bounds[n, 0], y_bounds[n, 1])
        bounds[3 * n + 2] = (z_bounds[n, 0], z_bounds[n, 1])
    return pos0, bounds

@local_subgraph
@hdf_snapshot
def find_energy_min(eptm, method='fmin_l_bfgs_b',
                    tol=1e-8, approx_grad=0, epsilon=1e-8):
    '''
    Performs the energy minimisation
    '''
    pos0, bounds = precondition(eptm)
    eptm.stamp += 1
    output = 0
    if method == 'fmin_l_bfgs_b':
        ## I set `factr` to 1e11 to avoid too long computation
        output = optimize.fmin_l_bfgs_b(opt_energy, pos0.flatten(),
                                        fprime=opt_gradient,
                                        #approx_grad=approx_grad,
                                        bounds=bounds.flatten(),
                                        args=(eptm,), factr=1e10,
                                        m=10, pgtol=tol,
                                        epsilon=epsilon, iprint=1,
                                        maxfun=150, disp=None)

    elif method=='fmin':
        output = optimize.fmin(opt_energy,
                               pos0.flatten(),
                               ftol=tol, xtol=0.01,
                               args=(eptm,),
                               callback=opt_callback)
    elif method=='fmin_ncg':
        output = optimize.fmin_ncg(opt_energy,
                                   pos0.flatten(),
                                   fprime=opt_gradient,
                                   args=(eptm,),
                                   avextol=tol,
                                   retall=True,
                                   maxiter=100)# ,

    elif method=='fmin_tnc':
        output = optimize.fmin_tnc(opt_energy,
                                   pos0.flatten(),
                                   fprime=opt_gradient,
                                   args=(eptm,),
                                   pgtol=tol,
                                   bounds=bounds,
                                   maxCGit=0,
                                   disp=5)
    elif method=='fmin_bfgs':
        output = optimize.fmin_bfgs(opt_energy,
                                    pos0.flatten(),
                                    fprime=opt_gradient,
                                    args=(eptm,),
                                    gtol=tol,
                                    norm=np.inf,
                                    retall=1,
                                    callback=opt_callback)
    return pos0, output

@local_subgraph
def approx_grad(eptm):
    pos0, bounds = precondition(eptm)
    grad = optimize.approx_fprime(pos0,
                                  opt_energy,
                                  1e-9, eptm)
    return grad

@local_subgraph
def check_local_grad(eptm):
    pos0, bounds = precondition(eptm)
    log.info("Checking gradient")

    # grad_err = np.linalg.norm(approx_grad(eptm)
    #                           - eptm.gradient_array())
    grad_err = optimize.check_grad(opt_energy,
                                   opt_gradient,
                                   pos0.flatten(),
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
    return energy

def opt_gradient(pos, eptm):
    """
    """
    # eptm.set_new_pos(pos)
    # eptm.update_geometry()
    # eptm.update_gradient()
    return eptm.gradient_array()

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
