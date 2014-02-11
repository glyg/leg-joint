# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np 

from .utils import local_slice
from .topology import remove_cell, solve_all_rosettes
from .graph_representation import epithelium_draw

from .optimizers import find_energy_min



CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
GRAPH_SAVE_DIR = os.path.join(ROOT_DIR, 'saved_graphs')

'''
This module contains all the apoptosis specific files.


'''



def apoptosis_step(eptm, a_cell,
                   vol_reduction=0.01,
                   contractility=1.,
                   radial_tension=0.):
    '''
    Simulates an apoptotic step

    Parameters:
    -----------

    eptm:  ::class:`Epithelium` instance
    a_cell: the cell vertex of `eptm` to kill
    vol_reduction: float, the relative reduction of the
        the cell's equilibrium volume
    contractility: float, the relative increase in cell contractility
    radial_tension: float, the increase of apical basal tension
    '''
    eptm.set_local_mask(None)
    eptm.set_local_mask(a_cell, wider=True)
    
    eptm.cells.prefered_vol[a_cell] *= vol_reduction
    eptm.cells.contractilities[a_cell] *= contractility

    mean_lt = np.array([eptm.junctions.line_tensions[je]
                        for je in eptm.cells.junctions[a_cell]]).mean()
    for jv in a_cell.out_neighbours():
        eptm.junctions.radial_tensions[jv] += radial_tension * mean_lt
    find_energy_min(eptm)
    

def get_apoptotic_cells(eptm, num_cells=30, seed=10,
                        distribution='random', num_steps=10, **kwargs):
    '''Returns the specified number of apoptotic cells around the joint. 
    
    Parameters:
    ===========
    
    eptm: a :class:`Epithelium` instance

    num_cells: `int`, the number of cells seed: `int` passed to the
        random number generator

    distribution: {'random' | 'regular'}: the apoptotic cells
        distribution around the joint
    
    **kwargs are passed to the functions `random_apoptotic cells`
        and `regular_apoptotic_cells`
    
    See also:
    =========
    
    random_apoptotic_cells, regular_apoptotic_cells

    '''
    n_apopto = 0
    n_iter = 0
    eptm.set_local_mask(None)
    local_slice(eptm, zed_amp=kwargs['width_apopto'],
                theta_amp=2*np.pi)
    eptm.graph.set_vertex_filter(eptm.is_local_vert)
    fold_cells = np.array([cell for cell in eptm.cells])
    if distribution == 'regular':
        apopto_cells = regular_apoptotic_cells(eptm, num_cells, **kwargs)        
    elif distribution == 'random':
        while n_apopto != num_cells:
            seed += 1
            n_iter += 1
            if n_iter > 1000:
                raise RuntimeError('''Number of trials to high, 
                                   Try changing the parameters''')
            apopto_cells = random_apoptotic_cells(eptm, num_cells,
                                                  fold_cells, seed=seed, **kwargs)
    apopto_sequence = _get_sequence(apopto_cells, num_steps)
    return apopto_cells, fold_cells, apopto_sequence


def _get_sequence(apopto_cells, num_steps):
    apopto_sequence = []
    num_cells = apopto_cells.size
    for k in np.arange(- num_steps + 1, num_cells):
        start = max(k, 0)
        stop = min(k + num_steps, num_cells)
        apopto_sequence.append(apopto_cells[start:stop])
    return apopto_sequence

    
def random_apoptotic_cells(eptm, num_cells, fold_cells,
                           seed=42, **kwargs):
    '''
    Returns a **random** number of apoptotic cells,
    such that in average there are `num_cells` returned
    '''
    all_probas = np.array([_apopto_pdf(eptm.zeds[cell],
                                       eptm.thetas[cell],
                                       **kwargs)
                           for cell in fold_cells])
    all_probas *=  num_cells /  all_probas.sum()
    np.random.seed(seed)
    dices = np.random.random(size=all_probas.size)
    apopto_cells = fold_cells[all_probas > dices]
    thetas = np.array([eptm.thetas[cell] for cell in apopto_cells])
    theta_idx = np.argsort(np.cos(thetas/2))#[::-1]
    return apopto_cells.take(theta_idx)

def _apopto_pdf(zed, theta, z0=0., width_apopto=1.5, p0=0.95, amp=0.4):
    p = p0 * np.exp(- (zed - z0)**2 / width_apopto**2)\
        * (1  - amp * (np.cos(theta/2)**2))
    return p

def regular_apoptotic_cells(eptm, num_cells, **kwargs):
    ''' Returns a list of `num_cells` evenly reparted around the joint.

    If `gamma` is not `None` 
    '''
    is_apoptotic = eptm.is_alive.copy()
    is_apoptotic.a[:] = 0

    eptm.update_rhotheta()
    thetas_in = np.linspace(0, 2*np.pi,
                            num=num_cells,
                            endpoint=False)
    
    thetas_out = _ventral_enhance(thetas_in, kwargs['gamma']) - np.pi
    sigmas_out = thetas_out * eptm.rhos.a.mean()
    zeds_out = np.random.normal(0,
                                scale=kwargs['width_apopto'],
                                size=num_cells)
    #eptm.graph.set_vertex_filter(eptm.is_cell_vert)
    for sigma, zed in zip(sigmas_out, zeds_out):
        vfilt = eptm.is_cell_vert.copy()
        vfilt.a -= is_apoptotic.a
        eptm.graph.set_vertex_filter(vfilt)
        cell = eptm.closest_vert(sigma, zed)
        is_apoptotic[cell] = 1
    eptm.graph.set_vertex_filter(None)
    apopto_cells = np.array([cell for cell in eptm.cells
                             if is_apoptotic[cell]])
    thetas = np.array([eptm.thetas[cell] for cell in apopto_cells])
    theta_idx = np.argsort(np.cos(thetas/2))#[::-1]
    return apopto_cells.take(theta_idx)

def _ventral_enhance(thetas_in, gamma=1):
    ''' '''
    thetas_in = np.atleast_1d(thetas_in)
    thetas_out = np.zeros_like(thetas_in)
    gamma_low = lambda x, gamma: np.pi * (x / np.pi)**gamma
    gamma_high = lambda x, gamma: np.pi * (2 - (2 - x / np.pi)**(gamma)) 
    thetas_out[thetas_in <= np.pi] = gamma_low(thetas_in[thetas_in <= np.pi],
                                               gamma)
    thetas_out[thetas_in > np.pi] = gamma_high(thetas_in[thetas_in > np.pi],
                                               gamma)
    return thetas_out

def post_apoptosis(eptm, new_jv, fold_cells, **kwargs):

    # for cell in fold_cells:
    #     if not eptm.is_alive[cell]:
    #         continue
    #     eptm.set_local_mask(None)
    #     eptm.set_local_mask(cell)
    #     lj.find_energy_min(eptm)
    #     if pola:
    #         eptm.update_tensions(phi, np.pi / 4, 1.26**4)

    modified_cells = [cell for cell in new_jv.in_neighbours()
                      if eptm.is_cell_vert[cell]]
    
    rosette_jvs = solve_all_rosettes(eptm)
    new_jvs = rosette_jvs.append(new_jv)
    new_edges = np.unique([je for je in jv.all_edges()
                           for jv in new_jvs])
    
    if  kwargs['residual_ab_tension'] is not None:
        for jv in new_jvs:
            eptm.junctions.radial_tensions[jv] = kwargs['residual_ab_tension']

    if  kwargs['tension_increase'] is not None:
        increase_tension(eptm, new_edges,
                         kwargs['tension_increase'])

    if  kwargs['contractility_increase'] is not None:
        increase_contractility(eptm, modified_cells,
                               kwargs['contractility_increase'])
    
    np.random.shuffle(fold_cells)
    for cell in fold_cells:
        if not eptm.is_alive[cell]:
            continue
        eptm.set_local_mask(None)
        eptm.set_local_mask(cell)
        find_energy_min(eptm)

def increase_contractility(eptm, cells, contractility_increase):
    for cell in cells:
        eptm.cells.contractility[cell] *= contractility_increase 
        
def increase_tension(eptm, edges, tension_increase):

    for edge in edges:
        if eptm.is_junction_edge[edge]:
            eptm.junctions.line_tensions[edge] *= tension_increase

def gradual_apoptosis(eptm, seq_kwargs,
                      apopto_kwargs, post_kwargs,
                      basepath='', save_pngs=True):
    
    apopto_cells, fold_cells, apopto_sequence = get_apoptotic_cells(eptm, **seq_kwargs)
    eptm.set_local_mask(None)
    i = 0
    prev_first = apopto_cells[0]

    def to_png(i):
        png_dir = os.path.join(GRAPH_SAVE_DIR, 'png',
                               basepath)
        if not os.path.isdir(png_dir):
            os.mkdir(png_dir)
        fname_png_3d = os.path.join(png_dir,
                                    'apopto_3d_%04i.png' %i)
        fname_png_2d = os.path.join(GRAPH_SAVE_DIR,'png',
                                    basepath,
                                    'apopto_2d_%04i.png' %i)
        epithelium_draw(eptm, d_theta=0, z_angle=-0.15,
                        output_3d=fname_png_3d,
                        output_2d=fname_png_2d)

    for sub_sequence in apopto_sequence:
        for a_cell in sub_sequence:
            i += 1
            apoptosis_step(eptm, a_cell, **apopto_kwargs)
            if save_pngs:
                to_png(i)
            
        new_jv = remove_cell(eptm, prev_first)
        i+=1
        post_apoptosis(eptm, new_jv,
                       fold_cells, post_kwargs)
        if save_pngs:
            to_png(i)
        
    # lj.running_local_optimum(eptm, tol=1e-6)

    
def show_distribution(eptm):

    from .graph_representation import plot_edges_generic
    
    local_slice(eptm, zed_amp=2., theta_amp=None)
    axes = plot_edges_generic(eptm, eptm.zeds, eptm.wys, ctoj=False)#,
                                 #efilt=eptm.is_local_edge)
    x_min = eptm.ixs.a.min()
    x_max = eptm.ixs.a.max()
    for cell in eptm.cells:
        if eptm.is_apoptotic[cell]:
            alpha = 0.3 + 0.7 * (eptm.ixs[cell] - x_min)/(x_max - x_min)
            axes.plot(eptm.zeds[cell], eptm.wys[cell], 'ro', alpha=alpha)
    return axes

