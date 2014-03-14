# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np 
import graph_tool.all as gt

from .filters import local_slice, focus_on_cell
from .topology import remove_cell, solve_rosette, find_rosettes
from .graph_representation import png_snapshot
from .epithelium import hdf_snapshot

from .optimizers import find_energy_min, running_local_optimum
from .topology import type1_transition

import logging
log = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
GRAPH_SAVE_DIR = os.path.join(ROOT_DIR, 'saved_graphs')

'''
This module contains all the apoptosis specific files.


'''


@hdf_snapshot
@png_snapshot
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
    if radial_tension is not None:
        for jv in a_cell.out_neighbours():
            eptm.junctions.radial_tensions[jv] += radial_tension * mean_lt
    find_energy_min(eptm)
    
def get_apoptotic_cells(eptm, **kwargs):
    '''Returns the specified number of apoptotic cells around the joint. 
    
    Parameters:
    ===========
    
    eptm: a :class:`Epithelium` instance

    num_cells: `int`, the number of cells seed: `int` passed to the
        random number generator

    random: {True | False}: the apoptotic cells
        distribution around the joint, wether random with
        ventral bias (if True) or regular (if False)
    
    **kwargs are passed to the functions `random_apoptotic cells`
        and `regular_apoptotic_cells`
    
    See also:
    =========
    
    random_apoptotic_cells, regular_apoptotic_cells

    '''
    num_cells = kwargs['num_cells']
    seed = kwargs['seed']
    num_steps = kwargs['num_steps']
    regular_kwargs = {name:kwargs[name]
                      for name in ['width_apopto', 'gamma']}
    proba_kwargs = {name:kwargs[name]
                    for name in ['width_apopto', 'amp']}
    
    n_apopto = 0
    n_iter = 0
    local_slice(eptm, zed_amp=kwargs['width_apopto'], ##Might be higher
                theta_amp=2*np.pi)
    eptm.graph.set_vertex_filter(eptm.is_local_vert)
    fold_cells = np.array([cell for cell in eptm.cells])
    eptm.graph.set_vertex_filter(None)
    
    if not kwargs['random']:
        apopto_cells = regular_apoptotic_cells(eptm, num_cells,
                                               **regular_kwargs)        
    else:
        while n_apopto != num_cells:
            seed += 1
            n_iter += 1
            if n_iter > 1000:
                raise RuntimeError('''Number of trials to high, 
                                   Try changing the parameters''')
            apopto_cells = random_apoptotic_cells(eptm, num_cells,
                                                  fold_cells, seed,
                                                  proba_kwargs)
            n_apopto = len(apopto_cells)
    apopto_sequence = _get_sequence(apopto_cells, num_steps)
    return apopto_cells, fold_cells, apopto_sequence

def _get_sequence(apopto_cells, num_steps):
    apopto_sequence = []
    num_cells = apopto_cells.size
    for k in np.arange(- num_steps + 1, num_cells):
        start = int(max(k, 0))
        stop = int(min(k + num_steps, num_cells))
        apopto_sequence.append(apopto_cells[start:stop])
    return apopto_sequence

    
def random_apoptotic_cells(eptm, num_cells, fold_cells, seed, proba_kwargs):
    '''
    Returns a **random** number of apoptotic cells,
    such that in average there are `num_cells` returned
    '''
    
    all_probas = np.array([_apopto_pdf(eptm.zeds[cell],
                                       eptm.thetas[cell],
                                       **proba_kwargs)
                           for cell in fold_cells])
    all_probas *=  num_cells /  all_probas.sum()
    np.random.seed(seed)
    dices = np.random.random(size=all_probas.size)
    apopto_cells = fold_cells[all_probas > dices]
    thetas = np.array([eptm.thetas[cell] for cell in apopto_cells])
    theta_idx = np.argsort(np.cos(thetas/2)**2)#[::-1]
    return apopto_cells.take(theta_idx)

def _apopto_pdf(zed, theta, z0=0., width_apopto=1.5, amp=0.4):
    p = np.exp(- (zed - z0)**2 / width_apopto**2)\
        * (1  - amp * (np.cos(theta/2)**2))
    return p

def regular_apoptotic_cells(eptm, num_cells, gamma=1, width_apopto=1.8):
    ''' Returns a list of `num_cells` evenly reparted around the joint.

    '''
    is_apoptotic = eptm.is_alive.copy()
    is_apoptotic.a[:] = 0

    eptm.update_rhotheta()
    thetas_in = np.linspace(0, 2*np.pi,
                            num=num_cells,
                            endpoint=False)
    
    thetas_out = _ventral_enhance(thetas_in, gamma) - np.pi
    sigmas_out = thetas_out * eptm.rhos.a.mean()
    zeds_out = np.random.normal(0,
                                scale=width_apopto,
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
    theta_idx = np.argsort(np.cos(thetas/2)**2)#[::-1]
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

@hdf_snapshot
@png_snapshot
def post_apoptosis(eptm, a_cell, fold_cells, mode='shorter', **kwargs):
    try:
        induce_contractility(eptm, a_cell,
                             max_ci=kwargs['max_ci'],
                             rate_ci=kwargs['rate_ci'],
                             span=kwargs['span_ci'])
        out = find_energy_min(eptm)
    except KeyError:
        pass
    eptm.set_local_mask(None)
    neighbours = eptm.cells.get_neighbor_cells(a_cell)
    new_edges = []
    if mode == 'rosette':
        new_jv = remove_cell(eptm, a_cell)
        new_jvs = solve_all_rosettes(eptm)
        new_jvs.append(new_jv)
        for jv in new_jvs:
            new_edges.extend([je for je in jv.all_edges()
                              if (not je in new_edges)
                              and eptm.is_junction_edge[je]])
    elif mode == 'shorter':
        a_cell_edges = eptm.cells.junctions[a_cell]
        while len(a_cell_edges) > 3:
            modified_cells, modified_jverts = type1_at_shorter(eptm,
                                                               a_cell_edges)
            a_cell_edges = eptm.cells.junctions[a_cell]
            for neighbour in neighbours:
                new_edges.extend([je for je in eptm.cells.junctions[neighbour]
                                  if (not je in new_edges)
                                  and eptm.is_junction_edge[je]])
        new_jv = remove_cell(eptm, a_cell)
    wide_relax(eptm)
    # if  kwargs.get('residual_ab_tension') is not None:
    #     for jv in new_jvs:
    #         eptm.junctions.radial_tensions[jv] = kwargs['residual_ab_tension']

    # if  kwargs.get('tension_increase') is not None:
    #     increase_tension(eptm, new_edges,
    #                      kwargs['tension_increase'])

    # if  kwargs.get('contractility_increase') is not None:
    #     increase_contractility(eptm, neighbours,
    #                            kwargs['contractility_increase'])
    
    # np.random.shuffle(fold_cells)
    # for cell in fold_cells:
    #     if not eptm.is_alive[cell]:
    #         continue
    #     eptm.set_local_mask(None)
    #     eptm.set_local_mask(cell)
    #     find_energy_min(eptm)


        
def solve_all_rosettes(eptm, **kwargs):
    
    rosettes = find_rosettes(eptm)
    print('solving %i rosettes' %len(rosettes))
    new_jvs = []
    while len(rosettes):
        for central_vert in rosettes:
            new_jv  = solve_rosette_opt(eptm, central_vert, **kwargs)
            new_jvs.append(new_jv)
        rosettes = find_rosettes(eptm)
    return new_jvs

@hdf_snapshot
@png_snapshot
def solve_rosette_opt(eptm, central_vert, **kwargs):

    new_jv  = solve_rosette(eptm, central_vert, **kwargs)
    find_energy_min(eptm)
    return new_jv
        
@hdf_snapshot
@png_snapshot
def type1_at_shorter(eptm, local_edges):
    
    edge_lengths = np.array([eptm.edge_lengths[je] 
                             for je in local_edges])
    shorter_edge = local_edges[np.argmin(edge_lengths)]
    modified_cells, modified_jverts = type1_transition(eptm,
                                                       shorter_edge)
    eptm.set_local_mask(modified_cells[0], wider=True)
    eptm.set_local_mask(modified_cells[1], wider=True)
    out = find_energy_min(eptm)
    return modified_cells, modified_jverts
        
# def increase_contractility(eptm, cells, contractility_increase):
#     for cell in cells:
#         eptm.cells.contractilities[cell] *= contractility_increase 
        
# def increase_tension(eptm, edges, tension_increase):
#     for edge in edges:
#         eptm.junctions.line_tensions[edge] *= tension_increase

def tension_increase(t, theta, tau, max_ti=2.):
    delay = tau * np.sin(theta/2)**2
    exp = np.exp(-(t - delay) / tau)
    increase = 1 + max_ti * (1 - exp)
    return increase.clip(1, 1+max_ti)

def contractility_increase(t, theta, tau, max_ci=2.):
    delay = tau * np.sin(theta/2)**2
    exp = np.exp(-(t - delay) / tau)
    increase = 1 + (max_ci - 1) * (1 - exp)
    return increase.clip(1, max_ci)

def find_ring_jes(eptm, ring_width):

    eptm.set_local_mask(None)
    local_slice(eptm, theta_amp=2*np.pi,
                zed_amp=ring_width)

    is_ring = eptm.is_junction_edge.copy()
    is_ring.a[:] = 0

    for cell in eptm.cells.local_cells():
        for je in eptm.cells.junctions[cell]:
            n0, n1 = eptm.junctions.adjacent_cells[je]
            if not (eptm.is_local_vert[n0]
                    and eptm.is_local_vert[n1]):
                is_ring[je] = 1

    for cell in eptm.cells.local_cells():
        are_rings = np.array([is_ring[je] for je
                              in eptm.cells.junctions[cell]])
        if np.any(are_rings):
            eptm.is_local_vert[cell] = 0

    for cell in eptm.cells.local_cells():
        for je in eptm.cells.junctions[cell]:
            n0, n1 = eptm.junctions.adjacent_cells[je]
            if not (eptm.is_local_vert[n0]
                    and eptm.is_local_vert[n1]):
                is_ring[je] = 1
    
    return is_ring

def induce_contractility(eptm, a_cell, max_ci, rate_ci, span=1):
    """
    """
    
    focus_on_cell(eptm, a_cell, radius=3*span)
    c0 = eptm.params['contractility']
    eptm.graph.set_directed(False)
    for cell in eptm.cells.local_cells():
        dist = gt.shortest_distance(eptm.graph,
                                    source=a_cell, target=cell) / 2.
        increase = 1 + (rate_ci - 1) * np.exp((1 - dist) / span)
        new_c = eptm.cells.contractilities[cell] * increase
        eptm.cells.contractilities[cell] = min(new_c, max_ci*c0)
    eptm.graph.set_directed(True)

def induce_tension(eptm, a_cell, max_ti, rate_ti, span=1):
    """
    """
    
    focus_on_cell(eptm, a_cell, radius=3*span)
    t0 = eptm.params['line_tension']
    eptm.graph.set_directed(False)
    for cell in eptm.cells.local_cells():
        dist = gt.shortest_distance(eptm.graph,
                                    source=a_cell, target=cell) / 2.
        increase = 1 + (rate_ti - 1) * np.exp((1 - dist) / span)
        for je in eptm.cells.junctions[cell]:
            new_t = eptm.junctions.line_tensions[je] * increase
            eptm.junctions.line_tensions[je] = min(new_t, max_ti*t0)
    eptm.graph.set_directed(True)

    
@hdf_snapshot
@png_snapshot
def gradual_apoptosis(eptm, seq_kwargs,
                      apopto_kwargs, post_kwargs):
    
    (apopto_cells, fold_cells,
     apopto_sequence) = get_apoptotic_cells(eptm, **seq_kwargs)
    prev_first = apopto_cells[0]
    for n, sub_sequence in enumerate(apopto_sequence):
        for a_cell in sub_sequence:
            apoptosis_step(eptm, a_cell, **apopto_kwargs)
            first = sub_sequence[0]
            if first != prev_first:
                post_apoptosis(eptm, prev_first,
                               fold_cells, **post_kwargs)
            prev_first = first
    post_apoptosis(eptm, prev_first,
                   fold_cells,
                   **post_kwargs)
    
    wide_relax(eptm)
    xml_name = os.path.join(eptm.paths['xml'], 'after_apopto.xml')
    eptm.graph.save(xml_name)

@hdf_snapshot
@png_snapshot
def wide_relax(eptm, zed_amp=20):
    
    local_slice(eptm, theta_c=0, zed_c=0,
                theta_amp=2*np.pi, zed_amp=zed_amp)
    eptm.graph.set_vertex_filter(eptm.is_cell_vert)
    running_local_optimum(eptm, tol=1e-3)
    eptm.graph.set_vertex_filter(None)
    
def show_death_pattern(eptm):

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

