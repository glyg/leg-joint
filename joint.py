# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from os import mkdir
import numpy as np
# Import mpltlib before graph-tool to
# avoid Gtk seg fault (don't ask why)
import matplotlib.pyplot as plt
import graph_tool.all as gt     # 
import time
from datetime import datetime

import leg_joint as lj
import random

def p_apopto(zed, theta, z0=0., width_apopto=1.5, p0=0.95, amp=0.7):
    p = p0 * np.exp(- (zed - z0)**2 / width_apopto**2
                    ) * (1  - amp * (np.cos(theta/2)**2))
    return p

def get_apoptotic_cells(eptm, seed=42, random=True, gamma=1,
                        n_cells=1, **kwargs):
    ''' '''
    
    is_apoptotic = eptm.is_alive.copy()
    is_apoptotic.a[:] = 0
    np.random.seed(seed)
    if random:
        for cell in eptm.cells:
            dice = np.random.random()
            if dice < p_apopto(eptm.zeds[cell], eptm.thetas[cell], **kwargs):
                is_apoptotic[cell] = 1
        n_cells = is_apoptotic.a.sum()
    else:
        eptm.update_rhotheta()
        thetas_in = np.linspace(0, 2*np.pi,
                                num=n_cells,
                                endpoint=False)
        thetas_out = ventral_enhance(thetas_in, gamma) - np.pi
        sigmas_out = thetas_out * eptm.rhos.a.mean()
        zeds_out = np.random.normal(0,
                                    scale=kwargs['width_apopto'],
                                    size=n_cells)
        #eptm.graph.set_vertex_filter(eptm.is_cell_vert)
        for sigma, zed in zip(sigmas_out, zeds_out):
            vfilt = eptm.is_cell_vert.copy()
            vfilt.a -= is_apoptotic.a
            eptm.graph.set_vertex_filter(vfilt)
            cell = eptm.closest_vert(sigma, zed)
            # while is_apoptotic[cell]:
            #     zed += np.random.normal(**kwargs['width_apopto'])
            is_apoptotic[cell] = 1
        eptm.graph.set_vertex_filter(None)

    n_cells = is_apoptotic.a.sum()
    print("Number of apoptotic cells: %i" % n_cells)

    apopto_cells = np.array([cell for cell in eptm.cells
                             if is_apoptotic[cell]])
    thetas = np.array([eptm.thetas[cell] for cell in apopto_cells])
    theta_idx = np.argsort(np.cos(thetas/2))#[::-1]
    return apopto_cells.take(theta_idx)

def ventral_enhance(thetas_in, gamma=1):
    
    thetas_in = np.atleast_1d(thetas_in)
    thetas_out = np.zeros_like(thetas_in)
    gamma_low = lambda x, gamma: np.pi * (x / np.pi)**gamma
    gamma_high = lambda x, gamma: np.pi * (2 - (2 - x / np.pi)**(gamma)) 
    thetas_out[thetas_in <= np.pi] = gamma_low(thetas_in[thetas_in <= np.pi],
                                               gamma)
    thetas_out[thetas_in > np.pi] = gamma_high(thetas_in[thetas_in > np.pi],
                                               gamma)
    return thetas_out
    
    
def get_sequence(apopto_cells, num_steps):
    apopto_sequence = []
    num_cells = apopto_cells.size
    for k in np.arange(- num_steps + 1, num_cells):
        start = max(k, 0)
        stop = min(k + num_steps, num_cells)
        apopto_sequence.append(apopto_cells[start:stop])
    return apopto_sequence

def gradual_apoptosis(eptm, apopto_cells, num_steps, residual_tension=0.,
                      fold_width=1.8, pola=False, **kwargs):

    apopto_sequence = get_sequence(apopto_cells, num_steps)
    print('%i steps will be performed' % (len(apopto_cells) * num_steps))
    phi = eptm.dsigmas.copy()
    phi.a[:] = 0
    lj.local_slice(eptm, theta_amp=None, zed_c=0., zed_amp=fold_width)
    fold_cells = np.array([cell for cell in eptm.cells
                           if eptm.is_local_vert[cell]])
    # thetas = np.array([eptm.thetas[cell] for cell in fold_cells])
    # theta_idx = np.argsort(np.cos(thetas/2))
    # fold_cells = fold_cells[theta_idx]
    eptm.set_local_mask(None)
    i = 0
    prev_first = apopto_cells[0]
    for cell_seq in apopto_sequence:
        for a_cell in cell_seq:
            i += 1
            lj.apoptosis(eptm, a_cell, idx=i, **kwargs)
            # eptm.update_geometry()
            if pola:
                eptm.update_tensions(phi, np.pi / 4, 1.26**4)
        # for cell in fold_cells:
        #     if not eptm.is_alive[cell]:
        #         continue
        #     eptm.set_local_mask(None)
        #     eptm.set_local_mask(cell, wider=True)
        #     lj.find_energy_min(eptm)
        #     if pola:
        #         eptm.update_tensions(phi, np.pi/4, 1.26**4)
                
        first = cell_seq[0]
        if first != prev_first:
            new_jv = lj.remove_cell(eptm, prev_first)
            eptm.junctions.radial_tensions[new_jv] = residual_tension
            if pola:
                eptm.update_tensions(phi, np.pi / 4, 1.26**4)
            np.random.shuffle(fold_cells)
            for cell in fold_cells:
                if not eptm.is_alive[cell]:
                    continue
                eptm.set_local_mask(None)
                eptm.set_local_mask(cell)
                lj.find_energy_min(eptm)
                if pola:
                    eptm.update_tensions(phi, np.pi / 4, 1.26**4)
        prev_first = first
                
    new_jv = lj.remove_cell(eptm, prev_first)
    eptm.junctions.radial_tensions[new_jv] = residual_tension
    #lj.running_local_optimum(eptm, tol=1e-6)

def specific_apopto_cells_number(num_cells, *args, **kwargs):
    n_apopto = 0
    n_iter = 0
    seed = 0
    while n_apopto != num_cells:
        seed += 1
        n_iter += 1
        if n_iter > 100:
            raise RuntimeError('''Number of trials to high, 
                               Try changing the parameters''')
        apopto_cells = get_apoptotic_cells(*args, seed=seed, **kwargs )
        n_apopto = len(apopto_cells)
    
    return seed, apopto_cells
    
def show_distribution(eptm):
    lj.local_slice(eptm, zed_amp=2., theta_amp=None)
    axes = lj.plot_edges_generic(eptm, eptm.zeds, eptm.wys, ctoj=False)#,
                                 #efilt=eptm.is_local_edge)
    x_min = eptm.ixs.a.min()
    x_max = eptm.ixs.a.max()
    for cell in eptm.cells:
        if eptm.is_apoptotic[cell]:
            alpha = 0.3 + 0.7 * (eptm.ixs[cell] - x_min)/(x_max - x_min)
            axes.plot(eptm.zeds[cell], eptm.wys[cell], 'ro', alpha=alpha)
    return axes
            
    
if __name__ == '__main__':

    import sys
    args = [np.float(arg) if '.' in arg else np.int(arg)
            for arg in sys.argv[1:]]
    if len(args) != 4:
        print("""Usage:
              You need to provide the values for 5 parameters: 
              python joint.py p1 p2 p3 p4
              with the following meaning:
              p1 : number of steps to complete apoptosis
              p2 : volume reduction of the cell at each step
              p3 : contractility multiplication at each steps 
              p4 : radial tension
              """ )
        raise ValueError('Bad number of parameters')
    (num_steps, vol_reduction, contractility, radial_tension) = args

    conditions = {# 'ectopic': {'width_apopto':100,
                  #             'p0': 0.02,
                  #             'amp': 0.,
                  #             'residual_tension': 0.,
                  #             'seed': 0},
                  # 'no_theta_bias': {'width_apopto':1.8,
                  #                   'p0': 0.6,
                  #                   'amp': 0.,
                  #                   'residual_tension': 0.,
                  #                   'seed': 3},
                  # 'residual_tension': {'width_apopto':1.8,
                  #                      'p0': 0.75,
                  #                      'amp': 0.4,
                  #                      'residual_tension': 1.,
                  #                      'seed': 3},
                  '05_cells': {'width_apopto':1.8,
                               'residual_tension': 0.,
                               'p0': 0.1,
                               'amp': 0.4,
                               'n_cells': 5},
                  '10_cells': {'width_apopto':1.8,
                               'residual_tension': 0.,
                               'p0': 0.2,
                               'amp': 0.4,
                               'n_cells': 10},
                  '15_cells': {'width_apopto':1.8,
                               'residual_tension': 0.,
                               'p0': 0.4,
                               'amp': 0.4,
                               'n_cells': 15},
                  '20_cells': {'width_apopto':1.8,
                               'residual_tension': 0.,
                               'p0': 0.5,
                               'amp': 0.4,
                               'n_cells': 20},
                  '25_cells': {'width_apopto':1.8,
                               'residual_tension': 0.,
                               'p0': 0.6,
                               'amp': 0.4,
                               'n_cells': 25},
                  '30_cells': {'width_apopto':1.8,
                               'p0': 0.7,
                               'amp': 0.4,
                               'residual_tension': 0.,
                               'n_cells': 30}
        }

    for cond, params in conditions.items():
        print('**************'
              'Starting %s'
              '**************' % cond)
        eptm = lj.Epithelium(
            graphXMLfile='saved_graphs/xml/before_apoptosis.xml',#
            paramfile='default/params.xml')
        eptm.isotropic_relax()
        seed, apopto_cells_rnd = specific_apopto_cells_number(params['n_cells'], eptm, 
                                                              width_apopto=params['width_apopto'],
                                                              p0=params['p0'],
                                                              amp=params['amp'])

        # apopto_cells_rnd = get_apoptotic_cells(eptm,
        #                                        seed=params['seed'],
        #                                        width_apopto=params['width_apopto'],
        #                                        p0=params['p0'],
        #                                        amp=params['amp'])
        save_dir='random_{0}'.format(cond)
 
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        for cell in apopto_cells_rnd:
            ax.plot(eptm.wys[cell], eptm.ixs[cell], 'ko', alpha=0.7)
            ax.set_title('Sequence of apoptoses around the joint')
        ax.set_aspect('equal')
        fig.savefig('doc/imgs/repartition'+save_dir+'.svg')
        plt.close(fig)
        gradual_apoptosis(eptm, apopto_cells_rnd, num_steps,
                          fold_width=params['width_apopto'],
                          residual_tension=params['residual_tension'],
                          vol_reduction=vol_reduction,
                          contractility=contractility,
                          radial_tension=radial_tension,
                          save_dir=save_dir,
                          pola=False)
        
    params = {'width_apopto':1.8,
              'residual_tension': 0.,
              'p0': 0.5,
              'amp': 0.4,
              'seed': 2}
        
    for n_cells in [5, 10, 15, 30]:
    #     for gamma in [1, 1.2, 2.]:
            
        eptm = lj.Epithelium(
            graphXMLfile='saved_graphs/xml/before_apoptosis.xml',#
            paramfile='default/params.xml')
        eptm.isotropic_relax()
        gamma = 1
        apopto_cells_reg = get_apoptotic_cells(eptm, random=False,
                                               gamma=gamma, n_cells=n_cells,
                                               width_apopto=2)
        save_dir='regular_{}cells_gamma_{}'.format(n_cells, gamma)

        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        for cell in apopto_cells_reg:
            ax.plot(eptm.wys[cell], eptm.ixs[cell], 'ko', alpha=0.7)
            ax.set_title('Sequence of apoptoses around the joint')
        ax.set_aspect('equal')
        fig.savefig('doc/imgs/repartition'+save_dir+'.svg')
        plt.close(fig)

        gradual_apoptosis(eptm, apopto_cells_reg, num_steps,
                          fold_width=params['width_apopto'],
                          residual_tension=params['residual_tension'],
                          vol_reduction=vol_reduction,
                          contractility=contractility,
                          radial_tension=radial_tension,
                          save_dir=save_dir,
                          pola=False)
            
