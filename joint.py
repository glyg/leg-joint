# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from os import mkdir
import os
import numpy as np
# Import mpltlib before graph-tool to
# avoid Gtk seg fault (don't ask why)
import matplotlib.pyplot as plt
import graph_tool.all as gt     # 
import time
from datetime import datetime
import random
import logging


CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
GRAPH_SAVE_DIR = os.path.join(ROOT_DIR, 'saved_graphs')


import leg_joint as lj

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
            is_apoptotic[cell] = 1
        eptm.graph.set_vertex_filter(None)

    n_cells = is_apoptotic.a.sum()
    #print("Number of apoptotic cells: %i" % n_cells)

    apopto_cells = np.array([cell for cell in eptm.cells
                             if is_apoptotic[cell]])
    thetas = np.array([eptm.thetas[cell] for cell in apopto_cells])
    theta_idx = np.argsort(np.cos(thetas/2))#[::-1]
    return apopto_cells.take(theta_idx)

def specific_apopto_cells_number(eptm, num_cells, **kwargs):
    n_apopto = 0
    n_iter = 0
    seed = 10
    eptm.set_local_mask(None)
    lj.local_slice(eptm, zed_amp=kwargs['width_apopto'], theta_amp=2*np.pi)
    eptm.graph.set_vertex_filter(eptm.is_local_vert)
    fold_cells = np.array([cell for cell in eptm.cells])
    
    while n_apopto != num_cells:
        seed += 1
        n_iter += 1
        if n_iter > 100:
            raise RuntimeError('''Number of trials to high, 
                               Try changing the parameters''')
        apopto_cells = get_apoptotic_cells2(num_cells, fold_cells, seed=seed, **kwargs)
        n_apopto = len(apopto_cells)
    print('Iterated %i times' % n_iter)
    return apopto_cells, seed

def get_apoptotic_cells2(num_cells, fold_cells,  seed=42, **kwargs):
    
    total_cells = len(fold_cells)
    all_probas = np.array([p_apopto(eptm.zeds[cell],
                                    eptm.thetas[cell],
                                    **kwargs)
                           for cell in fold_cells])
    all_probas *=  num_cells /  all_probas.sum()
    np.random.seed(seed)
    dices = np.random.random(size=all_probas.size)
    apopto_cells = fold_cells[all_probas > dices]
    eptm.graph.set_vertex_filter(None)
    return apopto_cells

def get_sequence(apopto_cells, num_steps):
    apopto_sequence = []
    num_cells = apopto_cells.size
    for k in np.arange(- num_steps + 1, num_cells):
        start = max(k, 0)
        stop = min(k + num_steps, num_cells)
        apopto_sequence.append(apopto_cells[start:stop])
    return apopto_sequence
    
def gradual_apoptosis(eptm, apopto_cells, num_steps=10, residual_tension=0.,
                      fold_width=1.8, pola=False, tension_increase=None,
                      **kwargs):

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
            if tension_increase is not None:
                lj.enhance_tension(eptm, new_jv, tension_increase=2.)
            # for cell in fold_cells:
            #     if not eptm.is_alive[cell]:
            #         continue
            #     eptm.set_local_mask(None)
            #     eptm.set_local_mask(cell)
            #     lj.find_energy_min(eptm)
            #     if pola:
            #         eptm.update_tensions(phi, np.pi / 4, 1.26**4)
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
    # lj.running_local_optimum(eptm, tol=1e-6)

    
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

def ventral_enhance(thetas_in, gamma=1):
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

### Those will be the same across all simulations ran
fixed_params =  {'width_apopto':1.8,
                 'residual_tension': 0.,
                 'p0': 0.1,
                 'amp': 0.4,
                 'num_steps': 10.,
                 'vol_reduction':0.8,
                 'contractility': 1.0} # Constant contractility

### Those will change
grid_params = {'n_cells': [30, 5, 10, 20],
              'tension_increase': [1., 2., 1.2],
              'radial_tension': [0., 0.2, 0.1],
              'ventral_bias':[1, 0]}

grid_size = (len(grid_params['n_cells']) * len(grid_params['tension_increase'])
             * len(grid_params['radial_tension']) * 2)
total_apoptoses = grid_size * np.sum(grid_params['n_cells'])
total_steps = total_apoptoses * fixed_params['num_steps']

print('Total number of simulations: %i' % grid_size)
print('Total number of apoptoses: %i' % total_apoptoses)
print('Total number of apoptoses: %i' % total_steps)
    
if __name__ == '__main__':

    import sys
    try:
        assert(len(sys.argv[1]) == 1)
        core_num = int(sys.argv[1])
    except:
        """Usage: python joint.py core_num"""

    ## Making chunks
    n_cores = 6
    chunk_size = grid_size // (2 * n_cores)
    start = core_num * chunk_size
    stop = (core_num + 1) * chunk_size - 1
    
    grid_elements = []
    for n_cells in grid_params['n_cells']:
        for radial_tension in grid_params['radial_tension']:
            for tension_increase in grid_params['radial_tension']:
                grid_elements.append((n_cells, tension_increase,
                                      radial_tension))
    apopto_seq_kws = {'width_apopto':fixed_params['width_apopto'],
                      'amp':fixed_params['amp']}
    gradual_apopto_kws = {'num_steps':fixed_params['num_steps'],
                          'fold_width':fixed_params['width_apopto'],
                          'residual_tension':fixed_params['residual_tension'],
                          'vol_reduction':fixed_params['vol_reduction'],
                          'contractility':fixed_params['contractility'],
                          'pola':False}
    print(len(grid_elements))
    
    xml_dir = os.path.join(os.path.join(GRAPH_SAVE_DIR, 'xml'))
    if not os.path.isdir(xml_dir):
        raise(IOError, 'Directory %s not found ' % xml_dir )

    print(start, stop)
    for params in grid_elements[start:stop]:

        n_cells, tension_increase, radial_tension = params

        ## Ventral bias
        
        identifier = 'N_{}_TI_{}_RT_{}_VB'.format(n_cells,
                                                  tension_increase,
                                                  radial_tension)
        with open('core_num_%i.log' % core_num, 'w+') as log_txt:
            log_txt.write('starting %s\n' % identifier)
        
        eptm = lj.Epithelium(
            graphXMLfile='saved_graphs/xml/before_apoptosis.xml',
            paramfile='default/params.xml')
        apopto_cells_rnd, seed = specific_apopto_cells_number(eptm, n_cells,
                                                              **apopto_seq_kws)
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        for cell in apopto_cells_rnd:
            ax.plot(eptm.wys[cell], eptm.ixs[cell], 'ko', alpha=0.7)
            ax.set_title('Sequence of apoptoses around the joint')
        ax.set_aspect('equal')
        fig.savefig('doc/imgs/repartition'+identifier+'.svg')
        plt.close(fig)
        gradual_apoptosis(eptm, apopto_cells_rnd,
                          tension_increase=tension_increase,
                          radial_tension=radial_tension,
                          save_dir=identifier,
                          **gradual_apopto_kws)
        out_xml = os.path.join(xml_dir, 'apopto_%s.xml' % identifier)
        eptm.graph.save(out_xml)

        with open('core_num_%i.log' % core_num, 'w+') as log_txt:
            log_txt.write('done %s\n' % identifier)

        
        ## No ventral bias

        identifier = 'N_{}_TI_{}_RT_{}_RG'.format(n_cells,
                                                  tension_increase,
                                                  radial_tension)
        
        if tension_increase==2. and radial_tension in (0, 0.2):
            with open('core_num_%i.log' % core_num, 'w+') as log_txt:
                log_txt.write('starting %s\n' % identifier)

            eptm = lj.Epithelium(
                graphXMLfile='saved_graphs/xml/before_apoptosis.xml',
                paramfile='default/params.xml')
            apopto_cells_reg = get_apoptotic_cells(eptm, random=False,
                                                   gamma=1, n_cells=n_cells,
                                                   width_apopto=2)

            fig, ax = plt.subplots(figsize=(2.5, 2.5))
            for cell in apopto_cells_reg:
                ax.plot(eptm.wys[cell], eptm.ixs[cell], 'ko', alpha=0.7)
                ax.set_title('Sequence of apoptoses around the joint')
            ax.set_aspect('equal')
            fig.savefig('doc/imgs/repartition'+identifier+'.svg')
            plt.close(fig)

            gradual_apoptosis(eptm, apopto_cells_reg,
                              tension_increase=tension_increase,
                              radial_tension=radial_tension,
                              save_dir=identifier,
                              **gradual_apopto_kws)

            out_xml = os.path.join(xml_dir, 'apopto_%s.xml' % identifier)
            eptm.graph.save(out_xml)
    
            with open('core_num_%i.log' % core_num, 'w+') as log_txt:
                log_txt.write('done %s\n' % identifier)
            
