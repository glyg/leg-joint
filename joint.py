#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

def p_apopto(zed, theta, z0=0., width_apopto=1.5, p0=0.95):
    p = p0 * np.exp(- (zed - z0)**2 / width_apopto**2
                    ) * (0.3 + 0.7 * np.abs(np.cos(theta/2.)))
    return p

def get_apoptotic_cells(eptm, seed, **kwargs):
    ''' '''
    is_apoptotic = eptm.is_alive.copy()
    is_apoptotic.a[:] = 0
    np.random.seed(seed)
    for cell in eptm.cells:
        dice = np.random.random()
        if dice < p_apopto(eptm.zeds[cell], eptm.thetas[cell], **kwargs):
            is_apoptotic[cell] = 1
    print "Number of apoptotic cells: %i" % is_apoptotic.a.sum()

    apopto_cells = np.array([cell for cell in eptm.cells
                             if is_apoptotic[cell]])
    thetas = np.array([eptm.thetas[cell] for cell in apopto_cells])
    theta_idx = np.argsort(np.cos(thetas/2))[::-1]
    return apopto_cells.take(theta_idx)

def get_sequence(apopto_cells, num_steps):
    apopto_sequence = []
    num_cells = apopto_cells.size
    for k in np.arange(- num_steps, num_cells + num_steps):
        start = max(k, 0)
        stop = min(k + num_steps, num_cells)
        apopto_sequence.append(apopto_cells[start:stop])
    return apopto_sequence

def gradual_apoptosis(eptm, apopto_cells, num_steps, **kwargs):

    apopto_sequence = get_sequence(apopto_cells, num_steps)
    print '%i steps will be performed' % len(apopto_sequence)

    i = 0
    for cell_seq in apopto_sequence:
        for a_cell in cell_seq:
            i += 1
            lj.apoptosis(eptm, a_cell, idx=i, **kwargs)
            eptm.update_geometry()
        
def show_distribution(eptm):
    lj.local_slice(eptm, zed_amp=2., theta_amp=None)
    axes = lj.plot_edges_generic(eptm, eptm.zeds, eptm.wys, ctoj=False)#,
                                 #efilt=eptm.is_local_edge)
    x_min = eptm.ixs.a.min()
    x_max = eptm.ixs.a.max()
    for cell in eptm.cells:
        if is_apoptotic[cell]:
            alpha = 0.3 + 0.7 * (eptm.ixs[cell] - x_min)/(x_max - x_min)
            axes.plot(eptm.zeds[cell], eptm.wys[cell], 'ro', alpha=alpha)
    return axes
            
    
if __name__ == '__main__':
    eptm = lj.Epithelium(graphXMLfile='saved_graphs/xml/before_apoptosis.xml',
                         paramfile='default/params.xml')

    import sys
    args = [np.float(arg) if '.' in arg else np.int(arg)
            for arg in sys.argv[1:]]
    if len(args) != 5:
        print("""Usage:
              You need to provide the values for 5 parameters: 
              python joint.py p1 p2 p3 p4 p5
              with the following meaning:
              p1 : seed for the random process
              p2 : number of steps to complete apoptosis
              p3 : volume reduction of the cell at each step
              p4 : contractility multiplication at each steps 
              p5 : radial tension
              """ )
        raise ValueError('Bad number of parameters')
    (seed, num_steps, vol_reduction, contractility, radial_tension) = args

    apopto_cells = get_apoptotic_cells(eptm, seed)
    gradual_apoptosis(eptm, apopto_cells, num_steps,
                      vol_reduction=vol_reduction,
                      contractility=contractility,
                      radial_tension=radial_tension)
    