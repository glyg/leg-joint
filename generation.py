#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import mkdir
import numpy as np
# Import mpltlib before graph-tool to
# avoid Gtk seg fault (don't ask why)
import matplotlib.pyplot as plt
import graph_tool.all as gt
import time
from datetime import datetime

import leg_joint as lj
import random


# eptm = lj.Epithelium(paramfile='default/few_big_cells.xml')
def new_generation(eptm, growth_rate=1.5, pola=False):
    eptm.graph.set_vertex_filter(eptm.is_cell_vert)
    cells =  [cell for cell in eptm.graph.vertices()
              if eptm.is_alive[cell]]
    print 'there are %i cells to devide' % len(cells)
    num = 0
    eptm.graph.set_vertex_filter(None)
    t0  = time.time()
    try:
        mkdir('tmp')
    except OSError:
        pass
    if pola:
        phi = eptm.dsigmas.copy()
    random.seed(3)
    random.shuffle(cells)
    while len(cells):
        mother_cell = cells.pop()
        print('dividing cell %s' %str(mother_cell))
        skip = False
        # No division
        # Mother cell might have died earlier in the loop
        if not eptm.is_alive[mother_cell]:
            print('skipping dead cell %s' % str(mother_cell))
            skip = True
        for je in eptm.cells.junctions[mother_cell]:
            try:
                cell0, cell1 = eptm.junctions.adjacent_cells[je]
                if not eptm.is_alive[cell0]:
                    print('skipping dead cell %s' % str(cell0))
                    skip = True
                    if not eptm.is_alive[cell1]:
                        print('skipping dead cell %s' % str(cell1))
                        skip = True
            except ValueError:
                skip = True
        if skip: continue
        eptm.set_local_mask(None)
        eptm.set_local_mask(mother_cell)
        eptm.cells.prefered_vol[mother_cell] *= growth_rate
        pos0, pos1 = lj.find_energy_min(eptm, tol=1e-5)
        
        eptm.isotropic_relax()
        #rand_phi = np.random.normal(0, np.pi/8.)
        j = lj.cell_division(eptm, mother_cell,
                             #phi_division=rand_phi,
                             verbose=False)
        if pola:
            eptm.update_tensions(phi, np.pi/3)
        if j is not None:
            pos0, pos1 = lj.find_energy_min(eptm, tol=1e-5)
            eptm.isotropic_relax()
            small_cells = [cell for cell in eptm.cells
                           if eptm.cells.areas[cell] < 1e-1]
            if len(small_cells) > 0:
                for small_cell in small_cells:
                    if eptm.is_alive[small_cell]:
                        print 'removing cell %s' % str(small_cell)
                        lj.remove_cell(eptm, small_cell)
            eptm.graph.save("saved_graphs/xml/latest.xml")
            now = datetime.now()
            outfname3d = 'saved_graphs/png/tmp/generation_3d_%03i.png' % num
            outfname2d = 'saved_graphs/png/tmp/generation_sz_%03i.png' % num
            lj.draw(eptm, output2d=outfname2d,
                    output3d=outfname3d)
            
        else:
            print 'division failed'
        num += 1
        elapsed = time.time() - t0
        time_left = (elapsed / num) * (len(cells) - num)
        print str(num)+'/'+str(len(cells))
        print 'time left: %3f' % time_left

    eptm.isotropic_relax()
    eptm.update_geometry()
    eptm.params['n_zeds'] *= 2
    eptm.params['n_sigmas'] *= 2
    eptm.graph.save("saved_graphs/xml/generation%s.xml" % now.isoformat())

    
if __name__ == '__main__':
    #graph = gt.load_graph('saved_graphs/xml/initial_graph.xml')
    eptm = lj.Epithelium(#graphXMLfile='saved_graphs/xml/initial_graph.xml',
                         paramfile='default/params.xml')

    pola = False
    eptm.isotropic_relax()
    lj.running_local_optimum(eptm, tol=1e-3, pola=pola, save_to=None)
    lj.draw(eptm)
    if pola:
        eptm.graph.save('saved_graphs/xml/initial_squeezed.xml')
    else:
        eptm.graph.save('saved_graphs/xml/initial_graph.xml')
    for n in range(2):
        new_generation(eptm, pola)
        lj.running_local_optimum(eptm, tol=1e-3, pola=pola, save_to=None)
        eptm.isotropic_relax()

    eptm.graph.save('saved_graphs/xml/before_apoptosis.xml')