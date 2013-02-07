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

# eptm = lj.Epithelium(paramfile='default/few_big_cells.xml')
def new_generation(eptm, growth_rate=1.8):
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
    random.seed(3)
    random.shuffle(cells)
    for mother_cell in cells:
        print('dividing cell %s' %str(mother_cell))
        skip = False
        # No division 
        for je in eptm.cells.junctions[mother_cell]:
            cell0, cell1 = eptm.junctions.adjacent_cells[je]
            if not eptm.is_alive[cell0]:
                print('skipping dead cell %s' % str(cell0))
                skip = True
            if not eptm.is_alive[cell1]:
                print('skipping dead cell %s' % str(cell1))
                skip = True
        if skip: continue
        eptm.set_local_mask(None)
        eptm.set_local_mask(mother_cell)
        eptm.cells.prefered_vol[mother_cell] *= growth_rate
        pos0, pos1 = lj.find_energy_min(eptm, tol=1e-3, approx_grad=0)
        j = lj.cell_division(eptm, mother_cell, verbose=False)
        if j is not None:
            pos0, pos1 = lj.find_energy_min(eptm, tol=1e-3, approx_grad=0)
            now = datetime.now()
            # eptm.graph.save("saved_graphs/xml/tmp/generation%s.xml"
            #                 % now.isoformat())
            # outfname3d = 'saved_graphs/png/tmp/generation_3d_%03i.png' % num
            # outfname2d = 'saved_graphs/png/tmp/generation_sz_%03i.png' % num
            # lj.draw(eptm, output2d=outfname2d,
            #         output3d=outfname3d)
            #lj.optimizers.isotropic_optimum(eptm, 1e-6)
            #lj.resolve_small_edges(eptm, threshold=0.25)
        else:
            print 'division failed'
        num += 1
        elapsed = time.time() - t0
        time_left = (elapsed / num) * (len(cells) - num)
        print str(num)+'/'+str(len(cells))
        print 'time left: %3f' % time_left
    # eptm.anisotropic_relax()
    eptm.update_geometry()
    eptm.params['n_zeds'] *= 2
    eptm.params['n_sigmas'] *= 2
    eptm.graph.save("saved_graphs/xml/generation%s.xml" % now.isoformat())

    
if __name__ == '__main__':
    eptm = lj.Epithelium(graphXMLfile='saved_graphs/xml/initial_graph.xml',
                         paramfile='default/params.xml')
    lj.optimizers.isotropic_optimum(eptm, 1e-6)
    for n in range(3):
        new_generation(eptm)
        lj.optimizers.isotropic_optimum(eptm, 1e-6)
    # z_min = eptm.zeds.fa.min()
    # z_max = eptm.zeds.fa.max()
    # zed0 = (z_max - z_min) / 3.
    # zed1 = 2. * (z_max - z_min) / 3.
    # lj.create_frontier(eptm, zed0, tension_increase=4.)
    # lj.create_frontier(eptm, zed1, tension_increase=4.)
