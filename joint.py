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
    random.shuffle(cells)
    for mother_cell in cells:
        eptm.set_local_mask(None)
        eptm.set_local_mask(mother_cell)
        eptm.cells.prefered_area[mother_cell] *= growth_rate
        eptm.update_geometry()
        eptm.update_gradient()
        pos0, pos1 = eptm.find_energy_min(tol=1e-4, approx_grad=1)
        j = lj.cell_division(eptm, mother_cell, verbose=False)
        eptm.update_geometry()
        eptm.update_gradient()
        if j is not None:
            pos0, pos1 = eptm.find_energy_min(tol=1e-4, approx_grad=1)
            now = datetime.now()
            eptm.graph.save("saved_graphs/xml/tmp/generation%s.xml"
                            % now.isoformat())
            outfname3d = 'saved_graphs/png/tmp/generation_3d_%03i.png' % num
            outfname2d = 'saved_graphs/png/tmp/generation_sz_%03i.png' % num
            lj.draw(eptm, output2d=outfname2d,
                    output3d=outfname3d)
            eptm.isotropic_relax()
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
    eptm = lj.Epithelium(paramfile='default/params.xml')
    for n in range(4):
        new_generation(eptm)
    # z_min = eptm.zeds.fa.min()
    # z_max = eptm.zeds.fa.max()
    # zed0 = (z_max - z_min) / 3.
    # zed1 = 2. * (z_max - z_min) / 3.
    # lj.create_frontier(eptm, zed0, tension_increase=4.)
    # lj.create_frontier(eptm, zed1, tension_increase=4.)
