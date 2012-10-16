#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import mkdir
import numpy as np
import graph_tool.all as gt
import time
from datetime import datetime

import leg_joint as lj
import random

# eptm = lj.Epithelium(paramfile='default/few_big_cells.xml')


def new_generation(eptm):

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
        eptm.cells.prefered_area[mother_cell] /= 2.
        eptm.cells.contractilities[mother_cell] /= 2.
        eptm.set_local_mask(mother_cell)
        j = lj.cell_division(eptm, mother_cell, verbose=False)
        if j is not None:
            pos0, pos1 = eptm.find_energy_min(tol=1e-5)
            now = datetime.now()
            eptm.graph.save("saved_graphs/xml/tmp/generation%s.xml" % now.isoformat())
            outfname3d = 'saved_graphs/png/tmp/generation_3d_%03i.png' % num
            outfname2d = 'saved_graphs/png/tmp/generation_sz_%03i.png' % num
            lj.draw.epithelium_draw(eptm, output2d=outfname2d, output3d=outfname3d)
        else:
            print 'division failed'
        num += 1
        elapsed = time.time() - t0
        time_left = (elapsed / num) * (len(cells) - num)
        print str(num)+'/'+str(len(cells))
        print 'time left: %3f' % time_left
    eptm.graph.save("saved_graphs/xml/generation%s.xml" % now.isoformat())
    

if __name__ == '__main__':
    eptm = lj.Epithelium(paramfile='default/few_big_cells.xml')
    new_generation(eptm)