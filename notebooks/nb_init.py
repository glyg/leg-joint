#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys, os
curdir = os.path.abspath(os.path.curdir)
print os.path.sep.join(curdir.split(os.path.sep)[:-2])
sys.path.append(os.path.sep.join(curdir.split(os.path.sep)[:-2]))

import leg_joint as lj

import matplotlib.pyplot as plt
import IPython.core.display as disp

def before_after(func):
    def new_func(eptm, *args, **kwargs):
        
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        lj.draw.plot_cells_sz(eptm, axes[0], c_text=False, 
                              vfilt=eptm.is_local_vert,
                              efilt=eptm.is_local_edge)
        lj.draw.plot_gradients(eptm, axes[0])
        foutput = func(eptm, *args, **kwargs)
        lj.draw.plot_cells_sz(eptm, axes[1], c_text=False,
                              vfilt=eptm.is_local_vert,
                              efilt=eptm.is_local_edge)
        lj.draw.plot_gradients(eptm, axes[1])
        return foutput
    return new_func

@before_after
def local_optimum(eptm, tol):
    # eptm.update_gradient()
    pos0, pos1 = eptm.find_energy_min(tol=tol)
    return pos0, pos1



