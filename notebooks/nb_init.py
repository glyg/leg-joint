#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import IPython.core.display as disp

import sys, os
curdir = os.path.abspath(os.path.curdir)
#print os.path.dirname(curdir)
sys.path.append(os.path.dirname(curdir))

import leg_joint as lj


def before_after(func):
    def new_func(eptm, *args, **kwargs):
        import matplotlib.pyplot as plt
        import leg_joint as lj
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        lj.plot_cells_sz(eptm, axes[0], c_text=False, 
                              vfilt=eptm.is_local_vert,
                              efilt=eptm.is_local_edge)
        lj.plot_gradients(eptm, axes[0])
        foutput = func(eptm, *args, **kwargs)
        lj.plot_cells_sz(eptm, axes[1], c_text=False,
                              vfilt=eptm.is_local_vert,
                              efilt=eptm.is_local_edge)
        lj.plot_gradients(eptm, axes[1])
        return foutput
    return new_func

@before_after
def local_optimum(eptm, **kwargs):
    # eptm.update_gradient()
    pos0, pos1 = eptm.find_energy_min(**kwargs)
    return pos0, pos1



