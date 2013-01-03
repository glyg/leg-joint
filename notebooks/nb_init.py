#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import IPython.core.display as disp

import sys, os
curdir = os.path.abspath(os.path.curdir)
#print os.path.dirname(curdir)
sys.path.append(os.path.dirname(curdir))

import leg_joint as lj
import graph_tool.all as gt
import matplotlib.pyplot as plt
    

def before_after(func):
    def new_func(eptm, *args, **kwargs):
        import matplotlib.pyplot as plt
        import leg_joint as lj
        fig, axes = plt.subplots(1,2, figsize=(12,4),
                                 sharex=True, sharey=True)
        subaxes = lj.plot_ortho_proj(eptm, axes[0], c_text=False, 
                                     vfilt=eptm.is_local_vert,
                                     efilt=eptm.is_local_edge)
        lj.plot_ortho_gradients(eptm, subaxes)
        foutput = func(eptm, *args, **kwargs)
        eptm.update_gradient()
        subaxes = lj.plot_ortho_proj(eptm, axes[1], c_text=False,
                                     vfilt=eptm.is_local_vert,
                                     efilt=eptm.is_local_edge)
        lj.plot_ortho_gradients(eptm, subaxes)
        return foutput
    return new_func

@before_after
def local_optimum(eptm, **kwargs):
    eptm.update_gradient()
    eptm.update_radial_grad()
    pos0, pos1 = eptm.find_energy_min(**kwargs)
    eptm.update_radial_grad()
    return pos0, pos1



