#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import IPython.core.display as disp

import sys, os
curdir = os.path.abspath(os.path.curdir)
#print os.path.dirname(curdir)
sys.path.append(os.path.dirname(curdir))

import graph_tool.all as gt
#import matplotlib.pyplot as plt
import leg_joint as lj



def before_after(func):
    def new_func(eptm, *args, **kwargs):
        import matplotlib.pyplot as plt
        import leg_joint as lj
        #eptm.update_gradient()
        fig, axes = plt.subplots(1,2, figsize=(12,4),
                                 sharex=True, sharey=True)
        subaxes = lj.plot_ortho_proj(eptm, axes[0], c_text=False, 
                                     vfilt=eptm.is_local_vert,
                                     efilt=eptm.is_local_edge)
        lj.plot_ortho_gradients(eptm, subaxes, scale=1.)
        foutput = func(eptm, *args, **kwargs)
        #eptm.update_gradient()
        subaxes = lj.plot_ortho_proj(eptm, axes[1], c_text=False,
                                     vfilt=eptm.is_local_vert,
                                     efilt=eptm.is_local_edge)
        lj.plot_ortho_gradients(eptm, subaxes, scale=1.)
        return foutput
    return new_func

@before_after
def local_optimum(eptm, **kwargs):
    #eptm.update_gradient()
    pos0, pos1 = lj.find_energy_min(eptm, **kwargs)
    return pos0, pos1



