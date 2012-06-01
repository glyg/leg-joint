#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from numpy.random import normal, random_sample
import graph_tool.all as gt
import pylab as plt

from mpl_toolkits.mplot3d import Axes3D

FLOAT = np.dtype('float64')
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = os.path.join(ROOT_DIR, 'default', 'params.xml')



def cylindrical2cartesian(rtz):
    """
    Transforms cylindrical coordinates
    ::math:(\rho, \theta, z):
    into cartesian ::math:(x, y, z):
    """
    
    xs = rtz['rho'] * np.cos(rtz['theta'])
    ys = rtz['rho'] * np.sin(rtz['theta'])
    zeds = rtz['zed']
    num_points = rtz.shape[0]
    
    xyz_dtype = np.dtype([('ix', FLOAT),
                          ('wy', FLOAT),
                          ('zed',FLOAT)])
    xyz = np.zeros((num_points,), dtype=xyz_dtype)
    xyz['ix'] = xs
    xyz['wy'] = ys
    xyz['zed'] = zeds   
    return xyz

def vertices_projections(rtz, **kwards):
    """
    Plots a figure with the various 2D projections
    """

    figure, axes = plt.subplots((2,2))
    basalapical_ax = axes[1,0]
    basalapical_ax.plot(rtz['zed'], rtz['rho'],
                        'ro', alpha=0.3)
    basalapical_ax.set_xlabel(r'Proximo - distal axis $z$')
    basalapical_ax.set_ylabel(r'Basal - apical axis, $\rho$')
    basalapical_ax.set_aspect('equal')
    
    curv_ax = axes[1,1]
    curv_ax.plot(rtz['zed'], rtz['rho'] *  rtz['theta'],
                 'o-', alpha=0.3)
    curv_ax.set_aspect('equal')
    curv_ax.set_xlabel(r"Proximo - distal axis $z$")
    curv_ax.set_ylabel(r"Curvilinear $\sigma = \rho\theta$")

    cut_ax =  axes[0,0]
    cut_ax.plot(rtz['rho'] * np.cos(rtz['theta']),
                 rtz['rho'] * np.sin(rtz['theta']),
                 'o-', alpha=0.3)
    cut_ax.set_aspect('equal')
    cut_ax.set_xlabel(u'Anterior - posterior axis (µm)')
    cut_ax.set_ylabel(u'Ventral - dorsal axis (µm)')

def vertices_scatterplot(rtz, **kwargs):
    """
    **kwards are passed to `ax.scatter()`
    """

    fig = plt.figure()

    ax_3d = fig.add_subplot(111, projection='3d')
    xyz = cylindrical2cartesian(rtz)

    ax_3d.scatter(xyz['ix'], #Named after F.Herbert
                  xyz['wy'],
                  xyz['zed'],
                  **kwargs)
    ax_3d.set_aspect('equal')
    ax_3d.set_xlabel(u'Anterior - posterior axis (µm)')
    ax_3d.set_ylabel(u'Ventral - dorsal axis (µm)')
    ax_3d.set_zlabel(u'Proximal - distal axis (µm)')    
    plt.show()

    return fig, ax_3d 
