#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from numpy.random import normal, random_sample

import graph_tool.all as gt
from graph_tool.generation import geometric_graph, triangulation
from graph_tool.draw import graph_draw

import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

# TODO: proper parameter handling
RHO_0 = 60.
RHO_NOISE = 0.01
LAMBDA0 = 6


def compute_vertices_coordinates(rho_0=RHO_0,
                                 rho_noise=RHO_NOISE,
                                 lambda_avg=LAMBDA0):
    """
    Parameters:
    ===========

    rho_0: float,
        average radius of the central cylinder (`body`) and the caps
    rho_noise: float, the position noise along rho
    lambda_avg: float, th target average edge length

    Returns:
    =======
    rtz_all: (3, num_vertices) ndarray giving
        the rho, theta, zed positons of the vertices
    num_vertices: the number of vertices
    """

    surface = 8 * np.pi * rho_0**2
    cell_surface = 3 * np.sqrt(3) * lambda_avg**2 / 2
    num_vertices = 4 * (np.int(surface / cell_surface) // 4)
    params = num_vertices, rho_0, rho_noise

    rhos_body = normal(rho_0, rho_noise, num_vertices // 2)
    thetas_body = random_sample(num_vertices // 2) * 2 * np.pi - np.pi
    zeds_body = np.linspace(-rho_0, rho_0, num_vertices // 2)
    rtz_body = np.array([rhos_body, thetas_body, zeds_body])
    rtz_capA = generate_cap(-1, params)
    rtz_capB = generate_cap(1, params)
    rtz_all = np.hstack([rtz_capA, rtz_body, rtz_capB])
    
    return rtz_all, num_vertices

def generate_cap(sign, params):
    """
    building capA and capB. We place them on a regular grid
    """
    num_vertices, rho_0, rho_noise = params
    phis_cap = np.linspace(0, np.pi / 2., num_vertices // 4) 
    r_cap = normal(rho_0, rho_noise, num_vertices // 4)

    rhos_cap = r_cap * np.cos(phis_cap)
    thetas_cap = random_sample(num_vertices // 4) * 2 * np.pi - np.pi
    zeds_cap = sign * r_cap * np.sin(phis_cap) + sign * rho_0 

    rtz_cap = np.array([rhos_cap, thetas_cap, zeds_cap])

    return rtz_cap

    

def cylindrical2cartesian(rtz):
    """
    Transforms cylindrical coordinates ::math:(\rho, \theta, z)
    into cartesian ::math:(x, y, z).
    """
    rho, theta, zed = rtz
    xs = rho * np.cos(theta)
    ys = rho * np.sin(theta)
    return xs, ys, zed



def vertices_projections(rtz):
    """
    Plots a figure with the various 2D projections
    """

    figure, axes = plt.subplots(2, sharex=True)
    basalapical_ax = axes[0]
    basalapical_ax.plot(rtz[2,:], rtz[0,:], 'r.', alpha=0.1, ms=2)
    basalapical_ax.set_xlabel(r'Proximo - distal axis $z$')
    basalapical_ax.set_ylabel(r'Basal - apical axis, $\rho$')
    basalapical_ax.set_aspect('equal')

    curv_ax = axes[1]
    curv_ax.plot(rtz[2,:], rtz[0,:] *  rtz[1,:] / (2 * np.pi),
                 'o-', alpha=0.3)
    curv_ax.set_aspect('equal')
    curv_ax.set_xlabel(r"Proximo - distal axis $z$")
    curv_ax.set_ylabel(r"Curvilinear $\sigma = \rho\theta/2\pi$")



def vertices_scatterplot(rtz, **kwargs):
    """
    **kwards are passed to `ax.scatter()`
    """

    fig = plt.figure()

    ax_3d = fig.add_subplot(111, projection='3d')
    xyz = cylindrical2cartesian(rtz)

    ax_3d.scatter(xyz[0], xyz[1], xyz[2], 'ko-')
    ax_3d.set_aspect('equal')
    plt.show()
