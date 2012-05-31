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
LAMBDA0 = 6.


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
    params = num_vertices, lambda_avg, rho_0, rho_noise
    
    rtz_body = generate_body(params)
    rtz_capA = generate_cap(-1, params)
    rtz_capB = generate_cap(1, params)
    rtz_all = np.hstack([rtz_capA, rtz_body, rtz_capB])
    num_vertices = rtz_all.shape[1]
    return rtz_all, num_vertices

def generate_body(params, random=False):
    """
    Return the vertices on the cylindrical part of the
    epithelium
    """
    num_vertices, lambda_avg, rho_0, rho_noise = params
    if random:
        rhos_body = normal(rho_0, rho_noise, num_vertices // 2)
        thetas_body = random_sample(num_vertices // 2) * 2 * np.pi - np.pi
        zeds_body = np.linspace(-rho_0, rho_0, num_vertices // 2)
    else:
        n = np.floor(2 * np.pi * rho_0 / lambda_avg)
        n = int(n)
        m = np.floor( 2 * rho_0 / lambda_avg)
        thetas_body = np.array([i * lambda_avg / rho_0
                                for i in range(n)] * m).flatten() - np.pi
        rhos_body = np.ones(n * m) * rho_0 
        zeds_body = np.array([np.ones(n) * z * lambda_avg
                              for z in range(int(m))]).flatten()
        zeds_body -= zeds_body.max() / 2.
        
    return np.array([rhos_body, thetas_body, zeds_body])

def generate_cap(sign, params, random=False):
    """
    building capA and capB.
    """
    num_vertices, lambda_avg, rho_0, rho_noise = params
    if random:
        phis_cap = np.linspace(0, np.pi / 2., num_vertices // 4) 
        r_cap = normal(rho_0, rho_noise, num_vertices // 4)
        rhos_cap = r_cap * np.cos(phis_cap)
        thetas_cap = random_sample(num_vertices // 4) * 2 * np.pi - np.pi
        zeds_cap = sign * r_cap * np.sin(phis_cap) + sign * rho_0 
    else:
        m = int(0.5 * np.pi * rho_0 / lambda_avg)
        rtz_cap = np.array([[0], [0], [rho_0]])
        for i in range(m):
            psi_i = (m - i) * lambda_avg / rho_0
            n_i = np.floor(2 * np.pi * rho_0 * np.sin(psi_i) / lambda_avg)
            n_i = int(n_i)
            theta_i = np.array([k * 2 * np.pi / n_i
                                for k in range(n_i)]) - np.pi
            rho_i =  np.ones(n_i) * rho_0 * np.sin(psi_i)
            zed_i = rho_0 * np.ones(n_i)  * np.cos(psi_i)
            zed_i = sign * (zed_i + rho_0)

            rtz_i = np.array([rho_i, theta_i, zed_i])
            rtz_cap = np.hstack((rtz_cap, rtz_i))
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

    figure, axes = plt.subplots(3, sharex=True)
    basalapical_ax = axes[0]
    basalapical_ax.plot(rtz[2,:], rtz[0,:],
                        'r.', alpha=0.1, ms=2)
    basalapical_ax.set_xlabel(r'Proximo - distal axis $z$')
    basalapical_ax.set_ylabel(r'Basal - apical axis, $\rho$')
    basalapical_ax.set_aspect('equal')

    curv_ax = axes[1]
    curv_ax.plot(rtz[2,:], rtz[0,:] *  rtz[1,:] / (2 * np.pi),
                 'o-', alpha=0.3)
    curv_ax.set_aspect('equal')
    curv_ax.set_xlabel(r"Proximo - distal axis $z$")
    curv_ax.set_ylabel(r"Curvilinear $\sigma = \rho\theta/2\pi$")

    cut_ax =  axes[2]
    cut_ax.plot(rtz[0,:] * np.cos(rtz[1,:]),
                 rtz[0,:] * np.sin(rtz[1,:]),
                 'o-', alpha=0.3)
    cut_ax.set_aspect('equal')
    curv_ax.set_xlabel(r'Anterior - posterior axis (µm)')
    curv_ax.set_ylabel(r'Ventral - dorsal axis (µm)')


def vertices_scatterplot(rtz, **kwargs):
    """
    **kwards are passed to `ax.scatter()`
    """

    fig = plt.figure()

    ax_3d = fig.add_subplot(111, projection='3d')
    xyz = cylindrical2cartesian(rtz)

    ax_3d.scatter(xyz[0], xyz[1], xyz[2], 'ko-')
    ax_3d.set_aspect('equal')
    ax_3d.set_xlabel(r'Anterior - posterior axis (µm)')
    ax_3d.set_ylabel(r'Ventral - dorsal axis (µm)')
    ax_3d.set_xlabel(r'Proximal - distal axis (µm)')    
    plt.show()
