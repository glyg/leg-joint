#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from numpy.random import normal, random_sample

import graph_tool as gt
from graph_tool.generation import geometric_graph
from graph_tool.draw import graph_draw

import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

RHO_0 = 60.
RHO_NOISE = 0.01
LAMBDA0 = 6


def compute_vertices_coordinates(rho_0=RHO_0,
                                 rho_noise=RHO_NOISE,
                                 lambda_avg=LAMBDA0):

    surface = 8 * np.pi * rho_0**2
    num_vertices = np.int(surface / (lambda_avg**2))
    params = num_vertices, rho_0, rho_noise

    rhos_body = normal(rho_0, rho_noise, num_vertices // 2)
    thetas_body = random_sample(num_vertices // 2) * 2 * np.pi 
    zeds_body = np.linspace(-rho_0, rho_0, num_vertices // 2)
    rtz_body = np.array([rhos_body, thetas_body, zeds_body])
    rtz_capA = generate_cap(-1, params)
    rtz_capB = generate_cap(1, params)
    
    return rtz_body, rtz_capA, rtz_capB

def generate_cap(sign, params):

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

    figure, axes = plt.subplots(3)
    figure.set_size_inches(6, 12)
    basalapical_ax = axes[0]
    basalapical_ax.plot(rtz[2,:], rtz[0,:], '.', alpha=0.5)
    basalapical_ax.set_xlabel(r'Proximo -- distal axis $z$')
    basalapical_ax.set_ylabel(r'Basal -- apical axis, $\rho$')
    basalapical_ax.set_aspect('equal')

    hist_ax = axes[1]
    rho_hist = hist_ax.hist(rtz[0,:] * rtz[1,:], bins=50)
    hist_ax.set_xlabel(r'$\rho$')

    curv_ax = axes[2]
    curv_ax.plot(rtz[2,:], rtz[1,:] *  rtz[2,:],'.',alpha=0.5)
    curv_ax.set_aspect('equal')
    curv_ax.set_xlabel(r"Proximo - distal axis $z$")
    curv_ax.set_ylabel(r"Curvilinear coordiante $\sigma = \rho\theta$")



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
