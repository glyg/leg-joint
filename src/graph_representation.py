#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from numpy.random import normal, random_sample
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import graph_tool.all as gt

import filters

FLOAT = np.dtype('float32')

@filters.active
def plot_gradients(epithelium, ax=None, scale=1.):
    epithelium.update_gradient()
    grad = epithelium.gradient_array()
    sigmas = epithelium.sigmas.fa
    zeds = epithelium.zeds.fa
    grad_sigmas = grad[::2] * scale
    grad_zeds = grad[1::2] * scale
    v_sigmas = np.array([sigmas, grad_sigmas]).T
    v_zeds = np.array([zeds, grad_zeds]).T
    if ax is None:
        ax =  plot_cells_sz(epithelium, ax=None,
                            vfilt=epithelium.is_local_vert,
                            efilt=epithelium.is_local_edge)
    for s, z in zip(v_sigmas, v_zeds):
        ax.arrow(z[0], s[0], z[1], s[1], width=0.1,
                 ec='red', fc='red', alpha=0.5)
    plt.draw()
    return ax

@filters.active
def plot_active(epithelium, ax=None):
    sigmas = epithelium.sigmas.fa
    zeds = epithelium.zeds.fa
    if ax is None:
        ax =  plot_cells_sz(epithelium, ax=None,
                            vfilt=epithelium.is_local_vert,
                            efilt=epithelium.is_local_edge)
    ax.plot(zeds, sigmas,  'ro', alpha=0.5, ms=8)
    plt.draw()
    return ax
    
def plot_cells_sz(epithelium, ax=None, text=True,
                  vfilt=None, efilt=None,
                  c_text=True, j_text=False):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    epithelium.graph.set_vertex_filter(vfilt)
    sigmas = epithelium.sigmas.copy()
    zeds = epithelium.zeds.copy()
    for cell in epithelium.cells :
        if text and c_text:
            ax.text(zeds[cell],
                    sigmas[cell],
                    str(cell))
        ax.plot(zeds[cell], # FIXME this is not what the name suggests
                sigmas[cell], 'bo', alpha=0.3)
    epithelium.graph.set_vertex_filter(None)
    epithelium.graph.set_edge_filter(efilt)
    plot_edges_sz(epithelium, efilt, ax=ax, text=j_text)
    epithelium.graph.set_edge_filter(None)
    ax.set_xlabel('Proximo distal', fontsize='large')
    ax.set_ylabel('Around the leg', fontsize='large')
    plt.draw()
    return ax

def plot_edges_sz(epithelium, efilt=None,
                  text=False, ax=None, **kwargs):
    sigmas = []
    zeds = []
    if ax is None:
        fig, ax = plt.subplots(1,1)
    epithelium.graph.set_edge_filter(efilt)
    epithelium.graph.set_vertex_filter(None)
    for edge in epithelium.junctions:
        if edge is None:
            print "invalid edge %s" %str(edge)
            continue
        sigmas = (epithelium.sigmas[edge.source()],
                  epithelium.sigmas[edge.target()])
        zeds = (epithelium.zeds[edge.source()],
                epithelium.zeds[edge.target()])
        ax.plot(zeds,
                sigmas,
                'go-', lw=2, alpha=0.4, **kwargs)
        if text:
            ax.text(epithelium.zeds[edge.source()],
                    epithelium.sigmas[edge.source()],
                    str(edge.source()))
            ax.text(epithelium.zeds[edge.target()],
                    epithelium.sigmas[edge.target()],
                    str(edge.target()))
    ax.set_aspect('equal')
    epithelium.graph.set_edge_filter(None)




def sfdp_draw(graph, output="lattice_3d.pdf"):
    output = os.path.join('saved_graph/pdf', output)
    sfdp_pos = gt.graph_draw(graph,
                             pos=gt.sfdp_layout(graph,
                                                cooling_step=0.95,
                                                epsilon=1e-3,
                                                multilevel=True),
                             output_size=(300,300),
                             output=output)
    # print 'graph view saved to %s' %output
    return sfdp_pos


def pseudo3d_draw(graph, rtz, output="lattice_3d.pdf",
                  z_angle=0.12, theta_rot=0.1,
                  RGB=(0.8, 0.1, 0.), **kwargs):
    rhos, thetas, zeds = rtz
    thetas += theta_rot
    output = os.path.join('saved_graph/pdf', output)
    red, green, blue = RGB
    
    pseudo_x = graph.new_vertex_property('float')
    pseudo_y = graph.new_vertex_property('float')
    vertex_red = graph.new_vertex_property('float')
    vertex_green = graph.new_vertex_property('float')
    vertex_blue = graph.new_vertex_property('float')
    vertex_alpha = graph.new_vertex_property('float')
    
    pseudo_x.a = zeds * np.cos(z_angle)\
                 - rhos * np.cos(thetas) * np.sin(z_angle)
    pseudo_y.a = rhos * np.sin(thetas)
    depth = rhos * np.cos(thetas)
    normed_depth = (depth - depth.min()) / (depth.max() - depth.min())
    vertex_alpha.a = normed_depth * 0.7 + 0.3
    vertex_red.a = np.ones(rhos.shape, dtype=np.float) * red
    vertex_green.a = np.ones(rhos.shape, dtype=np.float) * green 
    vertex_blue.a = np.ones(rhos.shape, dtype=np.float) * blue
    rgba = [vertex_red, vertex_green, vertex_blue, vertex_alpha]
    pseudo3d_color = gt.group_vector_property(rgba, value_type='float')
    
    xy = [pseudo_x, pseudo_y]
    pseudo3d_pos = gt.group_vector_property(xy, value_type='float')
    pmap = gt.graph_draw(graph, pseudo3d_pos,
                         vertex_fill_color=pseudo3d_color,
                         vertex_color=pseudo3d_color,
                         edge_pen_width=2., 
                         output=output, **kwargs)
    del pmap
    return pseudo3d_pos
    
def epithelium_draw(eptm, z_angle=0.15, d_theta=0.1,
                    output3d="tissue_3d.pdf",
                    output2d='tissue_sz.pdf', verbose=False):

    file_type = output3d.split('.')[-1]
    output3d = os.path.join('../saved_graphs', file_type, output3d)
    output2d = os.path.join('../saved_graphs', file_type, output2d)

    graph = eptm.graph
    vertex_red = eptm.graph.new_vertex_property('float')
    vertex_green = eptm.graph.new_vertex_property('float')
    vertex_blue = eptm.graph.new_vertex_property('float') 
    vertex_alpha = eptm.graph.new_vertex_property('float') 
    vertex_size = eptm.graph.new_vertex_property('int') 

    edge_red = eptm.graph.new_edge_property('float')
    edge_green = eptm.graph.new_edge_property('float')
    edge_blue = eptm.graph.new_edge_property('float') 
    edge_alpha = eptm.graph.new_edge_property('float') 
    edge_width = eptm.graph.new_edge_property('float') 

    j_filt = eptm.is_cell_vert.copy()
    j_filt.a *= (1 - eptm.is_alive.a)
    eptm.graph.set_vertex_filter(j_filt,
                                 inverted=True)
    if verbose: print eptm.graph.num_vertices()
    vertex_red.fa = 105/256.
    vertex_green.fa = 182/256.
    vertex_blue.fa = 40/256.
    vertex_size.fa = 1.
    eptm.graph.set_vertex_filter(None)

    eptm.graph.set_edge_filter(eptm.is_junction_edge,
                               inverted=False)
    edge_red.fa = 105/256.
    edge_green.fa = 201/256.
    edge_blue.fa = 40/256.
    edge_width.fa = 1.
    eptm.graph.set_edge_filter(None)

    cell_filt = eptm.is_cell_vert.copy()
    cell_filt.a *= eptm.is_alive.a
    eptm.graph.set_vertex_filter(cell_filt,
                                 inverted=False)
    vertex_red.fa = 105/256.
    vertex_green.fa = 201/256.
    vertex_blue.fa = 237/256.
    vertex_size.fa = 5.
    eptm.graph.set_vertex_filter(None)

    eptm.graph.set_edge_filter(eptm.is_ctoj_edge,
                               inverted=False)
    edge_red.fa = 0.9
    edge_green.fa = 0.9
    edge_blue.fa = 0.9
    edge_width.fa = 0.3
    eptm.graph.set_edge_filter(None)

    rhos, sigmas, zeds = eptm.rhos, eptm.sigmas, eptm.zeds
    thetas = sigmas.copy()
    thetas.a = sigmas.a / rhos.a
    
    pseudo_x = eptm.graph.new_vertex_property('float')
    pseudo_y = eptm.graph.new_vertex_property('float')
    pseudo_x.a = zeds.a * np.cos(z_angle) - rhos.a * np.cos(
        thetas.a + d_theta) * np.sin(z_angle)
    pseudo_y.a = rhos.a * np.sin(thetas.a + d_theta)

    depth = rhos.a * (1 - np.cos(thetas.a + d_theta))
    normed_depth = (depth - depth.min()) / (depth.max() - depth.min())
    vertex_alpha.a = normed_depth * 0.6 + 0.4 * eptm.is_alive.a
    
    for edge in eptm.graph.edges():
        edge_alpha[edge] = vertex_alpha[edge.source()]

    vorder = eptm.graph.new_vertex_property('float') 
    vorder.a = np.argsort(vertex_alpha.a)

    eorder = eptm.graph.new_edge_property('float') 
    eorder.a = np.argsort(edge_alpha.a)
    
    vertex_rgba = [vertex_red, vertex_green, vertex_blue, vertex_alpha]
    vertex_color = gt.group_vector_property(vertex_rgba, value_type='float')
    edge_rgba = [edge_red, edge_green, edge_blue, edge_alpha]
    edge_color = gt.group_vector_property(edge_rgba, value_type='float')
    
    xy = [pseudo_x, pseudo_y]
    pseudo3d_pos = gt.group_vector_property(xy, value_type='float')
    pmap = gt.graph_draw(eptm.graph, pseudo3d_pos,
                         vertex_fill_color=vertex_color,
                         vertex_color=vertex_color,
                         edge_pen_width=edge_width, 
                         edge_color=edge_color,
                         vertex_size=vertex_size,
                         vorder=vorder, eorder=eorder,
                         output=output3d)
    if verbose: print 'saved tissue to %s' % output3d
    
    sigma = eptm.sigmas
    zs = [zeds, sigma]
    zs_pos = gt.group_vector_property(zs, value_type='float')
    pmap2 = gt.graph_draw(eptm.graph, zs_pos,
                          vertex_fill_color=vertex_color,
                          vertex_color=vertex_color,
                          edge_pen_width=edge_width, 
                          edge_color=edge_color,
                          vertex_size=vertex_size,
                          vorder=vorder, eorder=eorder,
                          output=output2d)
    if verbose: print 'saved tissue to %s' % output2d
    del pmap, pmap2

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
    figure, axes = plt.subplots(2, 2)
    basalapical_ax = axes[1,1]
    basalapical_ax.plot(rtz['zed'], rtz['rho'],
                        'o', alpha=0.3)

    basalapical_ax.axis((rtz['zed'].min() * 1.1, rtz['zed'].max() * 1.1,
                         0, rtz['rho'].max() * 1.1))
    basalapical_ax.set_xlabel(r'Proximo - distal axis $z$')
    basalapical_ax.set_ylabel(r'Basal - apical axis, $\rho$')
    basalapical_ax.set_aspect('equal')


    ax_3d = figure.add_subplot(2, 2, 1, projection='3d')
    xyz = cylindrical2cartesian(rtz)

    ax_3d.scatter(xyz['ix'], #Named after F.Herbert
                  xyz['wy'],
                  xyz['zed'])
    ax_3d.set_aspect('equal')
    ax_3d.set_xlabel(u'Anterior - posterior axis (µm)')
    ax_3d.set_ylabel(u'Ventral - dorsal axis (µm)')
    ax_3d.set_zlabel(u'Proximal - distal axis (µm)')    
    
    curv_ax = axes[0,1]
    curv_ax.plot(rtz['zed'], rtz['rho'] *  rtz['theta'],
                 'o-', alpha=0.3)
    curv_ax.set_aspect('equal')
    curv_ax.set_xlabel(r"Proximo - distal axis $z$")
    curv_ax.set_ylabel(r"Curvilinear $\sigma = \rho\theta$")

    cut_ax =  axes[1,0]
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
