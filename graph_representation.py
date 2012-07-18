#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from numpy.random import normal, random_sample
import graph_tool.all as gt
import pylab as plt

from mpl_toolkits.mplot3d import Axes3D

FLOAT = np.dtype('float32')
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = os.path.join(ROOT_DIR, 'default', 'params.xml')



def plot_gradients(epithelium):
    epithelium.junctions.graph.set_vertex_filter(epithelium.junctions.is_local)
    j_vertices = [j_vert for j_vert in epithelium.junctions.graph.vertices()]
    pos0 = np.array([epithelium.junctions.sz_pos[j_vert]
                     for j_vert in epithelium.junctions.graph.vertices()])
    epithelium.junctions.graph.set_vertex_filter(None)
    grad = epithelium.local_gradient(pos0.flatten(), j_vertices)
    grad_vec = grad.reshape(pos0.shape) + pos0
    plot_cells_sz(epithelium, local=True)
    for g, p in zip(grad_vec, pos0):
        plt.plot([p[0], g[0]], [p[1], g[1]], 'r-', lw=4, alpha=0.5)
        

def plot_cells_sz(epithelium, local=True, index='True'):
    if local:
        epithelium.cells.graph.set_vertex_filter(
            epithelium.cells.is_local)
    for cell in epithelium.cells.graph.vertices():
        plt.text(epithelium.cells.sz_pos[cell][0],
                 epithelium.cells.sz_pos[cell][1],
                 str(cell))
        edge_list = [edge for edge in epithelium.cells.junctions_edges[cell]]
        plot_edges_sz(epithelium.junctions, edge_list)
    if local:
        epithelium.cells.graph.set_vertex_filter(None)

def plot_edges_sz(junctions, edge_list, **kwargs):
    sigmas = []
    zeds = []
    for edge in edge_list:
        if not edge.is_valid():
            print edge
            continue
        sigmas = (junctions.sigmas[edge.source()],
                  junctions.sigmas[edge.target()])
        zeds = (junctions.zeds[edge.source()],
                junctions.zeds[edge.target()])
        plt.plot(sigmas, zeds, 'k-', lw=2, alpha=0.5, **kwargs)
        plt.text(junctions.sigmas[edge.source()],
                 junctions.zeds[edge.source()],
                 str(edge.source()))
        plt.text(junctions.sigmas[edge.target()],
                 junctions.zeds[edge.target()],
                 str(edge.target()))
    ax = plt.gca()
    ax.set_aspect('equal')


def sfdp_draw(graph, output="lattice_3d.pdf", **kwargs):
    output = os.path.join('drawings', output)
    sfdp_pos = gt.graph_draw(graph,
                             pos=gt.sfdp_layout(graph,
                                                cooling_step=0.95,
                                                epsilon=1e-3,
                                                multilevel=True),
                             output_size=(300,300),
                             output=output)
    print 'graph view saved to %s' %output
    return sfdp_pos

def pseudo3d_draw(graph, rtz, output="lattice_3d.pdf",
                  z_angle=0.12, theta_rot=0.1,
                  RGB=(0.8, 0.1, 0.), **kwargs):
    rhos, thetas, zeds = rtz
    thetas += theta_rot
    output = os.path.join('drawings', output)
    red, green, blue = RGB
    
    pseudo_x = graph.new_vertex_property('float')
    pseudo_y = graph.new_vertex_property('float')
    vertex_red = graph.new_vertex_property('float')
    vertex_green = graph.new_vertex_property('float')
    vertex_blue = graph.new_vertex_property('float')
    vertex_alpha = graph.new_vertex_property('float')
    
    pseudo_x.a = zeds * np.cos(z_angle) - rhos * np.cos(thetas) * np.sin(z_angle)
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
    return pseudo3d_pos
    
def epithelium_draw(epithelium, z_angle=0.15, d_theta=0.1,
                    output="tissue_3d.pdf", output2='tissue_sz.pdf',
                    **kwargs):

    g1 = epithelium.junctions.graph
    g2 = epithelium.cells.graph

    vertex_red1 = g1.new_vertex_property('float')
    vertex_green1 = g1.new_vertex_property('float')
    vertex_blue1 = g1.new_vertex_property('float') 
    vertex_alpha1 = g1.new_vertex_property('float') 
    vertex_size1 = g1.new_vertex_property('int') 

    edge_red1 = g1.new_edge_property('float')
    edge_green1 = g1.new_edge_property('float')
    edge_blue1 = g1.new_edge_property('float') 
    edge_alpha1 = g1.new_edge_property('float') 
    edge_width1 = g1.new_edge_property('float') 

    vertex_red2 = g2.new_vertex_property('float')
    vertex_green2 = g2.new_vertex_property('float')
    vertex_blue2 = g2.new_vertex_property('float')
    vertex_alpha2 = g2.new_vertex_property('float') 
    vertex_size2 = g2.new_vertex_property('int') 


    edge_red2 = g2.new_edge_property('float')
    edge_green2 = g2.new_edge_property('float')
    edge_blue2 = g2.new_edge_property('float') 
    edge_alpha2 = g2.new_edge_property('float') 
    edge_width2 = g2.new_edge_property('float') 

    vertex_red1.a[:] = 105/256.
    vertex_green1.a[:] = 182/256.
    vertex_blue1.a[:] = 40/256.
    vertex_size1.a[:] = 1.

    edge_red1.a[:] = 105/256.
    edge_green1.a[:] = 201/256.
    edge_blue1.a[:] = 40/256.
    edge_width1.a[:] = 1.

    vertex_red2.a[:] = 105/256.
    vertex_green2.a[:] = 201/256.
    vertex_blue2.a[:] = 237/256.
    vertex_size2.a[:] = 5.


    edge_red2.a[:] = 0.
    edge_green2.a[:] = 0.
    edge_blue2.a[:] = 0.
    edge_width2.a[:] = 0.

    props = [(epithelium.junctions.rtz_pos, epithelium.cells.rtz_pos),
             (edge_red1, edge_red2),
             (edge_green1, edge_green2),
             (edge_blue1, edge_blue2),
             (edge_alpha1, edge_alpha2),
             (vertex_red1, vertex_red2),
             (vertex_green1, vertex_green2),
             (vertex_blue1, vertex_blue2),
             (vertex_alpha1, vertex_alpha2),
             (edge_width1, edge_width2),
             (vertex_size1, vertex_size2)]

    ug, u_props = gt.graph_union(g1, g2, props=props)

    (rtz_pos, edge_red, edge_green, edge_blue, edge_alpha,
     vertex_red, vertex_green, vertex_blue, vertex_alpha,
     edge_width, vertex_size) = u_props
    rhos, thetas, zeds = gt.ungroup_vector_property(rtz_pos, [0, 1, 2])

    pseudo_x = ug.new_vertex_property('float')
    pseudo_y = ug.new_vertex_property('float')
    pseudo_x.a = zeds.a * np.cos(z_angle) - rhos.a * np.cos(
        thetas.a + d_theta) * np.sin(z_angle)
    pseudo_y.a = rhos.a * np.sin(thetas.a + d_theta)

    depth = rhos.a * (1 - np.cos(thetas.a + d_theta))
    normed_depth = (depth - depth.min()) / (depth.max() - depth.min())
    vertex_alpha.a = normed_depth * 0.7 + 0.3
    for edge in ug.edges():
        edge_alpha[edge] = vertex_alpha[edge.source()]

    vorder = ug.new_vertex_property('float') 
    vorder.a = np.argsort(vertex_alpha.a)

    eorder = ug.new_edge_property('float') 
    eorder.a = np.argsort(edge_alpha.a)
    
    vertex_rgba = [vertex_red, vertex_green, vertex_blue, vertex_alpha]
    vertex_color = gt.group_vector_property(vertex_rgba, value_type='float')
    edge_rgba = [edge_red, edge_green, edge_blue, edge_alpha]
    edge_color = gt.group_vector_property(edge_rgba, value_type='float')
    
    xy = [pseudo_x, pseudo_y]
    pseudo3d_pos = gt.group_vector_property(xy, value_type='float')
    
    pmap = gt.graph_draw(ug, pseudo3d_pos,
                         vertex_fill_color=vertex_color,
                         vertex_color=vertex_color,
                         edge_pen_width=edge_width, 
                         edge_color=edge_color,
                         vertex_size=vertex_size,
                         vorder=vorder, eorder=eorder,
                         output=output, **kwargs)

    
    sigma = ug.new_vertex_property('float')
    sigma.a = rhos.a * thetas.a
    zs = [sigma, zeds]
    zs_pos = gt.group_vector_property(zs, value_type='float')
    pmap2 = gt.graph_draw(ug, zs_pos,
                          vertex_fill_color=vertex_color,
                          vertex_color=vertex_color,
                          edge_pen_width=edge_width, 
                          edge_color=edge_color,
                          vertex_size=vertex_size,
                          vorder=vorder, eorder=eorder,
                          output=output2, **kwargs)


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
