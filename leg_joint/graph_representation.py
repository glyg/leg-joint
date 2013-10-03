# -*- coding: utf-8 -*-

import os
import numpy as np

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon


import graph_tool.all as gt

import leg_joint.filters as filters
from .optimizers import precondition, approx_grad
from .utils import to_rhotheta

FLOAT = np.dtype('float32')

def draw_polygons(eptm, coord1, coord2, colors,
                  vfilt=None, efilt=None, ax=None):
    eptm.graph.set_vertex_filter(vfilt)
    eptm.graph.set_edge_filter(efilt)
    polygons = [eptm.cells.polygon(cell, coord1, coord2)[0]
                for cell in eptm.cells if eptm.is_alive[cell]]
    colors = np.array([colors[cell] for cell in eptm.cells
                       if eptm.is_alive[cell]])
    colors /= colors.max()
    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)

    colors = plt.cm.jet(colors)
    if ax is None:
        fig, ax = plt.subplots()
    for poly, color in zip(polygons, colors):
        patch = Polygon(poly, color=color,
                        fill=True, closed=True, alpha=0.4)
        ax.add_patch(patch)
    ax.autoscale_view()
    ax.set_aspect('equal')

    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)
    return ax

def plot_ortho_gradients(eptm, axes=None,
                         scale=1., approx=0, **kwargs):
    '''
    Displays the gradients for the active vertices on top of
    an `ortho_proj` graph
    '''
    if approx == 1:
        grad_xyz = approx_grad(eptm)
        ec = fc = 'blue'
    else:
        grad_xyz = eptm.gradient_array()
        ec = fc = 'red'
    pos0, bounds = precondition(eptm)
    rho_avg = eptm.rhos.a.mean()
    rhos, thetas  = to_rhotheta(pos0[::3], pos0[1::3])
    sigmas = thetas * rho_avg
    zeds = pos0[2::3]
    grad_rhos = (np.cos(thetas) * grad_xyz[::3]
                 + np.sin(thetas) * grad_xyz[1::3]) * scale
    grad_sigmas = (- np.sin(thetas) * grad_xyz[::3] +
                   np.cos(thetas) * grad_xyz[1::3]) * scale
    grad_zeds = grad_xyz[2::3] * scale
    # We plot the forces, which are easier to understand
    v_sigmas = np.array([sigmas, - grad_sigmas]).T
    v_zeds = np.array([zeds, - grad_zeds]).T
    v_radial = np.array([rhos, - grad_rhos]).T
    if axes is None:
        ax_zs, ax_zr, ax_rs =  plot_ortho_proj(eptm, ax=None,
                                               vfilt=eptm.is_local_vert,
                                               efilt=eptm.is_local_edge,
                                               **kwargs)
        axes = ax_zs, ax_zr, ax_rs
    else:
        ax_zs, ax_zr, ax_rs = axes
    for s, z, r in zip(v_sigmas, v_zeds, v_radial):
        ax_zs.arrow(z[0], s[0], z[1], s[1], width=0.01,
                    ec=ec, fc=fc, alpha=0.5)
        ax_zr.arrow(z[0], r[0], z[1], r[1], width=0.01,
                    ec=ec, fc=fc, alpha=0.5)
        ax_rs.arrow(r[0], s[0], r[1], s[1], width=0.01,
                    ec=ec, fc=fc, alpha=0.5)
        
    plt.draw()
    return axes


@filters.active
def plot_active(eptm, xcoord, ycoord, ax=None):
    
    
    xs = xcoord.fa
    ys = ycoord.fa
    if ax is None:
        ax =  plot_cells_generic(eptm, xcoord, ycoord, ax=None,
                                 vfilt=eptm.is_local_vert,
                                 efilt=eptm.is_local_edge)
    ax.plot(xs, ys, 'ro', alpha=0.5, ms=8)
    plt.draw()
    return ax

def plot_two_panels(eptm, ax=None, vfilt=None,
                    efilt=None, local=True,
                    depth_color=None,
                    **kwargs):
    
    if local:
        vfilt = eptm.is_local_vert
        efilt = eptm.is_local_edge
    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = ax.get_figure()
    
    plot_cells_zs(eptm, ax=ax,
                  vfilt=vfilt, efilt=efilt,
                  depth_color=depth_color, **kwargs)

    

    
def plot_ortho_proj(eptm, ax=None, vfilt=None,
                    efilt=None, local=True,
                    depth_color=None,
                    **kwargs):
    if local:
        vfilt = eptm.is_local_vert
        efilt = eptm.is_local_edge
    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = ax.get_figure()
    
    plot_cells_zs(eptm, ax=ax,
                  vfilt=vfilt, efilt=efilt,
                  depth_color=depth_color, **kwargs)

    divider = make_axes_locatable(ax)
    
    ax_zr = divider.append_axes("top", 2., pad=0.2, sharex=ax)
    ax_rs = divider.append_axes("right", 2., pad=0.2, sharey=ax)
    
    plot_cells_zr(eptm, ax=ax_zr,
                  vfilt=vfilt, efilt=efilt,
                  depth_color=depth_color, **kwargs)
    plot_cells_rs(eptm, ax=ax_rs,
                  vfilt=vfilt, efilt=efilt,
                  depth_color=depth_color, **kwargs)
    plt.setp(ax_zr.get_xticklabels() + ax_rs.get_yticklabels(),
             visible=False)
    plt.draw()

    ax_zr.set_xlabel('')
    ax_rs.set_ylabel('')

    axes = ax, ax_zr, ax_rs
    return fig, axes


def plot_cells_generic(eptm, xcoord, ycoord, ax=None, 
                       vfilt=None, efilt=None,
                       c_text=True, j_text=False,
                       depth_color=None,
                       xlabel='', ylabel=''):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    eptm.graph.set_vertex_filter(vfilt)
    for cell in eptm.cells :
        ax.plot(xcoord[cell],
                ycoord[cell], 'bo', alpha=0.1)
        if c_text:
            ax.text(xcoord[cell], ycoord[cell], str(cell))

        
    eptm.graph.set_vertex_filter(None)
    ax = plot_edges_generic(eptm, xcoord, ycoord,
                            efilt, ax, j_text=j_text,
                            depth_color=depth_color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def plot_edges_generic(eptm, xcoord, ycoord, efilt=None,
                       ax=None, j_text=False,
                       depth_color=None, **kwargs):
    edge_width = eptm.junctions.line_tensions.copy()
    if ax is None:
        fig, ax = plt.subplots(1,1)
        
    eptm.graph.set_edge_filter(efilt)
    eptm.graph.set_vertex_filter(None)
    edge_width.fa = 2. * (eptm.junctions.line_tensions.fa
                          / eptm.junctions.line_tensions.fa.mean())**0.5

    if depth_color is not None:
        depth_cmap = plt.cm.jet(depth_color.fa)
        edge_red = depth_color.copy()
        edge_red.fa = depth_cmap[:, 0]
        edge_green = depth_color.copy()
        edge_green.fa = depth_cmap[:, 1]
        edge_blue = depth_color.copy()
        edge_blue.fa = depth_cmap[:, 2]

    for edge in eptm.graph.edges():
        ixs = (xcoord[edge.source()],
               xcoord[edge.target()])
        wys = (ycoord[edge.source()],
               ycoord[edge.target()])
        if eptm.is_junction_edge[edge]:
            if depth_color is not None:
                c = [edge_red[edge], edge_green[edge], edge_blue[edge]]
                ax.plot(ixs, wys, '-', c=c,
                        lw=edge_width[edge],
                        alpha=0.4, **kwargs)
            else:
                ax.plot(ixs, wys, 'g-', lw=edge_width[edge],
                        alpha=0.4, **kwargs)
        # else:
        #     ax.plot(ixs, wys, 'k-', lw=1.,
        #             alpha=0.2, **kwargs)
    ax.set_aspect('equal')
    eptm.graph.set_edge_filter(None)
    return ax

    
def plot_cells_zr(eptm, ax=None, c_text=False,
                  vfilt=None, efilt=None, depth_color=None):
    zeds = eptm.zeds.copy()
    rhos = eptm.rhos.copy()
    ax = plot_cells_generic(eptm, zeds, rhos,
                            ax, vfilt, efilt,
                            c_text=c_text,
                            depth_color=depth_color,
                            xlabel='Proximal - distal',
                            ylabel='Radius')
    return ax

def plot_cells_rs(eptm, ax=None, c_text=False,
                  vfilt=None, efilt=None, depth_color=None):
    rhos = eptm.rhos.copy()
    sigmas = eptm.proj_sigma()
    eptm.update_dsigmas()
    if efilt is not None:
        efilt = efilt.copy()
        efilt.a *= (1 - eptm.at_boundary.a)
    ax = plot_cells_generic(eptm, rhos, sigmas,
                            ax, vfilt, efilt,
                            c_text=c_text,
                            depth_color=depth_color, 
                            xlabel='Radius',
                            ylabel='Latitude')
    return ax

def plot_cells_zs(eptm, ax=None, text=True,
                  vfilt=None, efilt=None,
                  c_text=True, j_text=False, depth_color=None):
    sigmas = eptm.proj_sigma()
    zeds = eptm.zeds.copy()
    eptm.update_dsigmas()
    if efilt is not None:
        efilt = efilt.copy()
        efilt.a *= (1 - eptm.at_boundary.a)
    ax = plot_cells_generic(eptm, zeds, sigmas,
                            ax, vfilt, efilt, c_text,
                            depth_color=depth_color,
                            xlabel='Proximo distal',
                            ylabel='Latitude')
    return ax

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

    eptm.graph.set_directed(False)
    file_type = output3d.split('.')[-1]
    # output3d = os.path.join('saved_graphs', file_type, output3d)
    # output2d = os.path.join('saved_graphs', file_type, output2d)

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
    edge_height = eptm.graph.new_edge_property('float') 

    
    eptm.update_rhotheta()
    rhos, thetas, zeds = eptm.rhos, eptm.thetas, eptm.zeds
    
    pseudo_x = eptm.graph.new_vertex_property('float')
    pseudo_y = eptm.graph.new_vertex_property('float')
    pseudo_x.a = zeds.a * np.cos(z_angle) - rhos.a * np.cos(
        thetas.a + d_theta) * np.sin(z_angle)
    pseudo_y.a = rhos.a * np.sin(thetas.a + d_theta)

    depth = vertex_alpha.copy()
    depth.a = rhos.a * (1 - np.cos(thetas.a + d_theta))
    depth.a = (depth.a - depth.a.min()) / (depth.a.max() - depth.a.min())
    vertex_alpha.a = (depth.a * 0.8 + 0.2) * eptm.is_alive.a
    for edge in eptm.graph.edges():
        edge_alpha[edge] = vertex_alpha[edge.source()]
        edge_height[edge] = (depth[edge.source()]
                             + depth[edge.target()]) * 0.5
        
    vertex_alpha.a *= (1 - eptm.is_cell_vert.a)

    vorder = eptm.graph.new_vertex_property('float') 
    vorder.a = np.argsort(vertex_alpha.a)


    ### Junction vertices
    j_filt = eptm.is_cell_vert.copy()
    #j_filt.a *= (1 - eptm.is_alive.a)
    eptm.graph.set_vertex_filter(j_filt,
                                 inverted=True)
    if verbose: print eptm.graph.num_vertices()
    vertex_red.fa = 105/256.
    vertex_green.fa = 182/256.
    vertex_blue.fa = 40/256.
    vertex_size.fa = 1.
    eptm.graph.set_vertex_filter(None)

    ### Junction edges
    eptm.graph.set_edge_filter(eptm.is_junction_edge,
                               inverted=False)
    cmap = plt.cm.jet(edge_height.fa)
    edge_red.fa = cmap[:, 0] #105/256.
    edge_green.fa = cmap[:, 1] #201/256.
    edge_blue.fa = cmap[:, 2] #40/256.
    edge_width.fa = 2. * (eptm.junctions.line_tensions.fa /
                          eptm.junctions.line_tensions.fa.mean())**0.5
    eptm.graph.set_edge_filter(None)

    ### Cell vertices
    cell_filt = eptm.is_cell_vert.copy()
    cell_filt.a *= eptm.is_alive.a
    eptm.graph.set_vertex_filter(cell_filt,
                                 inverted=False)
    vertex_red.fa = 105 / 256.
    vertex_green.fa = 201 / 256.
    vertex_blue.fa = 237 / 256.
    vertex_size.fa = 0.
    eptm.graph.set_vertex_filter(None)

    ### Cell to junction edges
    eptm.graph.set_edge_filter(eptm.is_ctoj_edge,
                               inverted=False)
    edge_red.fa = 105 / 256.
    edge_green.fa = 201 / 256.
    edge_blue.fa = 237 / 256.
    edge_width.fa = 0.
    eptm.graph.set_edge_filter(None)


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
    
    sigma = eptm.proj_sigma()
    zs = [zeds, sigma]
    zs_pos = gt.group_vector_property(zs, value_type='float')
    eptm.update_dsigmas()
    edge_alpha.a = 1.
    edge_alpha.a *= (1 - eptm.at_boundary.a)
    edge_alpha.a *= eptm.is_junction_edge.a
    
    edge_rgba = [edge_red, edge_green, edge_blue, edge_alpha]
    edge_color = gt.group_vector_property(edge_rgba, value_type='float')
    
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
    eptm.graph.set_directed(True)


    
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

def plot_validation(eptm):
    vfilt = eptm.is_cell_vert.copy()
    vfilt.a *= eptm.is_alive.a
    eptm.graph.set_vertex_filter(vfilt)
    mean_area = eptm.cells.areas.fa.mean()
    eptm.graph.set_vertex_filter(None)

    eptm.graph.set_edge_filter(eptm.is_ctoj_edge)
    eptm.graph.set_vertex_filter(eptm.is_alive)
    degrees = eptm.graph.degree_property_map('out').fa
    valid_degrees = degrees[degrees > 0]
    normed_areas = eptm.cells.areas.fa[degrees > 0]/mean_area
    unq_degrees = np.unique(valid_degrees)
    avg_area = np.zeros_like(unq_degrees)
    std_area = np.zeros_like(unq_degrees)
    for n, k in enumerate(unq_degrees):
        avg_area[n] = normed_areas[valid_degrees==k].mean()
        std_area[n] = normed_areas[valid_degrees==k].std()

    exp_degrees = np.array([[3, 4, 5, 6, 7, 8],
                            [1, 7, 35, 38.5, 14.5, 2]])

    exp_areas = np.array([[4, 5, 6, 7, 8],
                          [0.55, 0.8, 1.08, 1.35, 1.5]])

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6,6))

    axes[0].plot(valid_degrees, normed_areas, 'o', alpha=0.1)
    axes[0].plot(exp_areas[0,:], exp_areas[1,:], 'gs-')
    axes[0].errorbar(unq_degrees, avg_area, yerr=std_area, fmt='k-o')
    ylblb = axes[0].set_ylabel('Normalized cell area')

    h = axes[1].hist(valid_degrees, bins=8, range=(2.5,10.5), normed=True, fc='w')
    axes[1].plot(exp_degrees[0, :], exp_degrees[1, :]/100., 'gs')
    xlblb = axes[1].set_xlabel('Number of sides')
    ylblb = axes[1].set_ylabel('Number of cells')

    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)