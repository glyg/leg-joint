# -*- coding: utf-8 -*-

import os
import numpy as np

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon


import graph_tool.all as gt

from .filters import active, local_slice
from .optimizers import precondition, approx_grad
from .utils import to_rhotheta
FLOAT = np.dtype('float32')

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
GRAPH_SAVE_DIR = os.path.join(ROOT_DIR, 'saved_graphs')


def plot_repartition(eptm, apopto_cells, seq_kwargs):

    eptm.set_local_mask(None)
    local_slice(eptm, theta_amp=2*np.pi,
                   zed_amp=seq_kwargs['width_apopto'])

    eptm.update_rhotheta()
    d_theta = 0.
    z_angle = np.pi / 6
    pseudo_x = eptm.ixs.copy()
    pseudo_y = eptm.ixs.copy()
    pseudo_x.a = eptm.zeds.a * np.cos(z_angle) - eptm.rhos.a * np.sin(
        eptm.thetas.a + d_theta) * np.sin(z_angle)
    pseudo_y.a = eptm.rhos.a * np.cos(eptm.thetas.a + d_theta)
    is_apopto = eptm.is_cell_vert.copy()
    is_apopto.a[:] = 0
    color_dead = eptm.zeds.copy()
    color_dead.a[:] = 0.
    for cell in apopto_cells:
        color_dead[cell] = 1.
        is_apopto[cell] = 1
        for jv in cell.out_neighbours():
            is_apopto[jv] = 1
    ax = plot_eptm_generic(eptm,
                           pseudo_x, pseudo_y, local=True,
                           cell_kwargs={'cell_colors':color_dead, 'alpha':0.4},
                           edge_kwargs={'c':'g', 'lw':1, 'alpha':0.4})
    plt.savefig(os.path.join(eptm.paths['svg'],
                             'apopto_repartition_%s.svg'
                             % eptm.identifier))


def png_snapshot(func, *args, **kwargs):
    def new_func(eptm, *args, **kwargs):
        out = func(eptm, *args, **kwargs)
        png_dir = eptm.paths['png']
        outfname2d = os.path.join(png_dir, 'eptm2d_%04i.png'
                                  % eptm.stamp)
        outfname3d = os.path.join(png_dir, 'eptm3d_%04i.png'
                                  % eptm.stamp)
        try:
            epithelium_draw(eptm, d_theta= -np.pi/8,
                            output2d=outfname2d, output3d=outfname3d)
        except:
            pass
        return out
    return new_func

def local_svg_snapshot(func, *args, **kwargs):
    def new_func(eptm, *args, **kwargs):
        out = func(eptm, *args, **kwargs)
        svg_dir = eptm.paths['svg']
        outfname = os.path.join(svg_dir, 'local_%05i.svg'
                                % eptm.stamp)
        ax_zs, ax_xy = plot_2pannels(eptm, c_text=False)
        fig = ax_zs.get_figure()
        fig.savefig(outfname)
        plt.close(fig)
        return out
    return new_func

def average_rho(eptm, bin_width=10):

    eptm.update_rhotheta()
    zeds = eptm.zeds.a
    rhos = eptm.rhos.a
    rhos = rhos[np.argsort(zeds)]
    zeds = np.sort(zeds)

    rhos_cliped = rhos[: -(rhos.size % bin_width)]
    rhos_cliped = rhos_cliped.reshape((rhos_cliped.size // bin_width,
                                       bin_width))
    rhos_avg = rhos_cliped.mean(axis=1)
    rhos_max = rhos_cliped.max(axis=1)
    rhos_min = rhos_cliped.min(axis=1)

    zeds_cliped = zeds[: -(zeds.size % bin_width)]
    zeds_cliped = zeds_cliped.reshape((zeds_cliped.size // bin_width,
                                       bin_width))
    zeds_avg = zeds_cliped.mean(axis=1)

    return zeds_avg, rhos_avg, rhos_max, rhos_min

def plot_avg_rho(eptm, bin_width, ax=None, retall=False, ls='r-'):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,2))
    else:
        fig = ax.get_figure()
    zeds_avg, rhos_avg, rhos_max, rhos_min = average_rho(eptm, bin_width)

    ax.fill_between(zeds_avg,
                    rhos_max,
                    rhos_min,
                    facecolor='0.5', edgecolor='0.9')
    ax.plot(zeds_avg, rhos_avg, ls, lw=2, alpha=0.7)
    ax.set_aspect('equal')
    ax.set_xlabel('Proximal - distal (µm)')
    ax.set_ylabel('Radius (µm)')

    max_zed = ax.get_ylim()[1]
    ax.set_ylim(0, max_zed)
    if not retall:
        return ax
    return ax, (zeds_avg, rhos_avg, rhos_max, rhos_min)

def draw_polygons(eptm, coord1, coord2, colors=None,
                  vfilt=None, ax=None, alphas=None,
                  cmap='jet', **kwargs):

    eptm.graph.set_vertex_filter(vfilt)
    cmap = plt.get_cmap(cmap)

    eptm.update_dsigmas()

    if colors is not None:
        color_cmap = cmap(colors.fa)
        poly_red = colors.copy()
        poly_red.fa = color_cmap[:, 0]
        poly_green = colors.copy()
        poly_green.fa = color_cmap[:, 1]
        poly_blue = colors.copy()
        poly_blue.fa = color_cmap[:, 2]

    if ax is None:
        fig, ax = plt.subplots()

    for cell in eptm.cells:
        if not eptm.is_alive[cell] or eptm.cells.is_boundary(cell):
            continue
        poly = eptm.cells.polygon(cell, coord1, coord2)[0]
        if colors is not None:
            kwargs['color'] = [poly_red[cell], poly_green[cell], poly_blue[cell]]
        if alphas is not None:
            kwargs['alpha'] = alphas[cell]

        patch = Polygon(poly,
                        fill=True, closed=True, **kwargs)
        ax.add_patch(patch)
    #ax.autoscale_view()
    ax.set_aspect('equal')

    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)
    return ax

def plot_2pannels_gradients(eptm, axes=None,
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

    grad_ixs = grad_xyz[0::3] * scale

    grad_wys = grad_xyz[1::3] * scale

    grad_zeds = grad_xyz[2::3] * scale

    pos0, bounds = precondition(eptm)
    rhos, thetas  = to_rhotheta(pos0[::3], pos0[1::3])
    sigmas = thetas * eptm.rhos.a.mean()
    grad_sigmas = (- np.sin(thetas) * grad_xyz[::3] +
                   np.cos(thetas) * grad_xyz[1::3]) * scale

    # We plot the forces, which are easier to understand
    v_ixs = np.array([pos0[0::3], - grad_ixs]).T
    v_wys = np.array([pos0[1::3], - grad_wys]).T
    v_zeds = np.array([pos0[2::3], - grad_zeds]).T
    v_sigmas = np.array([sigmas, - grad_sigmas]).T

    if axes is None:
        ax_zs, ax_xy =  plot_2pannels(eptm, axes=None,
                                      **kwargs)
        axes = ax_zs, ax_xy
    else:
        ax_zs, ax_xy = axes
    x_lim = ax_zs.get_xlim()

    for s, z, x, y in zip(v_sigmas, v_zeds, v_ixs, v_wys):
        ax_xy.arrow(x[0], y[0], x[1], y[1], width=0.01,
                    ec=ec, fc=fc, alpha=0.5)
        ax_zs.arrow(z[0], s[0], z[1], s[1], width=0.01,
                    ec=ec, fc=fc, alpha=0.5)

    ax_zs.set_xlim(x_lim)
    return axes



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


@active
def plot_active(eptm, xcoord, ycoord, ax=None):
    '''
    '''
    xs = xcoord.fa
    ys = ycoord.fa
    if ax is None:
        ax =  plot_edges_generic(eptm, xcoord, ycoord, ax=None,
                                 vfilt=eptm.is_local_vert,
                                 efilt=eptm.is_local_edge)

    ax.plot(xs, ys, 'ro', alpha=0.5, ms=8)
    plt.draw()
    return ax

def plot_2pannels(eptm, axes=None,
                  edge_kwargs={}, cell_kwargs={}):
    if edge_kwargs.get('c') is None:
        edge_kwargs['c'] = 'g'
    if axes is None:
        fig, axes = plt.subplots(1, 2)
    xy_coords = ((eptm.zeds, eptm.proj_sigma()), (eptm.ixs, eptm.wys))
    axes = plot_pannels(eptm, xy_coords,
                        axes,
                        local=True,
                        edge_kwargs=edge_kwargs,
                        cell_kwargs=cell_kwargs)
    return axes

def plot_pannels(eptm, xy_coords,
                 axes,
                 local=True,
                 edge_kwargs={}, cell_kwargs={}):

    if not (len(axes) == len(xy_coords)):
        raise ValueError('the `xy_coords` and `axes` arguments'
                         ' should have the same length')

    if local:
        cell_kwargs['vfilt'] = eptm.is_local_vert
        edge_kwargs['efilt'] = eptm.is_local_edge
    if axes is None:
        fig, axes = plt.subplots(1, 2)
    for pannel_coords, pannel_ax in zip(xy_coords, axes):
        xcoord, ycoord = pannel_coords
        plot_eptm_generic(eptm, xcoord, ycoord,
                          ax=pannel_ax, local=local,
                          edge_kwargs=edge_kwargs,
                          cell_kwargs=cell_kwargs)
    return axes


def plot_ortho_proj(eptm, ax=None, local=True,
                    edge_kwargs={}, cell_kwargs={}):
    if local:
        vfilt = eptm.is_local_vert
        efilt = eptm.is_local_edge
    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = ax.get_figure()

    plot_eptm_generic(eptm, eptm.zeds,
                      eptm.proj_sigma(),
                      ax=ax,
                      cell_kwargs=cell_kwargs,
                      edge_kwargs=edge_kwargs)

    divider = make_axes_locatable(ax)

    ax_zr = divider.append_axes("top", 2., pad=0.1, sharex=ax)
    plot_eptm_generic(eptm, eptm.zeds,
                      eptm.rhos,
                      ax=ax_zr,
                      cell_kwargs=cell_kwargs,
                      edge_kwargs=edge_kwargs)

    ax_rs = divider.append_axes("right", 2., pad=0.1, sharey=ax)
    plot_eptm_generic(eptm,
                      eptm.rhos,
                      eptm.proj_sigma(),
                      ax=ax_rs,
                      cell_kwargs=cell_kwargs,
                      edge_kwargs=edge_kwargs)


    plt.setp(ax_zr.get_xticklabels() + ax_rs.get_yticklabels(),
             visible=False)

    ax_zr.set_xlabel('')
    ax_rs.set_ylabel('')

    axes = ax, ax_zr, ax_rs
    return axes



def plot_eptm_generic(eptm, xcoord, ycoord,
                      ax=None, local=True,
                      edge_kwargs={'c':'g'}, cell_kwargs={}):

    if ax is None:
        fig, ax = plt.subplots()

    if local:
        cell_kwargs['vfilt'] = eptm.is_local_vert
        edge_kwargs['efilt'] = eptm.is_local_edge

    plot_cells_generic(eptm, xcoord,
                       ycoord,
                       ax=ax,
                       **cell_kwargs)
    plot_edges_generic(eptm, xcoord,
                       ycoord,
                       ax=ax,
                       **edge_kwargs)
    return ax

def plot_cells_generic(eptm, xcoord, ycoord, ax=None,
                       vfilt=None, c_text=False,
                       cell_colors=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    eptm.graph.set_vertex_filter(vfilt)
    if c_text:
        for cell in eptm.cells :
            ax.text(xcoord[cell], ycoord[cell], str(cell))
    if cell_colors is not None:
        draw_polygons(eptm, xcoord,
                      ycoord,
                      cell_colors, ax=ax, vfilt=vfilt, **kwargs)
    eptm.graph.set_vertex_filter(None)
    return ax

def plot_edges_generic(eptm, xcoord, ycoord, efilt=None,
                       ax=None, j_text=False,
                       edge_color=None, edge_alpha=None,
                       edge_width=None,
                       cmap='jet', **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    cmap = plt.get_cmap(cmap)
    eptm.graph.set_edge_filter(efilt)
    eptm.graph.set_vertex_filter(None)
    if edge_color is not None:
        depth_cmap = cmap(edge_color.fa)
        edge_red = edge_color.copy()
        edge_red.fa = depth_cmap[:, 0]
        edge_green = edge_color.copy()
        edge_green.fa = depth_cmap[:, 1]
        edge_blue = edge_color.copy()
        edge_blue.fa = depth_cmap[:, 2]

    for edge in eptm.graph.edges():
        if (eptm.is_junction_edge[edge]):
            ixs = (xcoord[edge.source()],
                   xcoord[edge.target()])
            wys = (ycoord[edge.source()],
                   ycoord[edge.target()])
            if edge_alpha is not None:
                kwargs['alpha'] = edge_alpha[edge]

            if edge_color is not None:
                c = [edge_red[edge], edge_green[edge], edge_blue[edge]]
                kwargs['c'] = c
            if edge_width is not None:
                kwargs['lw'] = edge_width[edge]
            ax.plot(ixs, wys, '-', **kwargs)
    ax.set_aspect('equal')
    eptm.graph.set_edge_filter(None)
    return ax


def sfdp_draw(graph, output="lattice_3d.pdf"):
    '''
    Deprecated
    '''
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
    '''
    Deprecated
    '''
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
    pseudo3d_color = gt.group_vector_property(rgba,
                                              value_type='float')

    xy = [pseudo_x, pseudo_y]
    pseudo3d_pos = gt.group_vector_property(xy, value_type='float')
    pmap = gt.graph_draw(graph, pseudo3d_pos,
                         vertex_fill_color=pseudo3d_color,
                         vertex_color=pseudo3d_color,
                         edge_pen_width=2.,
                         output=output, **kwargs)
    del pmap
    return pseudo3d_pos

def epithelium_draw(eptm, z_angle=0.15, d_theta=4*np.pi/5,
                    output3d="tissue_3d.pdf",
                    output2d='tissue_sz.pdf', verbose=False,
                    vfilt=None,
                    efilt=None,
                    **kwargs):

    kwargs['inline'] = False
    eptm.graph.set_directed(False)

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
    if verbose: print(eptm.graph.num_vertices())
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
    #edge_width.fa[:] = 1.
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
    eptm.graph.set_vertex_filter(vfilt)
    eptm.graph.set_edge_filter(efilt)
    pmap = gt.graph_draw(eptm.graph, pseudo3d_pos,
                         vertex_fill_color=vertex_color,
                         vertex_color=vertex_color,
                         edge_pen_width=edge_width,
                         edge_color=edge_color,
                         vertex_size=vertex_size,
                         vorder=vorder, eorder=eorder,
                         output=output3d,  **kwargs)
    if verbose: print('saved tissue to %s' % output3d)
    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)

    #### 2D view
    eptm.rotate(-np.pi / 2)
    ### Junction edges
    eptm.graph.set_edge_filter(eptm.is_junction_edge,
                               inverted=False)
    depth.a = rhos.a
    depth.a = (depth.a - depth.a.min()) / (depth.a.max() - depth.a.min())
    vertex_alpha.a = (depth.a * 0.8 + 0.2) * eptm.is_alive.a
    for edge in eptm.graph.edges():
        edge_alpha[edge] = vertex_alpha[edge.source()]
        edge_height[edge] = (depth[edge.source()]
                             + depth[edge.target()]) * 0.5


    cmap = plt.cm.jet(edge_height.fa)
    edge_red.fa = cmap[:, 0] #105/256.
    edge_green.fa = cmap[:, 1] #201/256.
    edge_blue.fa = cmap[:, 2] #40/256.
    #edge_width.fa[:] = 1.
    edge_width.fa = 2. * (eptm.junctions.line_tensions.fa /
                          eptm.junctions.line_tensions.fa.mean())**0.5
    eptm.graph.set_edge_filter(None)


    sigma = eptm.proj_sigma()
    zs = [zeds, sigma]
    zs_pos = gt.group_vector_property(zs, value_type='float')
    eptm.update_dsigmas()
    edge_alpha.a = 1.
    edge_alpha.a *= (1 - eptm.at_boundary.a)
    edge_alpha.a *= eptm.is_junction_edge.a

    edge_rgba = [edge_red, edge_green, edge_blue, edge_alpha]
    edge_color = gt.group_vector_property(edge_rgba, value_type='float')
    eptm.graph.set_vertex_filter(vfilt)
    eptm.graph.set_edge_filter(efilt)
    pmap2 = gt.graph_draw(eptm.graph, zs_pos,
                          vertex_fill_color=vertex_color,
                          vertex_color=vertex_color,
                          edge_pen_width=edge_width,
                          edge_color=edge_color,
                          vertex_size=vertex_size,
                          vorder=vorder, eorder=eorder,
                          output=output2d,  **kwargs)
    if verbose: print('saved tissue to %s' % output2d)
    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)
    del pmap, pmap2
    eptm.graph.set_directed(True)
    eptm.rotate(np.pi / 2)

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

    h = axes[1].hist(valid_degrees, bins=8, range=(2.5,10.5),
                     normed=True, fc='w')
    axes[1].plot(exp_degrees[0, :], exp_degrees[1, :]/100., 'gs')
    xlblb = axes[1].set_xlabel('Number of sides')
    ylblb = axes[1].set_ylabel('Number of cells')

    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)