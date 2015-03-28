# -*- coding: utf-8 -*-

import os
import numpy as np

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon


import graph_tool.all as gt

from ..topology.filters import active, local_slice
from ..dynamics.optimizers import approx_grad

FLOAT = np.dtype('float32')
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
GRAPH_SAVE_DIR = os.path.join(ROOT_DIR, 'saved_graphs')


def plot_repartition(eptm, apopto_cells, seq_kwargs):

    eptm.set_local_mask(None)
    local_slice(eptm, theta_amp=2*np.pi,
                   zed_amp=seq_kwargs['width_apopto'])

    eptm.update_polar()
    eptm.update_pmaps()

    d_theta = 0.
    z_angle = np.pi / 6
    pseudo_x = eptm.x.copy()
    pseudo_y = eptm.x.copy()
    pseudo_x.a = eptm.z.a * np.cos(z_angle) - eptm.rho.a * np.sin(
        eptm.theta.a + d_theta) * np.sin(z_angle)
    pseudo_y.a = eptm.rho.a * np.cos(eptm.theta.a + d_theta)
    is_apopto = eptm.is_cell_vert.copy()
    is_apopto.a[:] = 0
    color_dead = eptm.z.copy()
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

    eptm.update_polar()
    eptm.update_pmaps()
    z = eptm.z.a
    rho = eptm.rho.a
    rho = rho[np.argsort(z)]
    z = np.sort(z)

    rho_cliped = rho[: -(rho.size % bin_width)]
    rho_cliped = rho_cliped.reshape((rho_cliped.size // bin_width,
                                       bin_width))
    rho_avg = rho_cliped.mean(axis=1)
    rho_max = rho_cliped.max(axis=1)
    rho_min = rho_cliped.min(axis=1)

    z_cliped = z[: -(z.size % bin_width)]
    z_cliped = z_cliped.reshape((z_cliped.size // bin_width,
                                       bin_width))
    z_avg = z_cliped.mean(axis=1)

    return z_avg, rho_avg, rho_max, rho_min

def plot_avg_rho(eptm, bin_width, ax=None, retall=False, ls='r-'):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,2))
    else:
        fig = ax.get_figure()
    z_avg, rho_avg, rho_max, rho_min = average_rho(eptm, bin_width)

    ax.fill_between(z_avg,
                    rho_max,
                    rho_min,
                    facecolor='0.5', edgecolor='0.9')
    ax.plot(z_avg, rho_avg, ls, lw=2, alpha=0.7)
    ax.set_aspect('equal')
    ax.set_xlabel('Proximal - distal (µm)')
    ax.set_ylabel('Radius (µm)')

    max_zed = ax.get_ylim()[1]
    ax.set_ylim(0, max_zed)
    if not retall:
        return ax
    return ax, (z_avg, rho_avg, rho_max, rho_min)

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

    grad_z = grad_xyz[2::3] * scale

    pos0, bounds = precondition(eptm)
    rho, theta  = to_rhotheta(pos0[::3], pos0[1::3])
    sigmas = theta * eptm.rho.a.mean()
    grad_sigmas = (- np.sin(theta) * grad_xyz[::3] +
                   np.cos(theta) * grad_xyz[1::3]) * scale

    # We plot the forces, which are easier to understand
    v_ixs = np.array([pos0[0::3], - grad_ixs]).T
    v_wys = np.array([pos0[1::3], - grad_wys]).T
    v_z = np.array([pos0[2::3], - grad_z]).T
    v_sigmas = np.array([sigmas, - grad_sigmas]).T

    if axes is None:
        ax_zs, ax_xy =  plot_2pannels(eptm, axes=None,
                                      **kwargs)
        axes = ax_zs, ax_xy
    else:
        ax_zs, ax_xy = axes
    x_lim = ax_zs.get_xlim()

    for s, z, x, y in zip(v_sigmas, v_z, v_ixs, v_wys):
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
    rho_avg = eptm.rho.a.mean()
    rho, theta  = to_rhotheta(pos0[::3], pos0[1::3])
    sigmas = theta * rho_avg
    z = pos0[2::3]
    grad_rho = (np.cos(theta) * grad_xyz[::3]
                 + np.sin(theta) * grad_xyz[1::3]) * scale
    grad_sigmas = (- np.sin(theta) * grad_xyz[::3] +
                   np.cos(theta) * grad_xyz[1::3]) * scale
    grad_z = grad_xyz[2::3] * scale
    # We plot the forces, which are easier to understand
    v_sigmas = np.array([sigmas, - grad_sigmas]).T
    v_z = np.array([z, - grad_z]).T
    v_radial = np.array([rho, - grad_rho]).T
    if axes is None:
        ax_zs, ax_zr, ax_rs =  plot_ortho_proj(eptm, ax=None,
                                               vfilt=eptm.is_local_vert,
                                               efilt=eptm.is_local_edge,
                                               **kwargs)
        axes = ax_zs, ax_zr, ax_rs
    else:
        ax_zs, ax_zr, ax_rs = axes
    for s, z, r in zip(v_sigmas, v_z, v_radial):
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
    sigma = eptm.z.copy()
    sigma.a = eptm.proj_sigma().values
    xy_coords = ((eptm.z, sigma), (eptm.x, eptm.y))
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
    sigma = eptm.z.copy()
    sigma.a = eptm.proj_sigma().values

    plot_eptm_generic(eptm, eptm.z,
                      sigma,
                      ax=ax,
                      cell_kwargs=cell_kwargs,
                      edge_kwargs=edge_kwargs)

    divider = make_axes_locatable(ax)

    ax_zr = divider.append_axes("top", 2., pad=0.1, sharex=ax)
    plot_eptm_generic(eptm, eptm.z,
                      eptm.rho,
                      ax=ax_zr,
                      cell_kwargs=cell_kwargs,
                      edge_kwargs=edge_kwargs)

    ax_rs = divider.append_axes("right", 2., pad=0.1, sharey=ax)
    plot_eptm_generic(eptm,
                      eptm.rho,
                      sigma,
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
                       vfilt=None, c_text=False, j_text=False,
                       cell_colors=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    eptm.graph.set_vertex_filter(vfilt)
    if c_text:
        for cell in eptm.cells :
            ax.text(xcoord[cell], ycoord[cell], str(cell))
    if j_text:
        for jv in eptm.graph.vertices() :
            if eptm.is_cell_vert[jv]: continue
            ax.text(xcoord[jv], ycoord[jv], str(jv))

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

def epithelium_draw(eptm, z_angle=0.15, d_theta=4*np.pi/5,
                    output3d="tissue_3d.pdf",
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

    eptm.update_polar()
    eptm.update_pmaps()
    rho, theta, z = eptm.rho, eptm.theta, eptm.z

    pseudo_x = eptm.graph.new_vertex_property('float')
    pseudo_y = eptm.graph.new_vertex_property('float')
    pseudo_x.a = z.a * np.cos(z_angle) - rho.a * np.cos(
        theta.a + d_theta) * np.sin(z_angle)
    pseudo_y.a = rho.a * np.sin(theta.a + d_theta)

    depth = vertex_alpha.copy()
    depth.a = rho.a * (1 - np.cos(theta.a + d_theta))
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
    eptm.graph.set_vertex_filter(j_filt,
                                 inverted=True)
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
    eptm.graph.set_edge_filter(eptm.is_junction_edge,
                               inverted=True)
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
    eptm.log.info('saved tissue to %s' % output3d)
    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)

def plot_validation(eptm):
    vfilt = eptm.is_cell_vert.copy()
    vfilt.a *= eptm.is_alive.a
    eptm.graph.set_vertex_filter(vfilt)
    mean_area = eptm.cells.areas.fa.mean()
    eptm.graph.set_vertex_filter(None)

    eptm.graph.set_edge_filter(eptm.is_junction_edge, inverted=True)
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
