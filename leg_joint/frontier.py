#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

def find_circumference(eptm, edge0):
    eptm.graph.set_edge_filter(eptm.is_junction_edge)
    front_verts = [edge0.source(), edge0.target()]
    ref_jv = edge0.source()
    front_edges = [edge0]
    cum_theta = eptm.dthetas[edge0]
    closed = False
    while cum_theta < 2 * np.pi:
        v0 = front_verts[-2]
        v1 = front_verts[-1]
        neighbs = [jv for jv in v1.all_neighbours() if jv != v0]
        sigmas = np.array([eptm.sigmas[jv] for jv in neighbs])
        zeds = np.array([eptm.zeds[jv] for jv in neighbs])
        tangent = np.abs((sigmas - eptm.sigmas[v0])
                         / (zeds - eptm.zeds[v0]))
        new_v = neighbs[np.argmax(tangent)]
        front_verts.append(new_v)
        new_e = eptm.any_edge(v1, new_v)
        front_edges.append(new_e)
        cum_theta += np.abs(eptm.dthetas[new_e]) 
    last_edge = eptm.any_edge(front_verts[0], front_verts[-1])
    if last_edge is not None:
        front_edges.append(last_edge)
        closed = True
    eptm.graph.set_edge_filter(None)
    return closed, front_edges

def create_frontier(eptm, zed, tension_increase=4.):

    sigma = 0.
    eptm.graph.set_vertex_filter(eptm.is_cell_vert, True)
    j_vert = eptm.closest_vert(0, zed)
    eptm.graph.set_vertex_filter(None)

    edge = [e for e in j_vert.all_edges()][0]
    closed, edge_list = find_circumference(eptm, edge)

    on_frontier_vert = eptm.graph.new_vertex_property("bool")
    on_frontier = eptm.graph.new_edge_property("bool")
    on_frontier.a[:] = 0
    for edge in edge_list:
        on_frontier[edge] = 1
        on_frontier_vert[edge.source()] = 1
        on_frontier_vert[edge.target()] = 1
        for cell in eptm.adjacent_cells(edge):
            on_frontier_vert[cell] = 1
    
    eptm.graph.set_edge_filter(on_frontier)
    eptm.junctions.line_tensions.fa *= tension_increase
    eptm.graph.set_edge_filter(None)
    eptm.graph.set_vertex_filter(None)
    for v in eptm.graph.vertices():
        if not (on_frontier_vert[v]
                and eptm.is_cell_vert[v]): continue  
        eptm.set_local_mask(None)
        eptm.set_local_mask(v)
        eptm.find_energy_min(tol=1e-6)

    # eptm.anisotropic_relax()
    eptm.update_apical_geom()
    eptm.set_local_mask(None)