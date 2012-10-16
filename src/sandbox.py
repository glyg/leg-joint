#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import splrep, splev


import filters

from graph_representation import epithelium_draw

def perform_dilation(eptm,
                     sigma_dilation_along_z,
                     zed_dilation_along_s):

    eptm.sigmas.a *= sigma_dilation_along_z
    eptm.zeds.a *= zed_dilation_along_s
    eptm.rhos.a *= sigma_dilation_along_z
    eptm.update_apical_geom()
    

    
def create_properties(eptm):
    # FIXME Should be implemented at init
    edge_zeds = eptm.graph.new_edge_property('float')
    edge_rhos = eptm.graph.new_edge_property('float')
    edge_sigmas = eptm.graph.new_edge_property('float')
    raw_dsigmas = eptm.graph.new_edge_property('float')
    for edge in eptm.graph.edges():
        src, trgt = edge.source(), edge.target()
        edge_zeds[edge] = (eptm.zeds[src] + 
                           eptm.zeds[trgt]) / 2.
        edge_sigmas[edge] = (eptm.sigmas[src] + 
                             eptm.sigmas[trgt]) / 2.
        edge_rhos[edge] = (eptm.rhos[src] + 
                           eptm.rhos[trgt]) / 2.
        raw_dsigmas[edge] = eptm.sigmas[src]- eptm.sigmas[trgt]
    return edge_zeds, edge_sigmas, edge_rhos, raw_dsigmas

    
def compute_aniso_dilation(eptm, bins, length_eq):
    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)
    edge_zeds, edge_sigmas, edge_rhos, raw_dsigmas = create_properties(eptm)

    eptm.set_edge_state([(eptm.is_junction_edge, False)])
    eptm.set_vertex_state([(eptm.is_alive, False)])
    period = 2 * np.pi * edge_rhos.fa
    # Delta sigma
    abs_ds = np.abs(raw_dsigmas.fa)
    first_period_dsigma = abs_ds[abs_ds < period / 2.]
    first_period_ez = edge_zeds.fa[abs_ds < period / 2.]
    tck_ds_vs_ez, H_dsez = compute_distribution(first_period_ez,
                                                first_period_dsigma, bins)
    #Delta zed
    dzeds = np.abs(eptm.dzeds.fa)
    tck_dz_vs_es, Hdzes = compute_distribution(edge_sigmas.fa,
                                               dzeds, bins)
    eptm.set_edge_state()
    eptm.set_vertex_state()

    smth_dsigma_mean = splev(eptm.zeds.a, tck_ds_vs_ez)
    smth_dzed_mean = splev(eptm.sigmas.a, tck_dz_vs_es)
    
    sigma_dilation_along_z = smth_dsigma_mean / length_eq
    zed_dilation_along_s = smth_dzed_mean / length_eq

    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)
    return sigma_dilation_along_z, zed_dilation_along_s
    

def compute_distribution(prop_u, prop_v, bins, smth=0):
    
    hist_uv, bins_u, bins_v = np.histogram2d(prop_u,
                                             prop_v, bins=bins)
    bin_wu = bins_u[1] - bins_u[0]
    bin_wv = bins_v[1] - bins_v[0]

    regular_u = bins_u[:-1] + bin_wu/2.
    regular_v = bins_v[:-1] + bin_wv/2.

    Huv = hist_uv, bins_u, bins_v
    
    print '''bin width = %.3f''' % bin_wu
    norm = hist_uv.sum(axis=1)
    mean_vfu = (regular_v * hist_uv).sum(axis=1) / hist_uv.sum(axis=1)
    tck_vfu = splrep(regular_u, mean_vfu, k=3, s=smth)
    return tck_vfu, Huv

    
def ref_distro(x, bin_w, s):
    return bin_w * (x / (s**2)) * exp(- (x**2 / (2 * s**2)))