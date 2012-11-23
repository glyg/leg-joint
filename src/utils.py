#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import splrep, splev

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
