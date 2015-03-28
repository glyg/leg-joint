# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import logging
log = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s -'
                              '%(name)s -%(funcName)s -- '
                              '%(levelname)s -'
                              '%(message)s')
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
log.addHandler(ch)
log.setLevel(logging.INFO)
log.propagate = False


from .data import default_params, before_apoptosis_xml, get_image
## Import matplotlib here to avoid instabilities
## with graph-tool
# import matplotlib.pylab as plt

from .epithelium import Epithelium
from .draw import graph_representation

# from .dynamics.optimizers import find_energy_min, isotropic_optimum
# from .dynamics.optimizers import approx_grad, check_local_grad, running_local_optimum
# from .topology import cell_division, type1_transition
# from .topology import type3_transition
# from .topology import remove_cell, resolve_small_edges
# from .frontier import find_circumference, create_frontier
# from .apoptosis import gradual_apoptosis, get_apoptotic_cells
# from .apoptosis import solve_all_rosettes, apoptosis_step, post_apoptosis
# from .graph_representation import plot_ortho_proj, plot_ortho_gradients
# from .graph_representation import plot_edges_generic, plot_cells_generic
# from .graph_representation import epithelium_draw as draw, plot_eptm_generic
# from .graph_representation import plot_2pannels, plot_2pannels_gradients
# from .graph_representation import plot_avg_rho
# from .graph_representation import plot_repartition

# from .explore_params import get_list_kwargs, get_grid_indices, get_grid_kwargs

# from .filters import local_slice, focus_on_cell

# from .explore_params import explore_delta_h_1D

# from .data import default_params, before_apoptosis_xml, get_image



# log.info('successfully imported leg_joint')
