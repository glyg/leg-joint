# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import logging
log = logging.getLogger(__name__)
# handler = logging.StreamHandler()

# logger.addHandler(handler)
log.setLevel(logging.INFO)
# logger.propagate = False


## Import matplotlib here to avoid instabilities
## with graph-tool
import matplotlib.pylab as plt

from .epithelium import Epithelium
from .optimizers import find_energy_min, isotropic_optimum, running_local_optimum
from .optimizers import approx_grad, check_local_grad
from .topology import cell_division, type1_transition
from .topology import type3_transition
from .topology import remove_cell, resolve_small_edges
from .frontier import find_circumference, create_frontier
from .apoptosis import gradual_apoptosis, get_apoptotic_cells, apoptosis_step, post_apoptosis
from .apoptosis import solve_all_rosettes
from .graph_representation import plot_ortho_proj, plot_ortho_gradients
from .graph_representation import plot_edges_generic, plot_cells_generic, plot_eptm_generic
from .graph_representation import epithelium_draw as draw
from .graph_representation import plot_2pannels, plot_2pannels_gradients
from .explore_params import get_kwargs, get_grid_indices

from .filters import local_slice, focus_on_cell

from .explore_params import explore_delta_h_1D


