# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

## Import matplotlib here to avoid instabilities
## with graph-tool
import matplotlib.pylab as plt

from .epithelium import Epithelium
from .optimizers import find_energy_min, isotropic_optimum, running_local_optimum
from .optimizers import approx_grad, check_local_grad
from .topology import cell_division, type1_transition
from .topology import type3_transition, apoptosis
from .topology import remove_cell, resolve_small_edges, enhance_tension
from .frontier import find_circumference, create_frontier
from .graph_representation import plot_ortho_proj, plot_ortho_gradients
from .graph_representation import draw_polygons, plot_cells_generic
from .graph_representation import epithelium_draw as draw
from .graph_representation import plot_2pannels, plot_2pannels_gradients


from .utils import local_slice

from .explore_params import explore_delta_h_1D






