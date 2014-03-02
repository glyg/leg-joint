# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
import os
import graph_tool.all as gt
import matplotlib.pyplot as plt
import leg_joint as lj

import itertools
from IPython import parallel
rc = parallel.Client()
engines = rc[:]

with engines.sync_imports():
    import matplotlib.pyplot as plt
    import leg_joint as lj
    
print(rc.ids)

# <codecell>

### Base directories for saving, might as well have them here
### Base directories for saving, might as well have them here

grid_kwargs =  {'seq_kwargs': {'num_cells': 1,
                               'width_apopto':2.,
                               'p0': 1.,
                               'amp': 0.3,
                               'gamma': 1.0,
                               'seed': 1,
                               'num_steps': 10,
                               'ventral_bias': True,
                               'random': True},
                'apopto_kwargs': {'vol_reduction':0.7,
                                  'contractility': 1.2,
                                  'radial_tension': [0., 0.2]},
                'post_kwargs': {'max_ci':3.,
                                'rate_ci':[1.],
                                'span_ci':2}
                }



grid_indices = lj.get_grid_indices(grid_kwargs)
n_sims = 1
for param  in grid_indices.values():
    n_sims = len(param)
    break
print('Total number of simulations: %i' % n_sims)

def single_simulation(args):
    import leg_joint as lj
    import matplotlib.pyplot as plt
    index, params = args

    kwargs = lj.get_kwargs(index, params)
    eptm = lj.Epithelium(
        graphXMLfile='saved_graphs/xml/before_apoptosis.xml',
        identifier='expl_%04i' % index)
    eptm.dump_json(kwargs)
    lj.gradual_apoptosis(eptm, kwargs['seq_kwargs'],
                         kwargs['apopto_kwargs'], 
                         kwargs['post_kwargs'])
    
    ax = lj.plot_avg_rho(eptm, bin_width=20)
    fig = ax.get_figure()
    plt.savefig(os.path.join(eptm.paths['svg'],
                             'avg_rho_%s.svg'
                             % eptm.identifier))
    print('Done %s' % eptm.identifier)
    return index

arguments = zip(range(n_sims),
                itertools.repeat(grid_kwargs))
results = engines.map(single_simulation, arguments)

for index in results:
    print('Done %s' % index)