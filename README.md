# This is the Drosophilia Leg Joint Simulation - or in short just leg-joint

copyright Guillaume Gay - 2012 - 2014 - http://damcb.com

This work is pending publication - once the associated scientific
article is out, it will be released as a GPL project.

This work was done in collaboration with
[Magali Suzanne group](http://www-lbcmcp.ups-tlse.fr/Nouveau_site/modeles/EquipeSuzanne-Accueil.htm)
at LBCMCP - Université de Toulouse.

<iframe src="//player.vimeo.com/video/107188046" width="500"
height="500" frameborder="0" webkitallowfullscreen mozallowfullscreen
allowfullscreen></iframe> <p><a href="http://vimeo.com/107188046">Fold
formation model</a> from <a
href="http://vimeo.com/user12210065">glyg</a> on <a
href="https://vimeo.com">Vimeo</a>.</p>


### Dependencies

* Python >= 3.3  (might work with minimal effort on 2.7, but it's untested)
* numpy >= 1.8
* scipy >= 0.12
* pandas >= 0.13
* matplotlib >= 1.3
* IPython >= 1.0
* [graph_tool](http://graph-tool.skewed.de/) >= 2.2.36 **warning** Due to
  a
  [bug](https://git.skewed.de/count0/graph-tool/commit/26f7d07b3359098fcc551b0a1159703bb4c10e18) in version 2.2.35, this version won't work

### Installing

The easiest route to install (nearly) all the dependencies is to use
[Anaconda](https://store.continuum.io/cshop/anaconda/). For
`graph-tool` follow the link above and the instructions
there. If 2.2.36 or higher is stable, the easiest way is to install
from the precompiled packaged binaries, if available for your OS. If
only 2.2.35 is released (as of the time of this writing), you need to
compile the version from `git`.


### Documentation

The best way to approach the simulations work-flow is to have a look (and eventually execute) at the notebooks.
You can browse static views of those here:
http://nbviewer.ipython.org/github/glyg/leg-joint/tree/master/notebooks/

### Acknowledgements


We are grateful to Corinne Benassayag, Emmanuel Farge, Yannick Gachet,
Thomas Lecuit, Pierre-François Lenne, François Payre, Ernesto
Sanchez-Herrero, Bénédicte Sanson and Sylvie Tournier for comments on
the manuscript, to Tiago de Paula Peixoto for providing the graph-tool
library, and to Marion Aguirrebengoa for helping us with
statistics. MS's lab is supported by an ANR grant, a RITC grant and a
grant from the University of Toulouse.
