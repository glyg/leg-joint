# This is the Drosophilia Leg Joint Simulation - or in short just leg-joint

copyright Guillaume Gay - 2012 - 2014 - http://damcb.com

This work is part of a research project which has been publish in Nature:

**Apico-basal forces exerted by apoptotic cells drive epithelium folding**

Bruno Monier, Melanie Gettings, Guillaume Gay, Thomas Mangeat, Sonia Schott, Ana Guarner, Magali Suzanne
Nature (21 January 2015), [doi:10.1038/nature14152](http://dx.doi.org/10.1038/nature14152)


This work was done in collaboration with
[Magali Suzanne group](http://www-lbcmcp.ups-tlse.fr/Nouveau_site/modeles/EquipeSuzanne-Accueil.htm)
at LBCMCP - Université de Toulouse.

If you're just here for eye candy, you can watch a video of the simulation [here](http://vimeo.com/107188046)
There's a short summary of the biology [here](http://damcb.com/paper_out.html), and you can ask for the paper itself by e-mail.

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
`graph-tool` follow the link above and the instructions there. The
easiest way is to install from the precompiled packaged binaries, if
available for your OS. Due to a bug, version 2.2.35 won't work.


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
