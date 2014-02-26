Mathematical model of the epithelium
====================================

Architecture
------------

### The elementary cell

The architecture of the epithelium model consists in a graph containing

-   two types of vertices:

    -   The cell centers
    -   The appical junctions vertices

-   and two types of edges:

    -   The junction edges, structuring the apical surface of the
    	epithelium
    -   The cell to junction edges linking one cell to its
         neighbouring junction vertices.
    
    
  ![image](imgs/one_cell.*)
    
### Topological modifications

-   Cell division
-   Type 1 transition
-   Cell removal

### Locality

In many aspects of the model, only a local section of the epithelium is
manipulated, most notably during energy minimization.

Dynamical aspects
-----------------

### 3D geometry

The positions of the vertices are given in two 3D coordinate systems,
Cartesian $(x, y, z)$ and cylindrical $(\rho, \theta, z)$.

Initially, the vertices are distributed over a cylinder centered around
the $z$ axis. The volume of a cell is computed as the sum of the area of
each triangle formed by a junction edge and two cell-to-junction edges
times the height of the cell center. Of course this is an approximation:

$$V_\alpha = \sum_{i,j} A_{\alpha ij} 
   h_\alpha c_{\alpha i} c_{\alpha j} c_{ij}$$

with $c_{uv} = 1$ if vertices $u$ and $v$ are connected and $0$
otherwise, and
$A_{\alpha ij} = ||\mathbf{r}_{\alpha i} \times \mathbf{ r}_{\alpha j} || / 2$

### Energetical aspects

The underlying physical properties of the epithelium are an adaptation
of the model proposed by R. Faradhifar et al. [Faradhifar07]_
generalized to 3D geometries (constrained for now to deformation of the
cylinder)

#### Interactions at the apical junctions

Three interactions are considered at the epithelium:

-   The line tension between two junction vertices, with associated
    energy $E^t_{ij} = \Lambda \ell_{ij}$, where $\ell_{ij}$ is the
    length of the junction edge between vertices $i$ and $j$.

-   The contraction of a cell, with associated energy
    $E^c_\alpha = \Gamma L_\alpha^2$, where $L_\alpha$ is the perimeter
    of the cell $\alpha$.

-   The volume elasticity $E^v_\alpha = K_v (V_\alpha - V_0)$, where
    $V_\alpha$ is the volume of the cell $\alpha$.

The total energy is given by:

$$E = \sum_\alpha (E^v_\alpha + E^c_\alpha) 
+ \sum_{i \leftrightarrow j} E^t_{ij}$$

#### Energy of the epithelium on a regular lattice

According to R. Faradhifar et al. in the case of a regular hexagonal
latice, energy is given by:

$$E = N_{\alpha}\frac{K}{2} (A - A_0)^2 + N\frac{\Gamma}{2} L^2 +
6N\frac{\Lambda}{2}\ell$$

In our model, the area dependant term is replaced by a volume term. The
cell volume is given by its height $h$ times its area $A$:

$$E = N_{\alpha}\frac{K_v}{2} (h A - h_0A_0)^2 + N\frac{\Gamma}{2}
L^2 + 6N\frac{\Lambda}{2}\ell$$

Note that the 3D model is exactly equivalent to the 2D one with a
constant cell height $(h_0 = h = 1)$.

As they did, we define the adimentional contractility
$\bar\Gamma = \Gamma/K_vh_0A_0$ and line tension
$\bar\Lambda = \Lambda /K_v(h_0A_0)^{3/2}$. The normalized energy
$\bar E = E/NK_v(h_0A_0)^2$ then reads:

$$\bar E = \frac{1}{2} \left(\left(\frac{hA}{h_0A_0}^2 - 1\right)^2 + \frac{\bar\Gamma}{2} \frac{L^2}{h_0A_0} + 6\frac{\bar\Lambda\ell}{\sqrt{h_0A_0}}\right)$$

The perimeter $L$ of a cell is equal to $6\ell$ and the area $A$ equals
$(3\sqrt{3}/2)\ell^2$. We define a dilatation factor $\delta$ such that
$\delta^2 = hA/(h_0A_0)$. Thus:

$$hA/h_0A_0 = \delta^2 = \frac{3\sqrt{3}}{2} \frac{\ell^2}{h_0A_0}
\Rightarrow  \ell = \sqrt{\frac{2}{3\sqrt{3}}}\delta\sqrt{h_0A_0}$$$$

We define the constant $\mu = 6\left(2/3\sqrt{3}\right)^{1/2}$, then
$\ell = \frac{\mu}{6}\delta\sqrt{h_0A_0}$ and $L^2 =
\mu^2 \delta^2 h_0A_0$

$$\bar E = ((\delta^2 -1)^2 + \bar\Gamma\mu^2\delta^2 + \bar\Lambda\mu\delta) / 2 =  (\delta^4 + (\bar\Gamma\mu^2 - 2)\delta^2 + \bar\Lambda\mu\delta - 1)/2$$

#### Computation of the gradient

In order to compute the local minimum of the energy, we calculate its
gradient at vertex $i$

$$\mathbf{\nabla_i} E &= (\frac{\partial E}{\partial x},
                  \frac{\partial E}{\partial y},
                  \frac{\partial E}{\partial z}) \\
\mathbf{\nabla_i} E &= \sum_\alpha \left(K (V_\alpha - V_0) 
\mathbf{\nabla_i} V_\alpha  
+ \Gamma L_\alpha \mathbf{\nabla_i} L_\alpha  \right)c_{i \alpha}  
+ \sum_i \Lambda_{ij} \mathbf{\nabla_i} \ell_{ij}c_{ij}$$

We have

$$\mathbf{\nabla_i}L_\alpha &= \sum_{kn} \mathbf{\nabla_i} \ell_{kn} c_{\alpha k} c_{\alpha n}
= \sum_{j} \mathbf{\nabla_i}\ell_{ij} c_{ij} c_{\alpha i} c_{\alpha j}
= \sum_{j} \frac{\mathbf{r_{ij}}}{\ell_{ij}}c_{ij}c_{\alpha i} c_{\alpha j}\\
\mathbf{\nabla_i}\ell_{ij} &=  \frac{\mathbf{r_{ij}}}{\ell_{ij}}c_{ij}$$
