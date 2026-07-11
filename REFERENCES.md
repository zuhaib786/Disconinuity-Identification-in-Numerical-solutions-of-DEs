# References

Key literature for this project (previously vendored as PDFs; removed from
the repository for copyright reasons).

## Methods implemented here

- J. S. Hesthaven, T. Warburton, *Nodal Discontinuous Galerkin Methods:
  Algorithms, Analysis, and Applications*, Springer, 2008.
  (DG solver, slope limiter; `tci/solvers/`, `tci/limiters.py`)
- R. Archibald, A. Gelb, J. Yoon, "Polynomial fitting for edge detection in
  irregularly sampled signals and images", *SIAM J. Numer. Anal.* 43 (2005).
  (polynomial annihilation; `tci/indicators/pa.py`)
- L. Krivodonova, J. Xin, J.-F. Remacle, N. Chevaugeon, J. E. Flaherty,
  "Shock detection and limiting with discontinuous Galerkin methods for
  hyperbolic conservation laws", *Appl. Numer. Math.* 48 (2004).
  (KXRCF indicator; `tci/indicators/classical.py`)
- P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio,
  "Graph Attention Networks", ICLR 2018. (`tci/models.py`)

## Closest prior work (position against these in a paper)

- D. Ray, J. S. Hesthaven, "An artificial neural network as a troubled-cell
  indicator", *J. Comput. Phys.* 367 (2018).
- D. Ray, J. S. Hesthaven, "Detecting troubled-cells on two-dimensional
  unstructured grids using a neural network", *J. Comput. Phys.* 397 (2019).
- A. D. Beck, J. Zeifang, A. Schwarz, D. G. Flad, "A neural network based
  shock detection and localization approach for discontinuous Galerkin
  methods", *J. Comput. Phys.* 423 (2020). arXiv:2001.08201.
- J. Zhu, J. Qiu et al., review of troubled-cell indicators for DG methods,
  arXiv:2309.11973 (2023).

## Background

- B. Cockburn, C.-W. Shu, "Runge-Kutta discontinuous Galerkin methods for
  convection-dominated problems", *J. Sci. Comput.* 16 (2001).
- G. Fu, C.-W. Shu, "A new troubled-cell indicator for discontinuous
  Galerkin methods for hyperbolic conservation laws", *J. Comput. Phys.*
  347 (2017).
