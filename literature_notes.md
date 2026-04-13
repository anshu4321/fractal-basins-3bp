# Literature Notes: Periodic Orbit Discovery in the Planar Equal-Mass Three-Body Problem

## Hristov, Hristova, Dmitrasinovic, Tanikawa (2024)

"Three-body periodic collisionless equal-mass free-fall orbits revisited." Celest. Mech. Dyn. Astron. 136, 7. arXiv:2308.16159.

The authors perform a systematic numerical search for periodic orbits in the equal-mass free-fall (zero angular momentum) three-body problem using a Taylor-series integrator at 100-digit precision combined with Newton-Raphson refinement. They produce a catalog of 12,409 distinct periodic solutions (24,582 initial conditions, with 236 self-dual) for scale-invariant period T* < 80, making this the current state-of-the-art catalog for the planar equal-mass problem. Their search is grid-based, covering an initial-condition space with brute-force discretization followed by high-precision convergence. Our approach differs fundamentally: rather than designing a physics-informed grid or using multi-precision arithmetic, we use a basin classifier trained only on escape outcomes to identify decision boundaries where periodic orbits concentrate, providing a data-driven sampling prior that requires no prior knowledge of any orbit.


## Hristov, Hristova, Tanikawa (2025)

"An extensive search for stable periodic orbits of the equal-mass zero angular momentum three-body problem." arXiv:2510.22802.

This follow-up to the 2024 catalog focuses specifically on linear stability, identifying 971 linearly stable periodic orbits from among the known solutions. The authors establish four stability regions and characterize orbits in each region by patterns in their syzygy sequences, providing a topological-combinatorial classification of stable orbits. They argue these orbits are candidates for KAM stability. Our work is complementary: we discover orbit candidates from escape-basin structure without filtering for stability, and their stability catalog provides a valuable cross-check for any new orbits our method finds.


## Li, Liao (2017)

"More than six hundred new families of Newtonian periodic planar collisionless three-body orbits." Sci. China PMA 60, 129511. arXiv:1705.00527.

Li and Liao use a 1000x1000 grid search over initial velocities in the collinear isosceles configuration space [0,1] x [0,1], combined with clean numerical simulation (CNS, a convergence-verified high-order Taylor method) and Newton-Raphson refinement. They find 695 families of periodic orbits, including the figure-eight family and the 11 families from Suvakov and Dmitrasinovic (2013), plus over 600 new families. This was the first large-scale catalog and established the grid-search-plus-refinement paradigm that Hristov et al. (2024) later scaled up. Our approach replaces the uniform grid with a learned sampling distribution derived from escape-basin classification, concentrating computational effort near basin boundaries where periodic orbits are expected to reside.


## Liao, Li, Yang (2022)

"Three-body problem -- from Newton to supercomputer plus machine learning." New Astron. 96, 101850. arXiv:2106.11010.

The authors train an artificial neural network (ANN) to interpolate initial conditions of known periodic orbits as a function of mass ratio, enabling them to continue orbit families from equal-mass to unequal-mass configurations. Given a known periodic orbit as a seed, the ANN predicts initial conditions at nearby mass ratios, which are then refined with Newton-Raphson. The key limitation is that the method requires a known orbit to start from and extends it along a parameter branch rather than discovering new topological families. Our method requires no known orbit as a seed: the MLP is trained on escape labels (which body escapes, or bound state), and periodic orbits emerge as a byproduct of the classifier's decision-boundary geometry.


## Suvakov, Dmitrasinovic (2013)

"Three classes of Newtonian three-body planar periodic orbits." Phys. Rev. Lett. 110, 114301. arXiv:1303.0181.

Suvakov and Dmitrasinovic discover 13 new periodic orbits (belonging to 11 new families) by numerical grid search over initial conditions in the equal-mass zero-angular-momentum problem. They introduce a topological classification using free-group words on the shape sphere, organizing orbits into four classes defined by geometric and algebraic symmetries. This classification scheme (refined in later work by Dmitrasinovic and collaborators) remains the standard taxonomy for three-body periodic orbits. Our method is agnostic to this classification at the discovery stage: the basin classifier knows nothing about free-group words, yet the orbits it identifies can be classified post hoc using the Suvakov-Dmitrasinovic scheme.


## Gil, Litteri, Rodriguez-Fernandez, Camacho, Vasile (2024)

"Generative Design of Periodic Orbits in the Restricted Three-Body Problem." arXiv:2408.03691.

Gil et al. train a variational autoencoder (VAE) with a CNN backbone on a dataset of known periodic orbits in the circular restricted three-body problem (CR3BP) and sample the learned latent space to generate new orbit candidates. The generated orbits resemble real orbits in shape but exhibit inherent fuzziness and require post hoc refinement. This is the closest ML-flavored work to ours, but there are three key differences: (1) they work in the restricted problem (one massless body), not the full equal-mass problem; (2) they train on known periodic orbits, while our classifier is trained only on escape outcomes and has never seen a periodic orbit during training; (3) their generative model produces orbit shapes, whereas our classifier produces a probability landscape over initial conditions.


## Bramburger, Kutz (2020)

"Poincare maps for multiscale physics discovery and nonlinear Floquet theory." Physica D 408, 132571. arXiv:1908.10958.

Bramburger and Kutz use SINDy (sparse identification of nonlinear dynamics) to learn interpretable, closed-form Poincare maps from trajectory data, with applications including the Sun-Jupiter-Saturn three-body system. The learned maps enable long-time forecasting and provide a data-driven framework for nonlinear Floquet stability analysis of periodic orbits. Their approach is an interpretable ML alternative to black-box neural network models of dynamical systems. Our work differs in objective: Bramburger and Kutz learn the return map to analyze dynamics near known structures, while we learn a classification map over outcomes to discover where new periodic orbits might exist in initial-condition space.


## Summary

All prior periodic-orbit searches in the equal-mass three-body problem rely on either uniform grid search (Suvakov 2013, Li-Liao 2017, Hristov 2024), continuation from known seeds (Liao-Li-Yang 2022), or generative models trained on known orbits (Gil 2024). Our approach is, to our knowledge, the first to use a classifier trained solely on escape/bound outcomes to construct a sampling prior for orbit discovery. The classifier's decision boundaries and prediction entropy concentrate exactly where periodic orbits are expected (at basin boundaries in initial-condition space), providing an efficient, seed-free, physics-agnostic search strategy that complements the existing high-precision catalogs.
