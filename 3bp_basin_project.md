# Learning the Basin Structure of the Planar Three-Body Problem

A long-running personal research project. ML + GPU + celestial mechanics. Not Dirac work.

## The Question

For the equal-mass, zero-angular-momentum planar three-body problem, the space of initial conditions partitions into basins corresponding to long-term fate: which body escapes, bound resonant motion, or the measure-zero set of periodic orbits. The boundaries are believed to be fractal. **No global predictive map of this partition exists.** Given an arbitrary IC on Montgomery's shape sphere, the only way to know the outcome is to integrate.

**Core question:** Can a neural network, trained on millions of GPU-integrated trajectories, learn the basin partition well enough to (a) predict escape outcome and time orders of magnitude faster than integration, (b) expose the geometric structure of basin boundaries through its learned representations, and (c) discover new periodic orbit families as critical points of its own loss landscape on the shape sphere?

## Why This Is Worth Doing

1. Genuinely open. Šuvakov and Dmitrašinović (2013) found ~13 new periodic orbits by brute-force shooting. Liao's group has since found thousands via high-precision GPU integration. The search is still essentially random. Nobody has used a learned basin model to guide it.
2. Breen et al. 2020 showed an MLP can replace Brutus for forward trajectory prediction on the 3BP, but they did not study the partition of phase space. That is the gap to leapfrog.
3. Fractal targets are a clean testbed for spectral bias, neural operators, and whether networks can represent functions with non-integer Hausdorff dimension.
4. Physics intuition matters. The shape sphere reduction (Montgomery coordinates) is where domain knowledge beats brute-force ML.

## Technical Stack

- JAX + `diffrax` for differentiable, vmappable integration. Fall back to a custom Wisdom-Holman or Kustaanheimo-Stiefel regularized integrator in CUDA if `diffrax` symplectic options are too slow near close encounters.
- Shape sphere parameterization (Montgomery): reduce to a 2-sphere plus momentum. Sample uniformly with respect to the natural measure.
- Equinox or Flax for models. Start with MLPs, escalate to FNO / DeepONet.
- Weights and Biases for tracking. Vercel for the eventual interactive blog.

## Milestones

### M1. Integrator and dataset (weeks 1 to 3)

- Implement equal-mass planar 3BP in JAX with adaptive symplectic integration.
- Validate against the figure-eight orbit (Chenciner-Montgomery) to machine precision over many periods.
- Implement shape sphere sampling and the inverse map back to Cartesian ICs.
- Define escape criterion (Standish or similar) and label scheme: {body 1 escapes, body 2 escapes, body 3 escapes, bound at cutoff}.
- Generate 10M ICs, integrate to fixed physical time on a single GPU. Store labels and escape times.

### M2. Baseline classifier (weeks 4 to 6)

- Train a small MLP on (shape sphere coords, momentum) to predict label and log escape time.
- Measure accuracy as a function of distance to known periodic orbits.
- Visualize the learned basin map on the shape sphere. Compare against published Agekian-Anosova maps.
- Ship a blog post with an interactive shape sphere viewer. This is the natural first deliverable.

### M3. Boundary geometry (months 2 to 4)

- Use model uncertainty (deep ensembles or evidential output) to localize basin boundaries.
- Estimate Hausdorff dimension of boundaries in different shape sphere regions via box counting on the model's predictions, then verify on a subsample with direct integration.
- Compare MLP, SIREN, FNO, and DeepONet on their ability to fit the fractal target. This is a clean ML paper on its own.

### M4. Periodic orbit discovery (months 4 to 9)

- Periodic orbits are fixed points of the return map and live exactly on basin boundaries. Treat the trained model's uncertainty surface as a heuristic for where to launch high-precision shooting methods.
- Active learning loop: model proposes candidates, high-precision integrator verifies, verified orbits go back into the training set.
- Target: rediscover figure-eight, Broucke-Hénon, and Šuvakov-Dmitrašinović families without telling the model where they are. Stretch: find a new family.

### M5. Writeup (month 9+)

Two outputs. A technical blog with the interactive viewer, and if M4 yields anything novel, a short paper.

## Risks and Things That Could Kill It

- Close encounters destroy non-regularized integrators. Budget time for KS regularization.
- Fractal targets may be genuinely unlearnable beyond a certain resolution. That itself is a publishable negative result if framed well.
- Decoherence risk. Mitigation: weekly build targets, each producing a visible artifact (a plot, a notebook, a deployed page). No abstract study weeks.

## Anchor References

- Montgomery, "The N-body problem, the braid group, and action-minimizing periodic solutions" (1998).
- Chenciner and Montgomery, "A remarkable periodic solution of the three-body problem" (2000).
- Šuvakov and Dmitrašinović, "Three classes of Newtonian three-body planar periodic orbits" (2013).
- Liao et al., clean numerical simulation papers on 3BP periodic orbit catalogs.
- Breen, Foley, Boekholt, Portegies Zwart, "Newton vs the machine" (2020).
- Agekian and Anosova, classical maps of the triple system phase space.

## Week 1 Concrete Targets

1. JAX integrator running on GPU, vmapped over a batch of ICs.
2. Figure-eight orbit reproduced and energy conservation plotted over 100 periods.
3. Shape sphere sampler with a notebook visualizing 10k sampled points.

That is the whole thing. Pick it up, hand it to Claude Code, start at M1.
