# Numbotics

Flexible prototyping for robotics simulation, planning, and control—built around fast numerical primitives instead of heavyweight class hierarchies.

> **Disclaimer:** Numbotics is an active work in progress. Expect evolving APIs, incomplete documentation, and rapid changes as the stack matures.

## Overview

Numbotics streamlines the workflow from idea to validated robotics experiments. It couples PyBullet-backed physics, Meshcat visualization, sampling-based planners, geometric safe-set computation, and torch-aware math utilities behind a consistent NumPy-first interface. Whether you are iterating on new robot designs, testing planners in cluttered environments, or validating control policies, Numbotics keeps the focus on the numbers while handling the bookkeeping.

## Design Goals

- **Prototype quickly:** Load robots from URDF or build chains programmatically, add collision geometry and obstacles, and simulate in minutes.
- **Stay numerical:** Functions accept standard arrays, homogeneous transforms, and tensors—so you can plug the library into existing research code.
- **Plan and validate:** Integrate motion planners, safe-set generators, and trajectory tools that work directly with the same physics world.
- **Visualize when you need to:** Launch Meshcat for live web visualization or keep things headless for batch experiments.

## Key Capabilities

- **Physics & Environment**
  - Thin wrapper around PyBullet with deterministic world management, headless execution, and optional Meshcat mirroring.
  - Reusable `World.pool` to spin up synchronized sub-worlds for parallel evaluation or sampling.
  - High-level helpers for registering rigid bodies, constraints, joints, and collision filters.

- **Robot Modeling**
  - `GraphChain.from_urdf` imports URDFs—including compound collision geometries—and exposes joint limits, inertias, and forward dynamics.
  - `Arm` adds manipulator-oriented tooling: self-collision management, proximity queries, stateless contexts, and batched kinematics.
  - Direct access to link objects for tweaking masses, friction, or transforms without hunting through PyBullet IDs.

- **Planning & Safe Sets**
  - Sampling-based planners (`RRT`, `RRT*`, `PRM`, `PRM*`) built from reusable state spaces, connectors, and trajectory interpolators.
  - IRIS-style safe-set generation (`IrisSolver`) leveraging convex optimization, collision checking, and multi-threaded sampling.
  - Trajectory utilities (e.g., uniform B-spline generation) for turning discrete plans into smooth references.

- **Visualization & Logging**
  - Meshcat-backed web visualization with transform updates synchronized to the physics step.
  - Utility logging and iostream helpers for keeping notebook and CLI outputs clean.

- **Numerical Utilities**
  - Torch integration (auto device detection, GPU-enabled batch kinematics when tensors are provided).
  - Geometry primitives (polytopes, ellipses, nearest neighbors, point clouds) and optimization helpers.
  - Resource-aware thread pools for distributing heavy collision checks or optimization routines.

## Installation

### Requirements

- macOS or Linux with Python 3.11 (Maybe works on Windows?)
- System dependencies for PyBullet (OpenGL headers) and Meshcat (node-backed web viewer)
- (Optional) MOSEK license for the fastest IRIS solver backend

### Using Poetry

```bash
git clone https://github.com/your-org/numbotics.git
cd numbotics
poetry install
```

### Using pip

```bash
git clone https://github.com/your-org/numbotics.git
cd numbotics
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```


## Quickstart

The snippet below creates a world, loads a Kinova arm from URDF, inserts obstacles, and solves a PRM query. Run it headless or pass `visualize=True` to mirror the state into Meshcat.

```python
import numpy as np

from numbotics.physics import World, GraphChain, Cube
from numbotics.robots import Arm
from numbotics.planning.sampling_based import (
    StateSpace,
    DiscreteConnector,
    ConnectorParams,
    PlannerParams,
)
from numbotics.planning.sampling_based.planners import PRM

# 1) Spin up a world and load a robot
world = World(visualize=True)
world.gravity = np.zeros(3)
chain = GraphChain.from_urdf("numbotics/tests/models/kortex/kinova_cyl.urdf")
arm = Arm(chain)

# 2) Populate the environment
Cube(half_extent=0.4, mass=0.0, position=np.array([1.0, 0.0, 0.2]))

# 3) Configure the planner
space = StateSpace(
    lower_bounds=arm.joint_limits[:, 0],
    upper_bounds=arm.joint_limits[:, 1],
)
connector = DiscreteConnector(
    ConnectorParams(
        resolution=0.1,
        max_distance=np.pi,
        validity_checker=lambda q: not arm.in_collision(q),
    )
)
planner = PRM(
    space=space,
    connector=connector,
    params=PlannerParams(max_iters=200, goal_bias=0.1, k_nearest=15),
)

# 4) Plan between configurations
start = np.zeros(arm.dof)
goal = np.array([0.25, 1.2, -0.4, 0.8, 0.0, -0.3, 0.0])
planner.add_start(start)
planner.add_goal(goal)
planner.plan()
path = planner.solution()

if path is not None:
    trajectory = np.vstack([node.state for node in path])
    print(f"Found path with {trajectory.shape[0]} waypoints.")
```

Want more? Browse the scripted experiments in `numbotics/tests/`, e.g.

```bash
python numbotics/tests/_test_rrt.py --vis
```

## Module Overview

### `numbotics.physics`
- `World` manages PyBullet clients, deterministic stepping, gravity, and Meshcat linkage.
- `PhysicsObject`, `Link`, `Joint`, and `Constraint` wrap PyBullet IDs with attribute accessors for friction, damping, and transforms.
- `GraphChain` builds articulated mechanisms from URDF or manual definitions, exposing batched kinematics, collision shapes, and joint-space caches.

### `numbotics.robots`
- `Arm` decorates a `GraphChain` with manipulator-specific utilities (collision pair management, jacobian helpers, stateless contexts, world pooling).
- `helpers` contains fast kinematics and jacobian kernels for control and optimization loops.

### `numbotics.planning`
- `sampling_based` offers reusable state spaces, connectors (discrete and continuous), and planners (PRM, PRM*, RRT, RRT*).
- `safe_sets.IrisSolver` implements IRIS-NP style convex region expansion with threaded collision checking and CVXPy/MOSEK subproblems.
- `trajectories.unit_bspline` converts sparse waypoints into smooth splines for controller references.

### `numbotics.graphics`
- Meshcat-based `Visualizer` for registering shapes, setting transforms, tweaking colors, and keeping a persistent browser session.
- `visualizer.VisualShape` ties into physics collision shapes for consistent rendering.

### `numbotics.math`
- Geometry primitives (`Polytope`, `Ellipse`, `ConvexSet`) for collision-free region reasoning.
- Spatial helpers (`trans_mat`, `rot_diff`) with NumPy- and torch-aware implementations.

### `numbotics.utils`
- Logging, IO streams, threading pools, mesh loading (including convex decomposition via VHACD), and shape utilities (boxes, cylinders, compound geometries).
- Configuration toggles in `numbotics.config` for Torch, visualization, and ffmpeg availability checks.

### `numbotics.learning`
- Lightweight neural-network scaffolding for leveraging torch modules alongside the rest of the stack.

## Repository Structure

```text
numbotics/
  physics/        # PyBullet world, objects, chains, constraints
  robots/         # High-level robot abstractions (Arm, helpers)
  planning/       # Sampling planners, IRIS safe sets, trajectories
  graphics/       # Meshcat visualizer bindings
  math/           # Geometry, optimization, spatial math
  utils/          # Logging, shapes, threading, IO, mesh helpers
  learning/       # Torch network utilities
  tests/          # End-to-end examples and regression suites
```

## Development Workflow

- **Tests:** `pytest` (configured to look inside `numbotics/tests`). Many tests double as runnable examples.
- **Linting:** `ruff` is included as a dev dependency. Run `ruff check .` before opening a PR.
- **Formatting:** The codebase favors explicit, math-heavy style—follow existing patterns rather than enforcing auto-formatters blindly.
- **Type hints:** Key public APIs are annotated; contributions should maintain or improve typing coverage.

## Example Assets

- `numbotics/tests/models/` contains URDFs and meshes (e.g., Kinova Gen3) ready for simulation.
- VHACD convex decompositions are pre-generated to keep collision detection stable out of the box.

## Roadmap

- Add support for reinforcement learning policies.
- Broaden planner coverage (task-space RRT, kinodynamic variants).
- Streamline optional dependency handling for lighter-weight installs.
- Expand support for different types of robots, the two biggest being robot arms and quadrotors.
- Improve documentation!

## Contributing

Issues and pull requests are welcome. Please:
- Include a focused test or example demonstrating new functionality.
- Update the README or inline docstrings when APIs change.
- Follow the GPLv3 licensing terms for derivative works.

## License

Distributed under the GNU General Public License v3.0 (GPL-3.0). See `LICENSE` for details.


