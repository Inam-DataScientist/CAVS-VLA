# CAVS-VLA v2

### Compositional Safety for Multi-Agent, Language-Conditioned Driving Policies

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-supported-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![nuPlan](https://img.shields.io/badge/nuPlan-supported-1f6feb)](#data)
[![Waymo](https://img.shields.io/badge/WOMD-supported-1f6feb)](#data)
[![CARLA](https://img.shields.io/badge/CARLA-supported-2ea44f)](#carla)
[![Safety](https://img.shields.io/badge/safety-certified%20stack-critical-b31b1b)](#safety-stack)
[![License](https://img.shields.io/badge/license-see%20LICENSE-lightgrey)](LICENSE)

</div>

---

## Overview

**CAVS-VLA v2** is a research implementation for **multi-agent,
language-conditioned autonomous driving with compositional safety
assurance**.

The repository combines a learned driving policy with independent
verification, reachability analysis, uncertainty calibration, predictive
control, and runtime safety shielding.

The central design principle is:

> **The learned policy proposes behavior. The safety stack determines whether
> that behavior can be executed safely.**

The implementation is designed so that the learned policy and the
safety-critical components remain modular.

```text
                         CAVS-VLA v2
                              │
                              ▼
                  ┌─────────────────────┐
                  │   Scene / Context   │
                  │  nuPlan / WOMD /    │
                  │       CARLA         │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   VLA Policy        │
                  │                     │
                  │  Action Tokens      │
                  │  (acceleration,     │
                  │   curvature)        │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Neural Verification │
                  │                     │
                  │   IBP / X0 bounds   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Safety Analysis   │
                  │                     │
                  │ HJ Reachability     │
                  │ Conformal Sets      │
                  │ Interaction Bounds  │
                  └──────────┬──────────┘
                             │
                             ▼
                     ┌───────────────┐
                     │  CERTIFIED?   │
                     └───────┬───────┘
                         YES  │  NO
                              │
             ┌────────────────┘
             ▼
       Execute Proposal

                              NO
                              │
                              ▼
                    ┌──────────────────┐
                    │ Predictive MPC   │
                    └────────┬─────────┘
                             │
                       Re-certify
                             │
                    ┌────────┴────────┐
                    │                 │
                   YES               NO
                    │                 │
                    ▼                 ▼
              Execute MPC       HJ Backup /
                                Emergency Brake
````

---

# ✨ Key Components

| Component                    | Purpose                                                                          |
| :--------------------------- | :------------------------------------------------------------------------------- |
| **Leak-Free Data Pipeline**  | Builds temporally valid training inputs from nuPlan and WOMD                     |
| **Multi-Agent Policy**       | Predicts driving behavior from scene context                                     |
| **Action-Token Decoder**     | Produces discrete action representations for acceleration and curvature          |
| **IBP Verification**         | Computes certified neural output bounds                                          |
| **X0 Control Set**           | Determines the set of controls compatible with the certified input region        |
| **HJ Reachability**          | Computes conservative safety information for longitudinal interactions           |
| **Conformal Prediction**     | Produces calibrated prediction sets for other agents                             |
| **Predictive Safety Shield** | Combines certificates, interaction bounds, MPC, and backup control               |
| **MPC**                      | Searches for a locally feasible alternative when the proposal is not certifiable |
| **HJ Backup**                | Provides a least-restrictive safety fallback                                     |
| **CARLA Bridge**             | Enables closed-loop stress testing with a simulator                              |
| **nuPlan Adapter**           | Connects the policy and safety stack to the nuPlan planner interface             |
| **CEG Refinement**           | Mines failures and feeds train-split counterexamples back into refinement        |
| **Self-Test Suite**          | Verifies core pipeline wiring before expensive experiments                       |

---

# 🛡️ Safety Stack

The safety stack is deliberately separated from the learned policy.

A proposed trajectory passes through the following stages:

```text
Policy Proposal
      │
      ▼
┌──────────────────────┐
│ Neural Output Bounds │
│       (IBP)          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Certified Control  │
│        Set X0        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Interval Tube     │
│   over Horizon       │
└──────────┬───────────┘
           │
           ├───────────────┐
           ▼               ▼
     HJ Reachability   Conformal Sets
           │               │
           └───────┬───────┘
                   ▼
          Safety Certificate
                   │
             ┌─────┴─────┐
             │           │
            YES          NO
             │           │
             ▼           ▼
          Execute       MPC
                         │
                         ▼
                    Re-certify
                         │
                   ┌─────┴─────┐
                   │           │
                  YES          NO
                   │           │
                   ▼           ▼
                Execute     HJ Backup /
                            Emergency Brake
```

The runtime decision hierarchy is therefore:

```text
1. Execute the learned proposal if certified.
2. Otherwise search for a locally safe MPC alternative.
3. Re-certify the MPC solution.
4. If certification still fails, use the HJ backup.
5. Emergency braking remains available as the final fallback.
```

The implementation is designed so that **MPC convergence is not itself the
basis of the safety decision**; the safety decision is based on the
specified model, disturbance, actuator, input, and prediction assumptions.

---

# 🧠 Policy Interface

The learned policy operates on a structured multi-agent scene representation.

The policy produces:

```text
                    Policy Output
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Action Tokens            Agent Prediction
             │                       │
       ┌─────┴─────┐                 │
       │           │                 ▼
       ▼           ▼          Uncertainty /
 Acceleration  Curvature      Prediction Scale
```

The action representation is based on:

```text
u = (a, κ)
```

where:

* `a` = longitudinal acceleration
* `κ` = curvature

The policy also produces trajectory and other-agent prediction outputs
used by the downstream safety stack.

---

# 📐 Core Tensor Contracts

The repository uses fixed tensor contracts to keep the data pipeline,
training model, simulation, verification, and safety stack consistent.

`B` denotes batch size.

| Tensor           |       Shape       | Description                   |
| :--------------- | :---------------: | :---------------------------- |
| `traj`           |  `(B, 64, 91, 7)` | Actor trajectories            |
| `valid`          |   `(B, 64, 91)`   | Actor validity mask           |
| `poly`           | `(B, 192, 20, 2)` | Map polylines                 |
| `poly_attr`      |   `(B, 192, 6)`   | Map attributes                |
| `poly_tl`        |   `(B, 192, 91)`  | Traffic-light state           |
| `feat.ego`       |    `(B, 11, 7)`   | Ego features                  |
| `feat.agents`    | `(B, 32, 11, 12)` | Agent features                |
| `feat.map`       |  `(B, 64, 20, 4)` | Map features                  |
| `map_attr`       |    `(B, 64, 9)`   | Map attributes                |
| `out.ctrl`       |  `(B, 6, 40, 2)`  | Control outputs               |
| `out.traj`       |  `(B, 6, 80, 4)`  | Predicted trajectories        |
| `acc_logits`     |  `(B, 6, 40, 25)` | Acceleration action tokens    |
| `kap_logits`     |  `(B, 6, 40, 41)` | Curvature action tokens       |
| `agent_pred`     |  `(B, 32, 80, 2)` | Other-agent prediction        |
| `agent_logscale` |  `(B, 32, 80, 2)` | Prediction uncertainty        |
| `cert.u_lo`      |    `(B, 40, 2)`   | Certified lower control bound |
| `cert.u_hi`      |    `(B, 40, 2)`   | Certified upper control bound |

The raw trajectory representation is:

```text
[x, y, yaw, vx, vy, length, width]
```

with the ego vehicle represented by actor `0`.

The observer frame is defined relative to the ego vehicle at the reference
time.

---

# 📂 Repository Structure

```text
CAVS-VLA/
│
├── cavs_vla/
│   │
│   ├── data/
│   │   ├── nuplan.py
│   │   ├── waymo.py
│   │   ├── maps.py
│   │   ├── scene.py
│   │   └── build.py
│   │
│   ├── model/
│   │   ├── layers.py
│   │   └── vla.py
│   │
│   ├── safety/
│   │   ├── hj.py
│   │   ├── conformal.py
│   │   ├── verifier.py
│   │   └── shield.py
│   │
│   ├── sim/
│   │   ├── logsim.py
│   │   ├── metrics.py
│   │   └── carla_env.py
│   │
│   ├── dynamics.py
│   ├── features.py
│   ├── nuplan_planner.py
│   ├── legacy_probe.py
│   └── selftest.py
│
├── configs/
│   ├── default.yaml
│   ├── nuplan_mini.yaml
│   └── full_nuplan_womd.yaml
│
├── tests/
│   └── test_numpy_stack.py
│
├── scripts/
│   └── run_pipeline.sh
│
├── data/
│   └── external/
│
├── results/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 🔧 Module Guide

## Data

### `cavs_vla/data/nuplan.py`

Reads nuPlan logs and extracts the scene information required by the pipeline.

Includes:

* lidar sweeps
* ego pose
* dynamic actors
* static objects
* traffic lights
* route information

---

### `cavs_vla/data/waymo.py`

Reads Waymo Open Motion Dataset Scenario records.

The implementation uses the Scenario representation without requiring TensorFlow
for the reader itself.

---

### `cavs_vla/data/maps.py`

Provides map access through the supported nuPlan map interfaces:

* nuPlan devkit
* GeoPackage
* compatible JSON/map representations

---

### `cavs_vla/data/scene.py`

Converts raw logs into fixed-shape scene representations.

The scene representation is generated in the ego-at-reference-time coordinate
frame.

---

### `cavs_vla/data/build.py`

Builds the processed dataset and manages:

* log-disjoint splits
* per-log staging
* memory-mapped merging
* dataset manifests
* hashes
* reproducibility metadata

---

# 🤖 Model

## `cavs_vla/model/vla.py`

Main policy implementation.

The model contains the policy and prediction heads used by the downstream
safety stack.

The architecture is designed to expose verification-friendly operations.

---

## `cavs_vla/model/layers.py`

Contains the neural-network building blocks.

The layers provide both normal forward operations and corresponding
verification-compatible implementations where required by the IBP pipeline.

---

# 🛡️ Safety

## `cavs_vla/safety/hj.py`

Hamilton-Jacobi reachability implementation.

Provides:

* 3-D HJ safety computation
* conservative envelopes
* validation routines
* differentiable lookup functionality

---

## `cavs_vla/safety/conformal.py`

Conformal prediction utilities for other-agent prediction.

Prediction sets are generated separately by agent type and can be used by the
runtime safety layer.

---

## `cavs_vla/safety/verifier.py`

Neural-output verification and certified control-set construction.

Responsible for:

* physical input region `X0`
* IBP bounds
* certified control bounds
* ONNX export
* VNNLIB generation

---

## `cavs_vla/safety/shield.py`

Runtime safety integration.

Combines:

* policy proposals
* certified control regions
* interval trajectory tubes
* interaction constraints
* HJ information
* conformal prediction sets
* MPC
* HJ fallback

---

# 🚗 Dynamics

## `cavs_vla/dynamics.py`

Contains the vehicle dynamics and disturbance model used by the safety stack.

The implementation includes:

* kinematic bicycle dynamics
* actuator lag
* noise
* latency/delay
* disturbance estimation

---

# 🧪 Simulation

## `cavs_vla/sim/logsim.py`

Batched closed-loop simulator for controlled experiments.

Supports:

* IDM-style agents
* control latency
* replanning intervals
* multiple controlled CAVs
* plan exchange

---

## `cavs_vla/sim/metrics.py`

Evaluation and statistical analysis utilities.

Supports:

* confidence intervals
* paired comparisons
* statistical tests
* closed-loop metrics

---

## `cavs_vla/sim/carla_env.py`

CARLA integration.

Provides:

* simulator bridge
* scenario execution
* safety sensors
* disturbance measurement
* stress-testing infrastructure

---

# 🔌 Planner Integration

## `cavs_vla/nuplan_planner.py`

Provides the nuPlan planner adapter.

The adapter exposes the system through the nuPlan
`AbstractPlanner` interface.

This allows the safety-aware policy to be connected to official nuPlan
closed-loop evaluation.

---

# 🔎 Leakage Detection

## `cavs_vla/legacy_probe.py`

Provides the legacy-pipeline probe used to inspect the previous implementation.

The purpose is to quantify information leakage in the old pipeline and verify
that the new data construction does not depend on future trajectory information.

---

# ✅ Self-Test

## `cavs_vla/selftest.py`

Runs the end-to-end synthetic-data checks required before expensive experiments.

The self-test should pass before proceeding to dataset construction or training.

Expected result:

```text
SELFTEST PASSED
```

---

# 💾 Data

The repository does **not** redistribute the original nuPlan or Waymo datasets.

Users must obtain the corresponding datasets according to their respective
licenses and terms.

Expected directory structure:

```text
data/
└── external/
    │
    ├── nuplan/
    │   ├── **/*.db
    │   └── maps/
    │       └── <location>/<version>/map.gpkg
    │
    └── womd/
        └── scenario/
            ├── training/
            └── validation/
```

Paths are configurable through the YAML configuration files in:

```text
configs/
```

---

# 📦 Installation

## 1. Create the environment

```bash
conda create -n cavs2 python=3.10 -y
conda activate cavs2
```

---

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

The core environment uses packages such as:

* PyTorch
* NumPy
* SciPy
* PyYAML

---

## 3. Install nuPlan support

Preferred:

```bash
pip install nuplan-devkit
```

Alternatively:

```bash
pip install pyogrio geopandas
```

for direct map-file access.

---

## 4. Optional Waymo support

```bash
pip install waymo-open-dataset-tf-2-12-0
```

Only the Scenario representation is required by the reader.

---

## 5. Optional verification tools

For ONNX/VNNLIB export and external verification:

```bash
pip install onnx
```

The exported models can subsequently be checked using compatible
verification tooling such as alpha-beta-CROWN.

---

# 🚀 Quick Start

Before using real datasets, run the synthetic self-test.

```bash
python -m cavs_vla selftest --workdir /tmp/cavs_selftest
```

Then run:

```bash
python tests/test_numpy_stack.py
```

The self-test should report:

```text
SELFTEST PASSED
```

Do not proceed to full training if the self-test fails.

---

# 🏗️ Build the Dataset

Start with the small nuPlan split:

```bash
python -m cavs_vla build-data \
    --config configs/nuplan_mini.yaml
```

Check the action-space fit:

```bash
python -m cavs_vla kinematic-check \
    --config configs/nuplan_mini.yaml
```

After validating the pipeline, configure the full nuPlan/WOMD dataset.

---

# 🌪️ Estimate Safety Disturbances

Estimate the disturbance envelope:

```bash
python -m cavs_vla estimate-disturbance \
    --config configs/default.yaml
```

The resulting estimate is used to configure:

```text
hj.w_bar
```

Build the HJ safety representation:

```bash
python -m cavs_vla build-hj \
    --config configs/default.yaml
```

---

# 🧠 Train the Policy

For distributed training:

```bash
torchrun \
    --nproc_per_node=8 \
    -m cavs_vla train \
    --config configs/full_nuplan_womd.yaml \
    --run main
```

Adjust `--nproc_per_node` to match the available GPU resources.

---

# 📏 Calibrate the Safety Predictor

After training:

```bash
python -m cavs_vla calibrate \
    --config <config> \
    --ckpt <run>/best.pt
```

This produces the conformal calibration information used by the safety
pipeline.

---

# 🔐 Verify the Trained Model

Run neural verification:

```bash
python -m cavs_vla verify \
    --config <config> \
    --ckpt <run>/best.pt
```

The verification pipeline produces the information required to inspect
certificate width and export verification artifacts.

---

# 🧪 Open-Loop Evaluation

Run the open-loop evaluation:

```bash
python -m cavs_vla eval-open \
    --config <config> \
    --ckpt <run>/best.pt
```

This evaluation can include:

* baseline comparisons
* input ablations
* alternative prediction methods

---

# 🔄 Closed-Loop Evaluation

Run the closed-loop evaluation:

```bash
python -m cavs_vla eval-closed \
    --config <config> \
    --ckpt <run>/best.pt
```

For selected methods:

```bash
python -m cavs_vla eval-closed \
    --config <config> \
    --ckpt <run>/best.pt \
    --methods B1,B7
```

---

# 🧪 Stress Testing

## Model mismatch

Test conditions outside the nominal disturbance assumptions:

```bash
python -m cavs_vla eval-closed \
    --config <config> \
    --ckpt <run>/best.pt \
    --methods B1,B7 \
    --tag mismatch \
    --set sim.accel_noise=1.0 sim.tau_a=0.3
```

---

## Latency

Test increased planning/control latency:

```bash
python -m cavs_vla eval-closed \
    --config <config> \
    --ckpt <run>/best.pt \
    --methods B1,B7 \
    --tag latency \
    --set sim.replan_interval=5 sim.latency_steps=2
```

---

## Multi-CAV coordination

Run the multi-controlled-vehicle setting:

```bash
python -m cavs_vla eval-closed \
    --config <config> \
    --ckpt <run>/best.pt \
    --methods B1,B6,B7 \
    --tag cav4 \
    --set sim.num_controlled=4
```

---

# 🧩 Safety Ablations

The implementation provides a configurable method table.

|   ID   |    Certificate    |  HJ |     MPC    | Fallback    |
| :----: | :---------------: | :-: | :--------: | :---------- |
| **B0** | Expert log replay |  –  |      –     | –           |
| **B1** |         –         |  –  |      –     | –           |
| **B2** |         –         |  –  |  Always on | –           |
| **B3** |         –         |  ✓  |      –     | HJ backup   |
| **B4** |         ✓         |  –  |      –     | Max braking |
| **B5** |         ✓         |  –  | On failure | Max braking |
| **B6** |    Point output   |  ✓  | On failure | HJ backup   |
| **B7** |         ✓         |  ✓  | On failure | HJ backup   |

The complete method configuration is defined through:

```text
config.METHOD_TABLE
```

---

# 🔁 Counterexample-Guided Refinement

Failures can be mined from the training split:

```bash
python -m cavs_vla mine-failures \
    --config <config> \
    --ckpt <run>/best.pt \
    --method B1 \
    --out results/failures_train.json
```

Run counterexample-guided refinement:

```bash
python -m cavs_vla ceg \
    --config <config> \
    --ckpt <run>/best.pt \
    --failures results/failures_train.json \
    --run ceg1
```

After refinement, rerun the evaluation pipeline.

---

# 🚦 nuPlan Closed-Loop Evaluation

The repository includes a nuPlan planner adapter:

```text
cavs_vla.nuplan_planner.CavsPlanner
```

Use the adapter for official nuPlan closed-loop evaluation.

The implementation distinguishes between:

```text
Repository log simulator
        ≠
Official nuPlan simulator
```

For official nuPlan closed-loop results, use the provided planner adapter and
the corresponding nuPlan evaluation infrastructure.

---

# 🎮 CARLA

A CARLA server must be running before executing CARLA experiments.

Run:

```bash
python -m cavs_vla carla \
    --config <config> \
    --ckpt <run>/best.pt \
    --method B7
```

To measure disturbances:

```bash
python -m cavs_vla carla \
    --config <config> \
    --ckpt <run>/best.pt \
    --measure-disturbance
```

Make sure the CARLA client and server versions are compatible.

---

# 🔁 Full Pipeline

The main pipeline can also be chained through:

```bash
scripts/run_pipeline.sh
```

The script covers the primary stages:

```text
Data
 ↓
Kinematic Check
 ↓
Disturbance Estimation
 ↓
HJ Construction
 ↓
Training
 ↓
Calibration / Verification
 ↓
Evaluation
 ↓
Failure Mining / Refinement
```

For reproducibility, it is recommended to run the individual stages first and
inspect their outputs before using the complete pipeline script.

---

# 🧪 Recommended Reproducibility Workflow

A clean reproduction should follow this order:

```text
┌──────────────────────┐
│ 1. Install           │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. Self-Test         │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. Prepare Data      │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 4. Kinematic Check   │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 5. HJ Construction   │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 6. Train Policy      │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 7. Calibration       │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 8. Verification      │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 9. Evaluation        │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 10. Stress Tests     │
└──────────────────────┘
```

---

# 🔬 Verification Workflow

The verification-oriented implementation follows:

```text
Neural Policy
      │
      ▼
ONNX Export
      │
      ▼
VNNLIB Specification
      │
      ▼
IBP Bounds
      │
      ▼
Certified Control Set
      │
      ▼
Safety Stack
```

IBP bounds are computed with the verification-compatible neural operations
implemented in the model layers.

For high-assurance verification experiments, representative cases should be
rechecked with an external neural verifier using the exported ONNX/VNNLIB
representation.

---

# 📊 Outputs

Typical experiment outputs are stored under:

```text
results/
```

Depending on the experiment, outputs can include:

```text
results/
├── checkpoints/
├── calibration/
├── verification/
├── failures/
├── evaluation/
├── carla/
└── logs/
```

Exact output locations are determined by the selected configuration and run
name.

---

# ⚙️ Configuration

Configuration files are stored in:

```text
configs/
```

Main configurations include:

```text
configs/default.yaml
configs/nuplan_mini.yaml
configs/full_nuplan_womd.yaml
```

Important parameters include:

* dataset paths
* map paths
* training parameters
* dynamics parameters
* disturbance bounds
* HJ configuration
* simulation parameters
* latency
* number of controlled vehicles
* evaluation settings

Avoid hard-coding local paths in source files. Use configuration files so that
experiments remain portable across machines.

---

# 🔒 Safety Assumptions

The safety guarantees provided by the implementation are conditional on the
assumptions used by the corresponding safety components.

These include assumptions concerning:

* kinematic dynamics
* actuator behavior
* disturbance bounds
* initial-state region
* other-agent acceleration bounds
* conformal prediction coverage
* numerical outward padding for IBP

The repository should therefore be interpreted as a **research implementation
of compositional safety mechanisms**, not as an unconditional guarantee of
real-world autonomous-driving safety.

---

# ⚠️ Important Limitations

## No camera input

The current nuPlan and WOMD pipeline does not use raw camera images.

The language input is derived from route information.

The safety stack is policy-agnostic and can be connected to another driving
policy through the expected control and trajectory interfaces.

---

## Longitudinal HJ model

The current HJ formulation models longitudinal interaction along the proposed
path.

Lateral interactions are handled through the corresponding prediction and
safety mechanisms rather than being represented by a complete lateral HJ
reachable-set formulation.

---

## Conformal coverage

Conformal prediction coverage relies on the relevant exchangeability
assumption.

Closed-loop coverage should therefore be measured and reported for the
operating distribution being evaluated.

---

## WOMD route information

WOMD does not provide the same route representation used by nuPlan.

The current pipeline reconstructs the route from the logged path and marks this
information as `oracle` in the dataset manifest.

For an ablation without route information:

```yaml
data:
    use_route: false
```

---

## Numerical verification

IBP bounds are implemented with an outward numerical pad.

For formal verification claims, representative cases should be independently
checked using the exported ONNX/VNNLIB representation and an external verifier.

---

## Simulator distinction

The repository includes a batched log simulator for controlled experimentation.

This simulator is **not the official nuPlan simulator**.

Official nuPlan closed-loop evaluation should use:

```text
cavs_vla.nuplan_planner.CavsPlanner
```

through the corresponding nuPlan evaluation infrastructure.

---

# 🧰 Useful Commands

### Self-test

```bash
python -m cavs_vla selftest --workdir /tmp/cavs_selftest
```

### Dataset construction

```bash
python -m cavs_vla build-data \
    --config configs/nuplan_mini.yaml
```

### Kinematic validation

```bash
python -m cavs_vla kinematic-check \
    --config configs/nuplan_mini.yaml
```

### Disturbance estimation

```bash
python -m cavs_vla estimate-disturbance \
    --config configs/default.yaml
```

### HJ construction

```bash
python -m cavs_vla build-hj \
    --config configs/default.yaml
```

### Training

```bash
torchrun \
    --nproc_per_node=8 \
    -m cavs_vla train \
    --config configs/full_nuplan_womd.yaml \
    --run main
```

### Calibration

```bash
python -m cavs_vla calibrate \
    --config <config> \
    --ckpt <checkpoint>
```

### Verification

```bash
python -m cavs_vla verify \
    --config <config> \
    --ckpt <checkpoint>
```

### Open-loop evaluation

```bash
python -m cavs_vla eval-open \
    --config <config> \
    --ckpt <checkpoint>
```

### Closed-loop evaluation

```bash
python -m cavs_vla eval-closed \
    --config <config> \
    --ckpt <checkpoint>
```

### Failure mining

```bash
python -m cavs_vla mine-failures \
    --config <config> \
    --ckpt <checkpoint> \
    --method B1 \
    --out results/failures_train.json
```

### Counterexample refinement

```bash
python -m cavs_vla ceg \
    --config <config> \
    --ckpt <checkpoint> \
    --failures results/failures_train.json \
    --run ceg1
```

### CARLA

```bash
python -m cavs_vla carla \
    --config <config> \
    --ckpt <checkpoint> \
    --method B7
```

---

# 📝 Development Notes

The project is intended for research use.

Before committing changes, it is recommended to run:

```bash
python -m cavs_vla selftest --workdir /tmp/cavs_selftest
python tests/test_numpy_stack.py
```

Then validate the affected pipeline stage before running full-scale experiments.

For changes to safety-critical modules, test both:

```text
Nominal behavior
+
Failure / boundary behavior
```

---

# 📌 Reproducibility Checklist

Before reporting results from a new experiment, record:

* [ ] Git commit
* [ ] Configuration file
* [ ] Dataset version
* [ ] Dataset split
* [ ] Random seed
* [ ] Checkpoint
* [ ] Hardware
* [ ] Software environment
* [ ] Safety assumptions
* [ ] Disturbance configuration
* [ ] Latency configuration
* [ ] Evaluation method
* [ ] Verification artifacts

A recommended run directory is:

```text
results/<experiment>/
├── config.yaml
├── checkpoint.pt
├── metrics.json
├── verification.json
├── environment.txt
└── README.md
```

---

# 📄 Associated Research

This repository contains the implementation and experimental infrastructure
associated with the research project.

The README intentionally focuses on:

* software architecture
* installation
* data interfaces
* safety modules
* commands
* reproducibility
* implementation assumptions

For the scientific formulation, methodology, theoretical analysis, and
experimental discussion, please refer to the associated manuscript.

---

# 📜 License

See [`LICENSE`](LICENSE) for the applicable license terms.

---

# 🙏 Acknowledgements

This implementation builds on publicly available datasets, simulators, and
research software ecosystems.

Please follow the respective licenses and citation requirements of:

* **nuPlan**
* **Waymo Open Motion Dataset**
* **CARLA**
* **PyTorch**
* **ONNX**
* Other third-party dependencies listed in `requirements.txt`

---

# ⭐ If You Use This Repository

If this implementation contributes to your research, please cite the
associated work and follow the citation requirements of the underlying
datasets and software dependencies.

Citation information for this repository can be added here when the associated
research is publicly released.

---

<div align="center">

## CAVS-VLA v2

**Learn → Verify → Predict → Shield → Execute**

*A modular research framework for safety-aware multi-agent autonomous driving.*

</div>
