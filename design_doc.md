# Design Doc: VLM-Guided Active Probing for Adaptive Dynamic Manipulation

**Status:** Draft  
**Author:** [Your Name]  
**Date:** April 2026  

---

## 0. Branch: `vlm_only` — Level 3 Simplification

> This section describes what is actually implemented on the `vlm_only` branch.
> Sections 1–14 below describe the **full** research plan (with learned belief
> and throw networks), which this branch defers. If you want the short answer
> to "what does this branch do," read only Section 0.

### 0.1 Why this branch exists

The full plan (§4, §10) routes probe observations through a learned belief-update
MLP and throw decisions through a learned throw MLP. Those two networks are
plumbing, not the research contribution, and they require ~weeks of dataset
generation + training before the core thesis can be tested empirically.

The `vlm_only` branch drops both networks. The VLM alone reads the rendered
image, picks probes, consumes raw probe observations as text, and when it
decides to throw, directly emits `(θ, v, Δt)`. This collapses the experiment
to verify the core claim into a single question:

> **Does VLM throw success rate increase with probe budget?**

If yes, the thesis (probing helps) is directionally validated even with worse
absolute numbers than a trained pipeline. If no, no amount of training
infrastructure will save the thesis — so it is the right first test.

### 0.2 Architecture (what replaces the learned modules)

```
┌─────────────────────────────────────────────────────────────┐
│                    VLM-Only Episode Loop                    │
│                                                             │
│  1. Render (side + top) → PIL image                         │
│  2. Prompt VLM with: image, target distance, probe budget,  │
│     and all prior probe observations as key=value text      │
│  3. VLM emits either `ACTION: <probe_name>` or              │
│     `ACTION: THROW` + `THETA/V/DT` numeric lines            │
│  4. If probe → execute, append observations to history,     │
│     loop back to 2 until budget exhausted                   │
│  5. If throw (or budget forced) → execute, record success   │
└─────────────────────────────────────────────────────────────┘
```

No belief state, no `(μ, σ)`, no learned inference. The VLM's "belief" is
whatever text it has accumulated in its prompt context.

### 0.3 Module map

| File | Purpose |
|------|---------|
| `tossing/vlm/client.py` | `VLMClient` ABC + `AnthropicClient` (Claude), `OpenAIClient` (GPT-4o), `FakeVLMClient` (tests). PIL → base64 encoding. Shared disk cache keyed on `(provider, model, prompt, image)` — repeat eval runs hit the cache and cost $0. |
| `tossing/vlm/prompts.py` | System prompt describing the 5 probes + THROW contract. `build_user_message()` renders probe history as `key=value` text. |
| `tossing/vlm/parser.py` | Regex parser for `ACTION: <probe_name>\|THROW` tail; for THROW, parses `THETA/V/DT`. Uses last-occurrence so the VLM may restate things mid-reasoning. Clamps mildly out-of-range floats; raises on wild values. |
| `tossing/vlm/loop.py` | `run_episode(env, target_distance, max_probes, client) → EpisodeResult`. One retry on parse failure, then abort. Over-budget probe requests are ignored and the next turn forces THROW. |
| `scripts/run_vlm_episode.py` | Single-object smoke test with verbose reasoning dump. |
| `scripts/eval_vlm_only.py` | Sweeps probe budgets over the catalog; writes `results.json` + `summary.json`; optional `--include-oracle` runs CEM-on-true-physics for the ceiling baseline. |
| `tests/test_vlm_loop.py` | 15 tests over parser edge cases, budget enforcement, retry-then-abort. No real API calls. |

### 0.4 VLM output contract

The VLM is instructed to reason freely, then end with exactly one of:

```
ACTION: vertical_toss   # or forward_toss / release_drop / wrist_flick / shake
```
or
```
ACTION: THROW
THETA: <float 20–80>    # degrees
V:     <float 1–8>      # m/s
DT:    <float -0.1–0.1> # seconds
```

Parser uses the last occurrence of each key, so intermediate mentions in the
chain-of-thought don't break parsing.

### 0.5 What is preserved vs. dropped from Sections 1–14

| Full plan (§1–§14) | `vlm_only` branch |
|--------------------|-------------------|
| Simulator + 5 probes + object catalog | **Kept** (Phase 1 code reused unchanged) |
| Oracle CEM-on-true-physics | **Kept** as ceiling baseline (`--include-oracle`) |
| Belief update MLP (§4.2, §10 Phase 2) | **Dropped** — VLM reads raw probe numbers |
| EKF fallback (§4.2, §10 Phase 2) | Dropped |
| Phase 1 oracle throw MLP (§10 Phase 1) | Dropped — not needed without deployed MLP comparison |
| Phase 3 deployed throw MLP (§10 Phase 3) | **Dropped** — VLM emits `(θ, v, Δt)` directly |
| Memory retrieval (§7.2) | Dropped — every object is novel to the VLM |
| Probe-cost model (probe=1, throw=5, B=8) | Dropped — `max_probes` is a hard cap |
| Distance sweep (§5.1) | Deferred — first eval uses a single distance (2.0 m) |
| Baselines (§8) | Deferred — the `budget=0` (zero-shot VLM) run serves as the initial baseline; random / fixed / blind-MLP are future |

### 0.6 First experiment on this branch

```bash
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY for --provider openai

# Smoke test (single object, verbose output)
MUJOCO_GL=egl PYTHONPATH=. python scripts/run_vlm_episode.py \
    --object dense_compacts_000 --distance 2.0 --max-probes 3 --provider claude

# Mini-eval: 10 objects × {0-probe zero-shot, 3-probe} with CEM ceiling
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_vlm_only.py \
    --catalog catalog.json --budgets 0,3 --n-objects 10 \
    --provider claude --include-oracle \
    --output outputs/vlm_only/first_eval
```

**Success criterion for the branch** (not for the paper): on the mini-eval, the
budget=3 success rate is measurably higher than budget=0 on the same catalog
subset. If yes, move to the full catalog and then graduate to the learned
pipeline in Sections 10 Phase 2–3. If no, diagnose before doing any training.

### 0.7 Known limitations of Level 3 (accepted on this branch)

- **VLMs are weak at numerical regression** (§12). Throw-param emission will
  be noisy; absolute success rates will likely be well below the CEM ceiling
  even with good probe selection. This is tolerated because the **slope**
  (success rate vs. budget) is what tests the thesis, not the intercept.
- **No σ calibration.** The loop has no notion of uncertainty; the VLM
  either probes or throws, with no robust-throw behavior under high uncertainty.
- **No learned observation-noise model.** Probes that return near-identical
  observations for different physics (ambiguous objects) may mislead the VLM.

These are the exact reasons the full plan includes the learned modules; they
are not bugs in Level 3, they are reasons to eventually move past it.

---

## 1. One-Sentence Thesis

A VLM-guided active probing strategy enables a manipulation agent to adapt dynamic throwing to novel objects using fewer interactions than zero-shot prediction, random probing, or blind policy adaptation.

## 2. Problem Statement

Foundation VLMs can reason about scenes and objects but cannot adapt their physical predictions to novel objects whose dynamics differ from training data. Dynamic manipulation tasks like tossing expose this gap sharply: small errors in estimated mass, drag, or center-of-mass produce large trajectory errors.

We propose a system where a VLM acts as an **experiment designer** — selecting a sequence of cheap diagnostic actions ("probes") on a novel object to efficiently reduce uncertainty about its hidden physical properties before committing to an expensive goal-directed throw.

## 3. Scope and Non-Goals

### In Scope
- 2D sagittal-plane tossing in MuJoCo simulation
- VLM-driven probe selection from a discrete menu of probe types
- Structured simulator feedback (no VLM-based trajectory interpretation)
- Systematic benchmark with object families varying in hidden physics
- Comparison against 4 baselines (see Section 8)

### Out of Scope (Future Work)
- Real robot experiments (sim-to-real transfer)
- 3D manipulation or dexterous grasping
- VLM-based visual trajectory interpretation
- Multi-step manipulation beyond single throws
- Learning the probe action primitives themselves

## 4. System Architecture

### 4.1 Two-Actor Design

**Actor A — Experiment Designer (VLM + Policy Head)**

| Input | Description |
|-------|-------------|
| Rendered image of novel object | Top-down + side view from simulator |
| Object memory bank | Structured logs of prior objects: visual features, true/estimated physics, probe outcomes, throw results |
| Current belief state | μ and σ over hidden properties: mass, drag coeff, CoM offset, moment of inertia |
| Probe budget remaining | Integer countdown |
| Target basket location | (x, y) in world frame |

| Output | Description |
|--------|-------------|
| Next action | Either a probe type index (from discrete menu) OR "commit to throw" |
| Throw parameters (if committing) | Release angle θ, release speed v, release timing offset Δt |

**Implementation:** The VLM (e.g., GPT-4V, Gemini, or Claude with vision) receives a structured prompt containing the image, belief state summary, memory retrieval, and probe budget. It outputs a chain-of-thought reasoning trace and a structured action decision. When committing to throw, a small learned MLP maps the current belief state → throw parameters (the VLM selects *when* to throw, the MLP selects *how*).

Critically, **the VLM's only output is a discrete action symbol** from `{vertical_toss, forward_toss, release_drop, wrist_flick, shake, THROW}`. It never produces (μ, σ) values, throw parameters, or any continuous quantity directly. All numerical estimates flow through the downstream modules (belief update network + throw MLP), which play to each module's strengths: the VLM does semantic reasoning over *which experiment to run*, and small learned networks do precise numerical regression. This separation is motivated by VLMs' known weakness at precise quantitative prediction (see §12).

**Throw MLP I/O (used when the VLM emits `THROW`):**

| Stage | Input dim | Components | Output dim |
|-------|-----------|------------|------------|
| Phase 1 "oracle" MLP (ceiling baseline) | 6 | [m, c_d, Δ_com_x, Δ_com_z, I, distance] — ground-truth physics | 3 — (θ, v, Δt) |
| Phase 3 "deployed" MLP (belief-conditioned) | 11 | [μ (5-dim), σ (5-dim), distance] — estimated physics + uncertainty | 3 — (θ, v, Δt) |

The oracle MLP in Phase 1 is a *different model* from the deployed throw MLP in Phase 3. The Phase 1 model exists solely as (a) a sanity check that the simulator + throw parameterization is well-posed (if >90% success with perfect physics is unreachable, no probing strategy can save the system), and (b) the Oracle baseline in §8. The Phase 3 model has wider input, incorporates uncertainty (σ), and is trained on belief-state trajectories rather than ground-truth physics (see §10 Phase 3).

**Actor B — Experiment Executor (MuJoCo Simulator)**

Executes the chosen probe or throw action and returns structured numerical observations:

| Probe Type | Returned Observations |
|------------|----------------------|
| Vertical micro-toss | Apex height, hang time, landing offset from release point |
| Short forward toss | Landing distance, flight time, lateral drift |
| Gentle release-drop | Fall time to ground, bounce behavior (coefficient of restitution) |
| Wrist flick | Angular velocity response, precession observed (bool) |
| Small shake | Oscillation frequency, damping ratio, perceived resistance |

For throw attempts: full trajectory (sampled at 30 Hz), binary success/failure, distance-to-basket-center.

### 4.2 Belief State Representation

A Gaussian belief over 4 hidden properties:

| Property | Symbol | Role | Unit |
|----------|--------|------|------|
| Mass | m | Governs ballistic arc | kg |
| Drag coefficient | c_d | Governs air resistance deceleration | dimensionless |
| Center-of-mass offset | Δ_com | Governs spin-induced drift | m (from geometric center) |
| Moment of inertia | I | Governs rotational dynamics | kg·m² |

**Flattened physics vector.** CoM offset is 2D in the sagittal plane (x and z components), so the conceptually 4-property vector becomes a **5-dim scalar vector** when flattened for MLP I/O:

```
p = [m, c_d, Δ_com_x, Δ_com_z, I]
```

Three same-shape versions of this vector appear throughout the project:

| Name | Meaning | Where it lives |
|------|---------|----------------|
| **p_true** | The physics MuJoCo actually simulates with | Simulator ground truth; available in training only |
| **μ** | Bayesian point estimate of p | Output of the belief update network — the system's best guess given probe observations |
| **σ** | Per-component uncertainty in the estimate | Also output of the belief update network — calibrated by training |

At deployment the system never sees `p_true`; it only constructs `(μ, σ)` from probe observations and feeds those to the throw MLP. The core empirical question of the paper — *does probing help?* — reduces to: *how close does μ get to p_true, as a function of how smartly the VLM picks probes?*

**Update mechanism (default: learned MLP).** After each probe, a small learned update network (2-layer MLP) maps

```
[μ_prior (5), σ_prior (5), probe_type_onehot (5), probe_observations (padded to max dim)]
  → [μ_post (5), σ_post (5)]
```

Probe observations are different dimension per probe (e.g., `vertical_toss` returns 3 numbers, `wrist_flick` returns 2 + a bool). We pad to a fixed vector whose entries are zero for non-applicable probes; the probe-type one-hot lets the network gate on which entries are valid. Training loss is Gaussian NLL so that σ calibrates rather than collapsing to zero:

```
L = 0.5 * [log(σ_post²) + (μ_post - p_true)² / σ_post²]   (per component, then summed)
```

Training data (Phase 2): for every object in the 100-object catalog, run every probe in simulation; record `(probe_type, observations, p_true)`. Synthesize training examples by sampling random priors `(μ_prior, σ_prior)` that straddle `p_true`, passing them through the network, and supervising toward `p_true` with NLL. Success criterion: after 3 probes, belief RMSE < 20% of prior RMSE (§10 Phase 2).

The VLM does *not* directly output (μ, σ) — it only picks the probe. The update network does the quantitative Bayesian-like inference.

**Classical alternatives (recommended as baselines or fallback).** A learned update net is not the only reasonable choice here — and for this problem is arguably not even the *best* choice. Three classical alternatives work cleanly:

1. **Per-probe analytical inversion.** Because each probe is deliberately designed to isolate one physics property, the forward model for several probes has a clean closed form. E.g., `vertical_toss`'s hang time relates `c_d / m` directly; `wrist_flick`'s angular deceleration gives `I` directly (no mass coupling). Combining `vertical_toss` + `release_drop` disambiguates `m` from `c_d`. No training required.
2. **Extended / Unscented Kalman Filter.** Standard Bayesian nonlinear state estimation: write `observation = h(physics) + noise`, linearize around μ, apply Kalman gain. EKF produces calibrated σ by construction and handles cross-coupled observations naturally.
3. **Particle filter / simulation-based inference.** Sample N physics vectors from the prior, replay the probe through MuJoCo for each, weight by observation likelihood, return the weighted mean/std as (μ, σ). Asymptotically exact; expensive (≈1000 sims per update) but no training.

The arguments *for* a learned net are: implicit observation-noise model (no hand-tuned covariances), better handling of nonlinear coupling at wide priors, fast inference (one forward pass vs. many simulator calls), and uniform "learned pipeline" narrative. The arguments *against* are: training complexity, risk of miscalibrated σ, and more failure modes than a classical filter.

**Our plan:** implement a learned update MLP as the primary method, but keep an EKF implementation as a drop-in baseline. If Phase 2 validation (belief RMSE reduction) fails or calibration is unreliable, we can swap to EKF without touching the rest of the pipeline. The paper's contribution is **VLM-driven probe selection**, not belief updating — making this component swappable keeps the contribution clean and the result less sensitive to implementation choices.

### 4.3 Information Flow

```
┌─────────────────────────────────────────────────┐
│                  Per-Episode Loop                │
│                                                  │
│  1. Render novel object → image                  │
│  2. Initialize belief prior from VLM visual      │
│     estimate + memory of similar objects          │
│  3. While probe_budget > 0:                      │
│     a. VLM receives: image, belief, memory,      │
│        budget, target                            │
│     b. VLM reasons + selects: probe_type OR      │
│        "commit to throw"                         │
│     c. If probe: executor runs it, returns       │
│        structured obs → update network refines   │
│        belief                                    │
│     d. If commit: MLP maps belief → throw params │
│        → executor runs throw → evaluate          │
│  4. If budget exhausted without commit:           │
│     MLP produces best-guess throw from final     │
│     belief                                       │
└─────────────────────────────────────────────────┘
```

## 5. Simulation Environment

### 5.1 Setup

- **Engine:** MuJoCo 3.x
- **Scene:** 2D sagittal plane (constrain to xz-plane via joint limits)
- **Gripper:** Point-mass launcher on horizontal rail at fixed height. Parameterized by release angle θ ∈ [20°, 80°], release speed v ∈ [1, 8] m/s, release timing offset Δt ∈ [-0.1, 0.1] s
- **Basket:** Cylindrical container at variable distance d ∈ [1.0, 3.0] m, fixed height
- **Physics:** Gravity + custom drag force applied at each timestep: F_drag = -½ ρ c_d A v|v|
- **Rendering:** MuJoCo built-in renderer for images passed to VLM (top-down + side views)

### 5.2 Probe Action Primitives

Each probe is a hard-coded controller that executes a specific diagnostic motion:

| Name | Probe | Controller Description | Diagnostic Purpose |
|------|-------|----------------------|-------------------|
| `vertical_toss` | Vertical micro-toss | Launch straight up at 2 m/s | Mass (hang time), drag (apex delta from ballistic prediction) |
| `forward_toss` | Short forward toss | Launch at 45° at 3 m/s | Drag (range shortfall), CoM offset (lateral drift) |
| `release_drop` | Gentle release-drop | Open gripper, let fall | Mass (fall time if drag present), restitution |
| `wrist_flick` | Wrist flick | Impart angular velocity only | Moment of inertia (angular deceleration rate) |
| `shake` | Small shake | Oscillate gripper ±5cm at 4 Hz | Inertia response, internal mass distribution |

**Cost model:** Each probe costs 1 unit. A throw attempt costs 5 units. Total budget per episode: B (default B = 8, allowing e.g., 3 probes + 1 throw, or 5 probes + best-guess throw at budget exhaustion).

## 6. Object Benchmark

### 6.1 Training Families (Known at Train Time)

Each family varies one dominant hidden property while keeping others moderate:

| Family | Examples | Dominant Variation | Visual Cue Reliability |
|--------|----------|-------------------|----------------------|
| Dense compacts | Steel ball, rock, clay ball | Mass (high), low drag | High — looks heavy |
| Light elongated | Pencil, straw, chopstick | Low mass, high CoM offset | Medium — length visible, mass ambiguous |
| Draggy flats | Paper plate, cardboard square, leaf | High drag coefficient | Medium — flatness visible, drag magnitude unclear |
| Asymmetric | Hammer, wrench, ladle | CoM offset, high I | Medium — shape visible, internal distribution unclear |
| Uniform moderate | Tennis ball, orange, stress ball | Baseline (moderate everything) | High |

### 6.2 Test Objects (Held-Out, Seen Only at Evaluation)

The key evaluation objects include **appearance–physics mismatches** where visual estimation alone fails:

| Object | Deception Type | Why Probing Helps |
|--------|---------------|-------------------|
| Hollow metal sphere | Looks heavy, actually light | Micro-toss reveals unexpectedly high apex |
| Lead-filled plastic bottle | Looks light, actually heavy | Drop or micro-toss reveals fast descent |
| Foam airplane shape | Looks rigid, extreme drag | Forward toss reveals massive range shortfall |
| Weighted dart (tail-heavy) | Symmetric appearance, asymmetric CoM | Flick reveals unexpected precession |
| Crumpled aluminum foil ball | Ambiguous everything | Multiple probes needed to disambiguate |

### 6.3 Object Parameterization in MuJoCo

Each object is defined by a tuple: (mesh, m, c_d, Δ_com, I, visual_texture). The mesh and texture control what the VLM sees. The physics parameters control simulation. This decoupling enables appearance–physics mismatches.

## 7. VLM Prompt Design

### 7.1 Prompt Structure (Per Decision Step)

```
System: You are an experiment designer for a robot tossing system.
You must decide what diagnostic probe to run next on a novel object,
or commit to a final throw.

[Image: side view and top view of the object on the gripper]

OBJECT MEMORY (3 most similar prior objects retrieved by visual embedding):
- Object: steel_ball_v2 | mass: 0.45kg | drag: 0.1 | Probes used: vertical_toss, release_drop
  | vertical_toss result: apex 0.21m, hang_time 0.41s | Throw: success at d=2.0m
  with θ=52°, v=5.1
- [...]

CURRENT BELIEF:
  mass: μ=0.30kg, σ=0.15kg
  drag_coeff: μ=0.3, σ=0.25
  com_offset: μ=0.00m, σ=0.02m
  inertia: μ=0.001, σ=0.0008

PROBES COMPLETED THIS EPISODE: [vertical_toss → apex: 0.45m, hang_time: 0.60s]
PROBES REMAINING: 2
TARGET: basket at x=2.0m

Reason step by step about what is still uncertain and which probe
would most reduce uncertainty. Then output exactly one of:
ACTION: vertical_toss | forward_toss | release_drop | wrist_flick | shake | THROW
```

### 7.2 Memory Retrieval

Prior object experiences are stored as structured records. At test time, retrieve top-k most similar objects by CLIP embedding cosine similarity on rendered images. This gives the VLM concrete analogies to reason over.

## 8. Baselines and Ablations

| Label | Description | What It Tests |
|-------|-------------|---------------|
| **Zero-shot VLM** | VLM sees image, no probes allowed, directly predicts throw params | Does probing help at all? |
| **Random probing** | Probes selected uniformly at random, same budget | Does *intelligent* probe selection matter? |
| **Blind MLP** | No VLM, no image. MLP sees only probe outcomes, selects next probe by learned policy | Does VLM visual reasoning add value? |
| **Oracle** | Ground-truth physics given to throw MLP | Performance ceiling |
| **No memory** | Full system but without prior object memory retrieval | Does experience transfer matter? |
| **Fixed probe sequence** | Always run vertical_toss → forward_toss → wrist_flick then throw | Does adaptive sequencing matter vs. a good fixed protocol? |

## 9. Metrics

**Primary:**
- **Success rate** at throw (binary: landed in basket) as a function of probe budget (0, 1, 2, ..., 5)
- **Probe efficiency curve**: success rate vs. number of probes used (area under this curve is the headline number)

**Secondary:**
- Belief accuracy: RMSE of estimated physics vs. ground truth after k probes
- Probe diversity: entropy of probe type distribution (does the VLM use all probe types or collapse to one?)
- Reasoning quality: manual inspection of VLM chain-of-thought on 50 test cases

**Breakdown:**
- Per object family (where does probing help most?)
- Appearance–physics mismatch vs. consistent objects (where should the gap be largest?)

## 10. Implementation Plan

### Phase 1: Simulator + Probe Primitives (Weeks 1–3)
- [ ] MuJoCo 2D tossing environment with parameterized objects
- [ ] 5 probe controllers with structured observation extraction
- [ ] Object generation pipeline (mesh + physics parameter sampling): 100 objects = 5 families × 20 samples (see §6.1)
- [ ] Rendering pipeline for VLM input images
- [ ] Sanity check: oracle throw MLP trained on ground-truth physics achieves >90% success

**Phase 1 oracle MLP — detailed recipe.** This model is the ceiling baseline and the Phase 1 sanity check; it is *not* the deployed throw policy.

- **Architecture:** 3-layer MLP, hidden dim 256, ReLU + BatchNorm, Sigmoid output head mapped to physical ranges.
- **Input (6-dim):** `[m, c_d, Δ_com_x, Δ_com_z, I, basket_distance]` — normalized by per-feature mean/std from the training set.
- **Output (3-dim):** `[θ, v, Δt]` in normalized [0,1], denormalized to physical ranges θ ∈ [20°, 80°], v ∈ [1, 8] m/s, Δt ∈ [−0.1, 0.1] s.
- **Training data generation (oracle dataset):**
    1. For each of the 100 catalog objects, and each of 20 basket distances spanning [1.0, 3.0] m,
    2. Run CEM (cross-entropy method) in MuJoCo: sample ≈300 candidate `(θ, v, Δt)` triples, score each by negative distance-to-basket, refit a Gaussian to the top quantile, repeat for 5 iterations.
    3. Record the CEM-winner throw and whether it landed in the basket.
- **Dataset size:** ≈2000 rows of `(physics, distance) → optimal throw + binary success`.
- **Training:** MSE regression in normalized output space.
- **Success criterion:** ≥90% basket-landing rate on held-out object–distance pairs.

### Phase 2: Belief Update Network (Weeks 3–4)
- [ ] Generate training data: run all probes on all training objects, record (probe_type, observation, true_physics) tuples
- [ ] Train belief update MLP: given prior belief + probe observation → posterior belief
- [ ] Implement EKF fallback as a drop-in baseline with the same I/O contract
- [ ] Validate: after 3 probes, belief RMSE should be <20% of prior RMSE; σ calibration check (empirical coverage of ±σ bands ≈ 68%)

**Detailed recipe — learned update network.**
- **I/O (see §4.2):** `(μ_prior (5), σ_prior (5), probe_type_onehot (5), observations_padded) → (μ_post (5), σ_post (5))`.
- **Observation padding:** per-probe observation vectors have different native dimensions (`vertical_toss`: 3 scalars; `forward_toss`: 3; `release_drop`: 2; `wrist_flick`: 2 scalars + 1 bool → 3; `shake`: 3). Concatenate all probe observation slots into a single fixed-width vector with zeros in non-applicable slots; the network reads the probe-type one-hot to know which slots to trust.
- **Training tuple synthesis:** for each (object, probe) pair in the raw data,
    1. Sample a random prior `σ_prior ∈ [σ_min, σ_max]` per component (spanning what a loose visual prior would produce).
    2. Sample `μ_prior ~ N(p_true, σ_prior²)` — a prior that straddles the truth.
    3. Target is `p_true` itself; the network must learn both to *shift* μ toward truth and to *shrink* σ proportional to the information the probe provided.
- **Loss:** Gaussian NLL per component, summed: `L = 0.5 * Σ_i [log(σ_post,i²) + (μ_post,i - p_true,i)² / σ_post,i²]`. MSE-only training collapses σ to zero; NLL is what makes σ calibrate.
- **Architecture:** 2-layer MLP (per §4.2), hidden dim 128. σ output passes through softplus to stay positive.

**Detailed recipe — EKF fallback.**
- Implement each probe's forward model `h_p(physics)` as either closed-form (`vertical_toss`, `release_drop`, `wrist_flick` are analytically tractable) or as a single MuJoCo rollout (`forward_toss`, `shake`).
- Linearize `h_p` by finite differences around μ_prior.
- Apply Kalman gain with a hand-tuned observation covariance `R_p` per probe (estimated from simulator noise or set as a fraction of each observation's magnitude).
- Same (μ, σ) I/O as the learned network so downstream components are unchanged.

### Phase 3: Throw Policy MLP (Weeks 4–5)
- [ ] Train MLP: belief state + target location → throw parameters (θ, v, Δt)
- [ ] Training data: generate `(μ, σ, distance) → optimal_throw` tuples via one of the three strategies below
- [ ] Validate: with ground-truth physics input (degenerate σ→0), success rate >90% (matches Phase 1 oracle); with realistic post-probe (μ, σ), success rate on training objects is the primary Phase 3 metric

**This is a new model, distinct from the Phase 1 oracle MLP.** Input dim grows from 6 → 11 because σ is now part of the input, and the training data distribution changes from "true physics" to "realistic belief states." The oracle MLP's weights are not reused. What persists is the **CEM → supervised regression recipe** for generating throw-parameter targets.

**Why σ in the input matters.** A belief-conditioned throw MLP should throw **risk-aversely** when σ is large: prefer a `(θ, v)` whose success is robust across the uncertain physics range, even if it's suboptimal at μ. A narrow optimum that only works at exactly μ = 0.5 kg is worse than a wider throw that works across μ ∈ [0.3, 0.7] kg. This risk-awareness only emerges if σ appears in both input *and* training targets (Strategy C below).

**Three data-generation strategies, in increasing fidelity and cost.**

**Strategy A — Synthetic noise injection (cheapest).** Reuse the Phase 1 oracle dataset directly. For each `(p_true, distance, CEM_throw)` row:
1. Sample σ from a reasonable range per component.
2. Sample `μ = p_true + ε`, ε ~ N(0, σ²).
3. Training row: `(μ, σ, distance) → CEM_throw`.

Pro: trivial to implement, zero new simulator cost.
Con: Gaussian isotropic noise doesn't match the structured, correlated errors the belief net actually produces at deployment (e.g., after `vertical_toss` mass is accurate but drag is still fuzzy). Target is still the *single-physics* optimum — the MLP has no incentive to be risk-aware, since σ doesn't change the target.

**Strategy B — Belief-net rollouts (faithful input distribution).** Only possible once Phase 2 is done.
1. For each training object, run many probe sequences — varying probe choices, orders, and counts (k = 0, 1, 2, 3 probes).
2. At each step, record the belief net's output `(μ, σ)`.
3. Pair every rolled-out `(μ, σ, distance)` with the CEM-optimal throw for *that object* at *that distance* (from the oracle dataset).
4. Training row: `(μ, σ, distance) → CEM_throw_for_p_true`.

Pro: input distribution matches deployment exactly; MLP sees the specific correlated error patterns Phase 2's belief net actually produces.
Con: coupled to Phase 2; if the belief net changes, the throw MLP must be retrained.

**Strategy C — Robust-CEM targets (risk-aware by construction).** Make the target itself depend on σ.
1. Sample `(μ, σ, distance)` either synthetically or from belief-net rollouts.
2. Re-run CEM with objective `E_{p ~ N(μ, diag(σ²))}[success(throw, p)]` — for each candidate throw, simulate ~20 physics samples from the belief and score by the fraction landing in the basket.
3. The CEM-winner is the **robust** throw for that belief.
4. Training row: `(μ, σ, distance) → robust_throw`.

Pro: σ in the input is now supervised by a σ-dependent target, so the MLP learns genuine risk-awareness (wider σ → wider/safer throws).
Con: ≈20× more simulator calls per training row; must be built fresh (cannot reuse Phase 1 oracle dataset).

**Recommended plan.** Start with Strategy A for a first working pipeline; swap to **B + C combined** (faithful inputs, robust targets) for the paper numbers. Report both to isolate how much of the Phase 3 gain comes from "belief-aware inputs" vs. "risk-aware targets."

### Phase 4: VLM Integration (Weeks 5–7)
- [ ] Implement prompt template and memory retrieval (CLIP-based)
- [ ] Build VLM decision loop (parse structured output, handle failures)
- [ ] Test with GPT-4o / Claude as VLM backbone (compare both)
- [ ] Iterate on prompt based on failure analysis

### Phase 5: Baselines + Evaluation (Weeks 7–9)
- [ ] Implement all 6 baselines
- [ ] Run full evaluation on test object set (50 objects × 5 seeds × all methods)
- [ ] Generate plots: probe efficiency curves, per-family breakdowns, belief accuracy trajectories
- [ ] Manual chain-of-thought analysis on 50 cases

### Phase 6: Paper Writing (Weeks 9–12)
- [ ] Draft figures first (system diagram, probe efficiency curves, qualitative examples)
- [ ] Write results → method → intro → related work
- [ ] Internal review and revision

## 11. Key Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| VLM probe selection is no better than random | Medium | High — kills the thesis | Start with a well-designed fixed sequence baseline. If VLM only matches it, pivot the contribution to "active probing helps" rather than "VLM-guided probing helps" |
| VLM output parsing is unreliable | Medium | Medium | Constrained output format, retry logic, fallback to most-informative-by-heuristic |
| Belief update network doesn't generalize to novel objects | Low | High | Use a simple Bayesian filter (EKF) as fallback; don't over-parameterize the update net |
| 2D simplification is seen as too simple by reviewers | Medium | Medium | Frame as "controlled scientific study"; show the approach is modular and the findings transfer to 3D conceptually. Include 1–2 3D MuJoCo demos if time permits |
| API costs for VLM calls during large-scale eval | Low | Low | Cache VLM responses for identical (image, belief, memory) inputs; use cheaper model for sweeps, expensive model for final numbers |

## 12. Design Decisions and Rationale

**Why structured simulator feedback (not VLM trajectory interpretation)?**  
Decoupling perception from reasoning isolates the research question. If VLM trajectory interpretation is noisy, we can't tell whether failures come from bad probe selection or bad observation parsing. This can be added as an ablation later.

**Why a discrete probe menu (not free-form VLM-designed probes)?**  
Free-form probe design requires solving the inverse problem of mapping natural language to robot actions — a full research problem on its own. A discrete menu keeps the action space tractable while still giving the VLM meaningful choices.

**Why 2D?**  
Reduces the physics parameter space (no lateral dynamics), the action space (3 params instead of 6+), and simulation cost. The conceptual contribution — active experiment design via VLM — is invariant to dimensionality.

**Why a separate throw MLP instead of VLM-predicted throw params?**  
VLMs are poor at precise numerical regression. The MLP converts a belief state (which the VLM helped construct through probe selection) into throw parameters. This plays to each module's strengths. Concretely, the VLM's output alphabet is only `{vertical_toss, forward_toss, release_drop, wrist_flick, shake, THROW}` — six tokens — and all continuous values (μ, σ, θ, v, Δt) are produced by small MLPs downstream.

**Why a learned belief update net instead of a classical Bayesian filter?**
This is a deliberate — and debatable — choice. A classical filter (EKF, UKF, particle filter) or even per-probe analytical inversion would work for this problem: the physics is deterministic, the simulator is accessible, and each probe is designed to isolate a specific property. A learned update net is appealing mainly because (a) it implicitly learns the simulator's observation-noise model without hand-tuned covariances, (b) it naturally handles nonlinear cross-coupling in observations, and (c) it fits the "learned pipeline" narrative. But it adds training complexity and risks miscalibrated σ. Our plan (§4.2, §10 Phase 2) is to implement the learned net as the primary method but keep an EKF as a drop-in fallback — the paper's contribution is **probe selection**, not inference, so making the update component swappable keeps the result robust to this choice.

**Why two different throw MLPs (Phase 1 oracle vs. Phase 3 deployed) instead of reusing one?**
They answer different questions and have different input contracts. The Phase 1 oracle MLP takes `(p_true, distance)` with no uncertainty — it's the ceiling baseline and sanity check. The Phase 3 deployed MLP takes `(μ, σ, distance)` — it must be risk-aware when σ is wide. Training on true physics alone produces a model that is overconfident on noisy deployment inputs; training with σ-dependent targets (Strategy C in §10 Phase 3) is what teaches risk-aversion. These are incompatible objectives for a single model, so they live as two models.

## 13. Open Questions

1. **How many training object families are needed?** Start with 5 families × 20 objects each (100 total). May need more if belief update network overfits.
2. **Should the VLM see probe outcome numbers or natural language descriptions?** Try both — numbers are more precise, but NL may help the VLM reason better.
3. **Is CLIP the right retrieval embedding?** It captures visual similarity but not physics similarity. Consider a learned embedding that jointly encodes appearance + discovered physics from probes.
4. **What if the VLM always picks the same probe sequence regardless of the object?** This would suggest the VLM isn't truly reasoning. Track probe-type entropy per object family as a diagnostic.

## 14. Related Work Positioning

This project sits at the intersection of:

- **Foundation models for robotics** (SayCan, VoxPoser, RT-2): We extend these by adding active physical experimentation rather than zero-shot execution.
- **System identification in manipulation** (Bauza et al., contact-rich ID): We replace hand-designed ID protocols with VLM-designed ones.
- **Active learning / Bayesian optimization for robot skills** (Cully et al. MAP-Elites, REPS): We add VLM-based reasoning over *which* experiment to run, rather than purely statistical acquisition functions.
- **Few-shot adaptation in RL** (MAML, RL²): We use structured probing rather than gradient-based meta-learning for adaptation.

The novel contribution is the **VLM as experiment designer** — using a foundation model's broad world knowledge to select physically informative diagnostic actions, bridging the gap between visual understanding and physical interaction.