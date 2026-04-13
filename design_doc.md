# Design Doc: VLM-Guided Active Probing for Adaptive Dynamic Manipulation

**Status:** Draft  
**Author:** [Your Name]  
**Date:** April 2026  

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

**Update mechanism:** After each probe, a small learned update network (2-layer MLP) maps [current belief μ, σ, probe_type_onehot, observed_outcomes] → [updated μ, σ]. This is trained on the training object set. The VLM does *not* directly output numbers — it reasons about *which probe to run*, and the update network handles the quantitative Bayesian-like update.

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

| ID | Probe | Controller Description | Diagnostic Purpose |
|----|-------|----------------------|-------------------|
| P1 | Vertical micro-toss | Launch straight up at 2 m/s | Mass (hang time), drag (apex delta from ballistic prediction) |
| P2 | Short forward toss | Launch at 45° at 3 m/s | Drag (range shortfall), CoM offset (lateral drift) |
| P3 | Gentle release-drop | Open gripper, let fall | Mass (fall time if drag present), restitution |
| P4 | Wrist flick | Impart angular velocity only | Moment of inertia (angular deceleration rate) |
| P5 | Small shake | Oscillate gripper ±5cm at 4 Hz | Inertia response, internal mass distribution |

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
- Object: steel_ball_v2 | mass: 0.45kg | drag: 0.1 | Probes used: P1, P3
  | P1 result: apex 0.21m, hang_time 0.41s | Throw: success at d=2.0m
  with θ=52°, v=5.1
- [...]

CURRENT BELIEF:
  mass: μ=0.30kg, σ=0.15kg
  drag_coeff: μ=0.3, σ=0.25
  com_offset: μ=0.00m, σ=0.02m
  inertia: μ=0.001, σ=0.0008

PROBES COMPLETED THIS EPISODE: [P1 → apex: 0.45m, hang_time: 0.60s]
PROBES REMAINING: 2
TARGET: basket at x=2.0m

Reason step by step about what is still uncertain and which probe
would most reduce uncertainty. Then output exactly one of:
ACTION: P1 | P2 | P3 | P4 | P5 | THROW
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
| **Fixed probe sequence** | Always run P1→P2→P4 then throw | Does adaptive sequencing matter vs. a good fixed protocol? |

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
- [ ] Object generation pipeline (mesh + physics parameter sampling)
- [ ] Rendering pipeline for VLM input images
- [ ] Sanity check: oracle throw MLP trained on ground-truth physics achieves >90% success

### Phase 2: Belief Update Network (Weeks 3–4)
- [ ] Generate training data: run all probes on all training objects, record (probe_type, observation, true_physics) tuples
- [ ] Train belief update MLP: given prior belief + probe observation → posterior belief
- [ ] Validate: after 3 probes, belief RMSE should be <20% of prior RMSE

### Phase 3: Throw Policy MLP (Weeks 4–5)
- [ ] Train MLP: belief state + target location → throw parameters (θ, v, Δt)
- [ ] Training data: sample physics params, compute optimal throw analytically or via CEM, train as regression
- [ ] Validate: with ground-truth physics input, success rate >90%

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
VLMs are poor at precise numerical regression. The MLP converts a belief state (which the VLM helped construct through probe selection) into throw parameters. This plays to each module's strengths.

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