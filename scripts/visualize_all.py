#!/usr/bin/env python3
"""Visualize the tossing environment: objects, probes, CEM throws."""

import os
os.environ["MUJOCO_GL"] = "egl"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image

from tossing.types import ObjectSpec, ThrowParams
from tossing.env import TossEnv, LAUNCHER_HEIGHT, BASKET_RADIUS, BASKET_HEIGHT, BASKET_FLOOR_HEIGHT
from tossing.objects.catalog import generate_catalog, generate_test_objects
from tossing.objects.families import FAMILIES
from tossing.oracle.dataset import cem_optimal_throw

# Register probes
import tossing.probes.vertical_toss
import tossing.probes.forward_toss
import tossing.probes.release_drop
import tossing.probes.wrist_flick
import tossing.probes.shake
from tossing.probes import list_probes


OUT = Path("outputs/viz")
OUT.mkdir(parents=True, exist_ok=True)


def plot_trajectory(ax, traj, color="blue", label=None, alpha=0.8):
    """Plot a 2D trajectory (x vs z) on an axes."""
    ax.plot(traj[:, 0], traj[:, 2], color=color, linewidth=1.5, alpha=alpha, label=label)
    ax.scatter(traj[0, 0], traj[0, 2], color=color, s=40, zorder=5, marker="o")
    ax.scatter(traj[-1, 0], traj[-1, 2], color=color, s=40, zorder=5, marker="x")


def draw_scene(ax, basket_distance):
    """Draw ground, launcher, and basket on axes."""
    # Ground
    ax.axhline(y=0, color="0.4", linewidth=2)
    ax.fill_between([-0.5, basket_distance + 1], -0.1, 0, color="0.85")

    # Launcher
    ax.plot(0, LAUNCHER_HEIGHT, "s", color="0.3", markersize=10, zorder=10)
    ax.annotate("gripper", (0, LAUNCHER_HEIGHT), textcoords="offset points",
                xytext=(10, 5), fontsize=8, color="0.3")

    # Basket
    bx = basket_distance
    by = BASKET_FLOOR_HEIGHT
    rect = Rectangle((bx - BASKET_RADIUS, by), 2 * BASKET_RADIUS, 2 * BASKET_HEIGHT,
                      linewidth=2, edgecolor="saddlebrown", facecolor="wheat", alpha=0.7)
    ax.add_patch(rect)
    ax.annotate("basket", (bx, by + 2 * BASKET_HEIGHT), textcoords="offset points",
                xytext=(-15, 5), fontsize=8, color="saddlebrown")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_xlim(-0.5, basket_distance + 1.0)
    ax.set_ylim(-0.15, LAUNCHER_HEIGHT + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------
# 1. Render objects from each family
# ---------------------------------------------------------------
def viz_object_gallery():
    print("1. Rendering object gallery...")
    catalog = generate_catalog()
    test_objs = generate_test_objects()

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Object Families — Side Views", fontsize=16)

    family_names = sorted(FAMILIES.keys())
    for col, fam in enumerate(family_names):
        objs = [o for o in catalog if o.family == fam]
        obj = objs[0]

        env = TossEnv(obj, basket_distance=2.0)
        imgs = env.render(width=320, height=240)

        axes[0, col].imshow(imgs["side"])
        axes[0, col].set_title(f"{fam}\nm={obj.mass:.2f}kg, cd={obj.drag_coeff:.2f}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(imgs["top"])
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Side view", fontsize=11)
    axes[1, 0].set_ylabel("Top view", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUT / "01_object_gallery.png", dpi=150)
    plt.close()
    print(f"  Saved {OUT / '01_object_gallery.png'}")

    # Test objects (mismatches)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Test Objects (Appearance–Physics Mismatches)", fontsize=14)
    for i, obj in enumerate(test_objs):
        env = TossEnv(obj, basket_distance=2.0)
        imgs = env.render(width=320, height=240)
        axes[i].imshow(imgs["side"])
        axes[i].set_title(f"{obj.name}\nm={obj.mass:.2f}kg, cd={obj.drag_coeff:.1f}", fontsize=8)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "01_test_objects.png", dpi=150)
    plt.close()
    print(f"  Saved {OUT / '01_test_objects.png'}")


# ---------------------------------------------------------------
# 2. Probe trajectories for one object
# ---------------------------------------------------------------
def viz_probe_trajectories():
    print("2. Rendering probe trajectories...")

    obj = ObjectSpec(
        name="demo_ball", family="uniform_moderate",
        geom_type="sphere", geom_size=[0.04],
        mass=0.25, drag_coeff=0.4, com_offset=[0.0, 0.0, 0.0],
        inertia=0.001, cross_section_area=0.01,
        rgba=[0.9, 0.3, 0.2, 1.0],
    )

    probe_names = {
        "vertical_toss": "Vertical Micro-Toss",
        "forward_toss": "Short Forward Toss",
        "release_drop": "Gentle Release-Drop",
        "wrist_flick": "Wrist Flick",
        "shake": "Small Shake",
    }
    colors = {
        "vertical_toss": "#e74c3c",
        "forward_toss": "#3498db",
        "release_drop": "#2ecc71",
        "wrist_flick": "#9b59b6",
        "shake": "#f39c12",
    }

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle(f"Probe Trajectories — {obj.name} (m={obj.mass}kg, cd={obj.drag_coeff})", fontsize=14)

    for idx, pt in enumerate(["vertical_toss", "forward_toss", "release_drop", "wrist_flick", "shake"]):
        ax = axes[idx]
        env = TossEnv(obj, basket_distance=2.0)
        result = env.run_probe(pt)
        traj = result.trajectory

        if traj is not None and len(traj) > 0:
            draw_scene(ax, basket_distance=2.0)
            plot_trajectory(ax, traj, color=colors[pt])

            # Annotate key observations
            obs_text = "\n".join(f"{k}: {v:.3f}" for k, v in result.observations.items())
            ax.text(0.02, 0.98, obs_text, transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(f"{pt}: {probe_names[pt]}", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT / "02_probe_trajectories.png", dpi=150)
    plt.close()
    print(f"  Saved {OUT / '02_probe_trajectories.png'}")


# ---------------------------------------------------------------
# 3. Probe comparison across object families
# ---------------------------------------------------------------
def viz_probe_sensitivity():
    print("3. Rendering probe sensitivity across families...")

    catalog = generate_catalog()
    families = sorted(FAMILIES.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Probe Sensitivity Across Object Families", fontsize=14)

    # vertical_toss apex by family
    ax = axes[0, 0]
    for fam in families:
        objs = [o for o in catalog if o.family == fam][:5]
        apexes = []
        for obj in objs:
            env = TossEnv(obj, basket_distance=2.0)
            r = env.run_probe("vertical_toss")
            apexes.append(r.observations["apex_height"])
        ax.bar(families.index(fam), np.mean(apexes), yerr=np.std(apexes) if len(apexes) > 1 else 0,
               capsize=4, color=f"C{families.index(fam)}", alpha=0.8)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=7)
    ax.set_ylabel("Apex height (m)")
    ax.set_title("vertical_toss — Apex")

    # forward_toss landing distance by family
    ax = axes[0, 1]
    for fam in families:
        objs = [o for o in catalog if o.family == fam][:5]
        dists = []
        for obj in objs:
            env = TossEnv(obj, basket_distance=2.0)
            r = env.run_probe("forward_toss")
            dists.append(r.observations["landing_distance"])
        ax.bar(families.index(fam), np.mean(dists), yerr=np.std(dists) if len(dists) > 1 else 0,
               capsize=4, color=f"C{families.index(fam)}", alpha=0.8)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=7)
    ax.set_ylabel("Landing distance (m)")
    ax.set_title("forward_toss — Range")

    # release_drop fall time by family
    ax = axes[0, 2]
    for fam in families:
        objs = [o for o in catalog if o.family == fam][:5]
        times = []
        for obj in objs:
            env = TossEnv(obj, basket_distance=2.0)
            r = env.run_probe("release_drop")
            times.append(r.observations["fall_time"])
        ax.bar(families.index(fam), np.mean(times), yerr=np.std(times) if len(times) > 1 else 0,
               capsize=4, color=f"C{families.index(fam)}", alpha=0.8)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=7)
    ax.set_ylabel("Fall time (s)")
    ax.set_title("release_drop — Fall Time")

    # wrist_flick angular velocity by family
    ax = axes[1, 0]
    for fam in families:
        objs = [o for o in catalog if o.family == fam][:5]
        omegas = []
        for obj in objs:
            env = TossEnv(obj, basket_distance=2.0)
            r = env.run_probe("wrist_flick")
            omegas.append(r.observations["angular_velocity_response"])
        ax.bar(families.index(fam), np.mean(omegas), yerr=np.std(omegas) if len(omegas) > 1 else 0,
               capsize=4, color=f"C{families.index(fam)}", alpha=0.8)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=7)
    ax.set_ylabel("Angular vel response (rad/s)")
    ax.set_title("wrist_flick — Angular Response")

    # shake perceived resistance by family
    ax = axes[1, 1]
    for fam in families:
        objs = [o for o in catalog if o.family == fam][:5]
        resists = []
        for obj in objs:
            env = TossEnv(obj, basket_distance=2.0)
            r = env.run_probe("shake")
            resists.append(r.observations["perceived_resistance"])
        ax.bar(families.index(fam), np.mean(resists), yerr=np.std(resists) if len(resists) > 1 else 0,
               capsize=4, color=f"C{families.index(fam)}", alpha=0.8)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=7)
    ax.set_ylabel("Perceived resistance")
    ax.set_title("shake — Resistance")

    # Hide last subplot
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "03_probe_sensitivity.png", dpi=150)
    plt.close()
    print(f"  Saved {OUT / '03_probe_sensitivity.png'}")


# ---------------------------------------------------------------
# 4. CEM throw visualization: successes and failures
# ---------------------------------------------------------------
def viz_cem_throws():
    print("4. Rendering CEM throw experiments...")

    obj = ObjectSpec(
        name="demo_ball", family="uniform_moderate",
        geom_type="sphere", geom_size=[0.04],
        mass=0.25, drag_coeff=0.4, com_offset=[0.0, 0.0, 0.0],
        inertia=0.001, cross_section_area=0.01,
        rgba=[0.9, 0.3, 0.2, 1.0],
    )
    basket_d = 2.0

    # Run many random throws and a few CEM-optimized ones
    rng = np.random.default_rng(42)
    env = TossEnv(obj, basket_distance=basket_d)

    # Random throws
    random_trajs = []
    random_success = []
    for _ in range(30):
        theta = rng.uniform(20, 80)
        v = rng.uniform(1, 8)
        result = env.throw(ThrowParams(theta=theta, v=v))
        if result.trajectory is not None and len(result.trajectory) > 1:
            random_trajs.append(result.trajectory)
            random_success.append(result.success)

    # CEM-optimized throw
    best_params, best_dist = cem_optimal_throw(obj, basket_d, n_samples=100, n_elite=10, n_iterations=3, seed=42)
    env2 = TossEnv(obj, basket_distance=basket_d)
    best_result = env2.throw(best_params)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: All random throws
    ax = axes[0]
    draw_scene(ax, basket_d)
    for traj, succ in zip(random_trajs, random_success):
        c = "#2ecc71" if succ else "#e74c3c"
        a = 0.7 if succ else 0.2
        ax.plot(traj[:, 0], traj[:, 2], color=c, linewidth=0.8, alpha=a)
    n_succ = sum(random_success)
    ax.set_title(f"30 Random Throws ({n_succ} success, {30-n_succ} miss)", fontsize=11)

    # Panel 2: CEM optimization result
    ax = axes[1]
    draw_scene(ax, basket_d)
    if best_result.trajectory is not None:
        plot_trajectory(ax, best_result.trajectory, color="#2ecc71", label="CEM optimal")
    ax.set_title(f"CEM-Optimized Throw\n\u03b8={best_params.theta:.1f}\u00b0, v={best_params.v:.2f} m/s\n"
                 f"{'SUCCESS' if best_result.success else 'MISS'} (dist={best_result.distance_to_basket:.3f}m)",
                 fontsize=11)
    ax.legend(fontsize=9)

    # Panel 3: Comparison — several distances
    ax = axes[2]
    draw_scene(ax, 3.5)  # wider view
    colors_dist = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    for i, d in enumerate([1.0, 1.5, 2.0, 2.5, 3.0]):
        params, _ = cem_optimal_throw(obj, d, n_samples=80, n_elite=8, n_iterations=3, seed=42+i)
        if params is not None:
            env3 = TossEnv(obj, basket_distance=d)
            res = env3.throw(params)
            if res.trajectory is not None and len(res.trajectory) > 1:
                plot_trajectory(ax, res.trajectory, color=colors_dist[i],
                               label=f"d={d}m (\u03b8={params.theta:.0f}\u00b0, v={params.v:.1f})")
        # Draw basket at this distance
        rect = Rectangle((d - BASKET_RADIUS, BASKET_FLOOR_HEIGHT), 2 * BASKET_RADIUS, 2 * BASKET_HEIGHT,
                          linewidth=1.5, edgecolor=colors_dist[i], facecolor="none", linestyle="--")
        ax.add_patch(rect)

    ax.set_title("CEM Throws at Different Distances", fontsize=11)
    ax.set_xlim(-0.5, 4.0)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT / "04_cem_throws.png", dpi=150)
    plt.close()
    print(f"  Saved {OUT / '04_cem_throws.png'}")


# ---------------------------------------------------------------
# 5. Effect of drag: same throw, different drag coefficients
# ---------------------------------------------------------------
def viz_drag_effect():
    print("5. Rendering drag effect comparison...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    draw_scene(ax, 3.0)

    drags = [0.0, 0.3, 1.0, 2.0, 3.0]
    colors = ["#2c3e50", "#2980b9", "#27ae60", "#f39c12", "#e74c3c"]

    for cd, col in zip(drags, colors):
        obj = ObjectSpec(
            name=f"drag_{cd}", family="test",
            geom_type="sphere", geom_size=[0.04],
            mass=0.2, drag_coeff=cd, com_offset=[0, 0, 0],
            inertia=0.001, cross_section_area=0.03,
            rgba=[0.5, 0.5, 0.5, 1.0],
        )
        env = TossEnv(obj, basket_distance=3.0)
        # Same throw params for all
        result = env.throw(ThrowParams(theta=45, v=5.0))
        if result.trajectory is not None and len(result.trajectory) > 1:
            plot_trajectory(ax, result.trajectory, color=col,
                           label=f"cd={cd} (landed {result.trajectory[-1, 0]:.2f}m)")

    ax.set_title("Same Throw (45\u00b0, 5 m/s) with Different Drag Coefficients\n"
                 "m=0.2kg, A=0.03m\u00b2", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "05_drag_effect.png", dpi=150)
    plt.close()
    print(f"  Saved {OUT / '05_drag_effect.png'}")


# ---------------------------------------------------------------
# 6. MuJoCo rendered frames during a throw
# ---------------------------------------------------------------
def viz_throw_frames():
    print("6. Rendering throw sequence frames...")

    obj = ObjectSpec(
        name="demo_ball", family="uniform_moderate",
        geom_type="sphere", geom_size=[0.05],
        mass=0.3, drag_coeff=0.3, com_offset=[0.0, 0.0, 0.0],
        inertia=0.001, cross_section_area=0.01,
        rgba=[0.9, 0.2, 0.2, 1.0],
    )
    basket_d = 2.0
    params, _ = cem_optimal_throw(obj, basket_d, n_samples=100, n_elite=10, n_iterations=3, seed=99)

    env = TossEnv(obj, basket_distance=basket_d)

    # Capture frames during throw
    theta_rad = np.radians(params.theta)
    vx = params.v * np.cos(theta_rad)
    vz = params.v * np.sin(theta_rad)
    env._release(linear_vel=np.array([vx, 0.0, vz]))

    frames = []
    frame_times = []
    steps_per_frame = int(1.0 / (10 * env.timestep))  # 10 fps for sparse frames

    for i in range(int(2.0 / env.timestep)):  # 2 seconds
        env._step()
        if i % steps_per_frame == 0:
            imgs = env.render(width=320, height=240)
            frames.append(imgs["side"])
            frame_times.append(env.sim_time)
        if env._object_settled():
            # Capture one more frame
            imgs = env.render(width=320, height=240)
            frames.append(imgs["side"])
            frame_times.append(env.sim_time)
            break

    # Show 8 evenly-spaced frames
    n_show = min(8, len(frames))
    indices = np.linspace(0, len(frames) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3.5))
    fig.suptitle(f"Throw Sequence: \u03b8={params.theta:.1f}\u00b0, v={params.v:.2f} m/s → basket at {basket_d}m",
                 fontsize=12)

    for j, idx in enumerate(indices):
        axes[j].imshow(frames[idx])
        axes[j].set_title(f"t={frame_times[idx]:.2f}s", fontsize=9)
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "06_throw_frames.png", dpi=150)
    plt.close()
    print(f"  Saved {OUT / '06_throw_frames.png'}")


# ---------------------------------------------------------------
# Run all
# ---------------------------------------------------------------
if __name__ == "__main__":
    viz_object_gallery()
    viz_probe_trajectories()
    viz_probe_sensitivity()
    viz_cem_throws()
    viz_drag_effect()
    viz_throw_frames()
    print(f"\nAll visualizations saved to {OUT}/")
