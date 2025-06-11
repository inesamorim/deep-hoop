import os
import glob
from itertools import product, combinations
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from tqdm.auto import tqdm


def load_scalar_from_tb_log(event_file, tag):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        return None

    events = ea.Scalars(tag)
    return pd.DataFrame({
        "step": [e.step for e in events],
        "value": [e.value for e in events]
    })


def load_scalar_all_files(event_files, tag):
    dfs = []
    for file in event_files:
        ea = event_accumulator.EventAccumulator(file)
        try:
            ea.Reload()
        except Exception as e:
            print(f"[!] Could not load {file}: {e}")
            continue

        if tag not in ea.Tags().get("scalars", []):
            continue

        events = ea.Scalars(tag)
        df = pd.DataFrame({
            "step": [e.step for e in events],
            "value": [e.value for e in events]
        })
        dfs.append(df)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="step")
    combined = combined.sort_values("step")
    return combined.reset_index(drop=True)


def plot_dual_x_groups(
        base_log_path: str,
        group1_runs: list[str],
        group2_runs: list[str],
        group1_label: str,
        group2_label: str,
        mean_tag: str,
        save_path: str,
        std_tag=None,  # string, dict, or None
        smooth_window: int = 10,
        dpi: int = 300,
        twin_y: bool = False  # NEW: Enable separate y-axes
):
    """
    Compare two groups of runs on a shared y-axis but separate x-axes.

    Parameters
    ----------
    base_log_path : str
        Root folder containing all run subdirectories.
    group1_runs : list[str]
        Subdirectory names for the first algorithm.
    group2_runs : list[str]
        Subdirectory names for the second algorithm.
    group1_label : str
        Label for the bottom x-axis (e.g. "Timesteps (PPO)").
    group2_label : str
        Label for the top x-axis (e.g. "Timesteps (SAC)").
    mean_tag : str
        TensorBoard scalar tag to plot (e.g. "rollout/ep_rew_mean").
    std_tag : str or dict or None
        Scalar tag(s) for standard deviation:
          • None → no shading
          • single str → same tag for all runs
          • dict → {run_name: tag_for_that_run}
    smooth_window : int
        Rolling window size for smoothing.
    save_path : str or None
        If provided, path to save the figure (PNG/PDF).
    dpi : int
        Resolution for saving.
    twin_y : bool
        If True, use separate y-axes for each group.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twiny()

    # Create twin y-axis if requested
    if twin_y:
        ax3 = ax2.twinx()  # ax3 shares x-axis with ax2 (top x-axis)
    else:
        ax3 = ax2  # Use same axis

    def _get_std_tag(run):
        if std_tag is None:
            return None
        if isinstance(std_tag, dict):
            return std_tag.get(run)
        return std_tag

    def _plot_runs(ax_x, ax_y, runs, alg_label, axis_label):
        nonlocal color_idx
        ax_x.set_xlabel(axis_label, labelpad=8)

        for run in runs:
            path = os.path.join(base_log_path, run)
            files = sorted(glob.glob(f"{path}/**/events.out.tfevents.*", recursive=True),
                           key=os.path.getmtime)
            if not files:
                print(f"[!] no event files for {run}")
                continue

            df_mean = load_scalar_all_files(files, mean_tag)
            if df_mean is None:
                print(f"[!] '{mean_tag}' missing in {run}")
                continue
            df_mean["mean_smooth"] = (
                df_mean["value"].rolling(smooth_window, min_periods=1).mean()
            )

            color = colors[color_idx % len(colors)]
            ax_y.plot(df_mean["step"], df_mean["mean_smooth"],
                      label=f"{alg_label} {' '.join(run.split('_')[1:])}", color=color)

            tag_std = _get_std_tag(run)
            if tag_std:
                df_std = load_scalar_all_files(files, tag_std)
                if df_std is not None:
                    merged = pd.merge(df_mean, df_std, on="step",
                                      suffixes=("", "_std"))
                    merged["std_smooth"] = (
                        merged["value_std"].rolling(smooth_window, min_periods=1).mean()
                    )
                    upper = merged["mean_smooth"] + merged["std_smooth"]
                    lower = merged["mean_smooth"] - merged["std_smooth"]
                    ax_y.fill_between(merged["step"], lower, upper,
                                      alpha=0.2, color=color)
            color_idx += 1

    # Color cycle
    colors = plt.cm.tab10.colors
    color_idx = 0

    # Plot group1 on bottom x-axis (ax1)
    _plot_runs(ax1, ax1, group1_runs, group1_label, f"{group1_label} Timesteps")
    # Plot group2 on top x-axis (ax2) with separate y-axis (ax3) if twin_y enabled
    _plot_runs(ax2, ax3, group2_runs, group2_label, f"{group2_label} Timesteps")

    # Y-axis & title
    metric_name = mean_tag.split('/')[1].replace('_', ' ').capitalize()
    if twin_y:
        ax1.set_ylabel(f"{group1_label} {metric_name}")
        ax3.set_ylabel(f"{group2_label} {metric_name}")
    else:
        ax1.set_ylabel(metric_name)
    ax1.set_title(f"{metric_name} comparison between {group1_label} and {group2_label}")
    ax1.grid(True)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    plt.tight_layout()
    print(f"[✓] Plot saved to: {save_path}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi)


def plot_eval_bar_groups(
    eval_root: str,
    group1: list[str],
    group2: list[str] = None,
    metric_tag1: str = "eval/success_rate",
    std_tag1: str = None,
    metric_tag2: str = None,
    std_tag2: str = None,
    title: str = "Evaluation Metrics",
    twin_y: bool = False,
    label1: str = "Group 1",
    label2: str = "Group 2",
    color1: str = "royalblue",
    color2: str = "crimson",
    save_path: str = None,
    dpi: int = 300
):
    def get_stats(configs, tag, std_tag):
        labels, means, stds = [], [], []
        for cfg in configs:
            folder = os.path.join(eval_root, cfg)
            event_files = sorted(glob.glob(os.path.join(folder, "**/events.out.tfevents.*"), recursive=True))
            if not event_files:
                print(f"[!] No TB files in {folder}")
                continue

            mean_df = load_scalar_all_files(event_files, tag)
            if mean_df is None or mean_df.empty:
                print(f"[!] No values for '{tag}' in {cfg}")
                continue

            mean_val = mean_df["value"].mean()
            std_val = 0.0
            if std_tag:
                std_df = load_scalar_all_files(event_files, std_tag)
                if std_df is not None and not std_df.empty:
                    std_val = std_df["value"].mean()

            labels.append(cfg.replace("_", " "))
            means.append(mean_val)
            stds.append(std_val)
        return labels, means, stds

    if group2 is None:
        group2 = []

    metric_tag2 = metric_tag2 or metric_tag1
    std_tag2 = std_tag2 or std_tag1

    labels1, means1, stds1 = get_stats(group1, metric_tag1, std_tag1)
    labels2, means2, stds2 = get_stats(group2, metric_tag2, std_tag2)

    if not labels1 and not labels2:
        print("[!] No valid data to plot.")
        return

    x1 = np.arange(len(labels1))
    x2 = np.arange(len(labels1), len(labels1) + len(labels2))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x1, means1, yerr=stds1 if std_tag1 else None, capsize=6,
            color=color1, edgecolor="black", alpha=0.9, label=label1)
    ax1.set_xticks(np.concatenate([x1, x2]))
    ax1.set_xticklabels(labels1 + labels2)
    ax1.set_ylabel(metric_tag1.split("/")[-1].replace("_", " ").capitalize(), color=color1)
    ax1.tick_params(axis="y", colors=color1)

    if twin_y and labels2:
        ax2 = ax1.twinx()
        ax2.bar(x2, means2, yerr=stds2 if std_tag2 else None, capsize=6,
                color=color2, edgecolor="black", alpha=0.9, label=label2)
        ax2.set_ylabel(metric_tag2.split("/")[-1].replace("_", " ").capitalize(), color=color2)
        ax2.tick_params(axis="y", colors=color2)
    elif not twin_y and labels2:
        ax1.bar(x2, means2, yerr=stds2 if std_tag2 else None, capsize=6,
                color=color2, edgecolor="black", alpha=0.9, label=label2)

    ax1.set_title(title)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
        print(f"[✓] Saved to {save_path}")


base_path = "./controllers/supervisor_controller/logs/paper"
ppo_runs = ["PPO_with_curriculum", "PPO_without_curriculum"]
sac_runs = ["SAC_with_curriculum", "SAC_without_curriculum"]
her_runs = ["HER_with_curriculum", "HER_without_curriculum"]

eval_metrics = [
    ("avg_ball_velocity", "std_ball_velocity", False),
    ("avg_distance", "std_distance", False),
    ("avg_joint_usage", "std_joint_usage", False),
    ("avg_throw_duration", "std_throw_duration", False),
    ("mean_ep_length", None, False),
    ("mean_reward", None, True),
    ("success_rate", None, False),
    ("avg_max_height", "std_max_height", False),
]

for metric, std, twin in eval_metrics:
    plot_eval_bar_groups(
        eval_root="./controllers/supervisor_controller/evaluation/",
        group1=["ppo_with_curriculum", "ppo_without_curriculum", "sac_with_curriculum", "sac_without_curriculum"],
        group2=["her_with_curriculum", "her_without_curriculum"],
        metric_tag1=f"eval/{metric}",
        std_tag1=f"eval/{std}" if std is not None else None,
        title=f"{metric.replace('_', ' ').capitalize()} of different models",
        color1="royalblue",
        color2="crimson" if twin else "royalblue",
        save_path=f"figs/trained/{metric}.png",
        twin_y=twin,
    )

train_metrics = [
    ("eval/avg_ball_velocity", "eval/std_ball_velocity", False),
    ("eval/avg_distance", "eval/std_distance", False),
    # ("eval/avg_joint_usage", "eval/std_joint_usage", False),  # pedro messed up this one during training :P
    ("eval/avg_throw_duration", "eval/std_throw_duration", False),
    ("eval/mean_ep_length", None, False),
    ("eval/mean_reward", None, True),
    ("eval/success_rate", None, False),
    # ("eval/avg_max_height", "eval/std_max_height", False),
    ("rollout/ep_rew_mean", None, True),
    ("rollout/ep_len_mean", None, False),
]

all_combinations = product(
    combinations(
        zip(["PPO", "SAC", "HER"], [ppo_runs, sac_runs, her_runs]),
        2,
    ),
    train_metrics,
)

for ((tag1, run1), (tag2, run2)), (metric, std, twin_y) in tqdm(all_combinations):
    plot_dual_x_groups(
        base_path,
        run1,
        run2,
        group1_label=tag1,
        group2_label=tag2,
        mean_tag=metric,
        std_tag=std,
        save_path=f"figs/{metric.split('/')[-1]}/{tag1}_vs_{tag2}.png",
        twin_y=twin_y,
    )
