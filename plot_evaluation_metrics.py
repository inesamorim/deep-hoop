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

    combined = pd.concat(dfs, ignore_index=True)
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
        smooth_window: int = 10,
        dpi: int = 300,
        twin_y: bool = False  # Enable separate y-axes
):
    """
    Compare two groups of runs by computing mean and std across multiple runs.

    Parameters
    ----------
    base_log_path : str
        Root folder containing all group subdirectories.
    group1_runs : list[str]
        Names of subdirectories for the first algorithm (each containing run_<i> subfolders).
    group2_runs : list[str]
        Names of subdirectories for the second algorithm.
    group1_label : str
        Label for the bottom x-axis (e.g. "Timesteps (PPO)").
    group2_label : str
        Label for the top x-axis (e.g. "Timesteps (SAC)").
    mean_tag : str
        TensorBoard scalar tag to plot (e.g. "rollout/ep_rew_mean").
    save_path : str
        Path to save the figure (PNG/PDF).
    smooth_window : int
        Rolling window size for smoothing.
    dpi : int
        Resolution for saving.
    twin_y : bool
        If True, use separate y-axes for each group.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twiny()

    # Create twin y-axis if requested
    if twin_y:
        ax3 = ax2.twinx()
    else:
        ax3 = ax2

    colors = plt.cm.tab10.colors
    color_idx = 0

    def _plot_group(ax_x, ax_y, groups, label, axis_label):
        nonlocal color_idx
        ax_x.set_xlabel(axis_label, labelpad=8)

        for group in groups:
            # Collect each run's smoothed mean series
            series_list = []
            for run_dir in sorted(glob.glob(os.path.join(base_log_path, group.lower(), 'run_*'))):
                # Find TensorBoard event file(s)
                event_files = glob.glob(os.path.join(run_dir, 'logs', '*', 'events.out.tfevents.*'))
                if not event_files:
                    print(f"[!] No event files for {run_dir}")
                    continue

                df = load_scalar_all_files(sorted(event_files), mean_tag)
                if df is None:
                    print(f"[!] '{mean_tag}' missing in {run_dir}")
                    continue

                # Smooth values
                df['mean_smooth'] = df['value'].rolling(smooth_window, min_periods=1).mean()
                series_list.append(df.set_index('step')['mean_smooth'].rename(os.path.basename(run_dir)))

            if not series_list:
                print(f"[!] No run files for {groups}")
                continue

            # Someone didn't reset the logs inês
            # Some indices are duplicated, this can happen if training is restarted without resetting the logs
            # hacky fix: keep the last
            series_list = [s[~s.index.duplicated(keep="last")] for s in series_list]

            # Combine all runs
            df_all = pd.concat(series_list, axis=1)
            df_mean = df_all.mean(axis=1)
            df_std = df_all.std(axis=1)

            color = colors[color_idx % len(colors)]
            ax_y.plot(df_mean.index, df_mean.values,
                      label=f"{' '.join(group.split('_'))}", color=color)
            ax_y.fill_between(df_mean.index,
                              df_mean.values - df_std.values,
                              df_mean.values + df_std.values,
                              alpha=0.2, color=color)
            color_idx += 1

    # Plot group1 on bottom x-axis (ax1)
    _plot_group(ax1, ax1, group1_runs, group1_label, f"{group1_label} Timesteps")
    # Plot group2 on top x-axis (ax2) with separate y-axis (ax3) if twin_y enabled
    _plot_group(ax2, ax3, group2_runs, group2_label, f"{group2_label} Timesteps")

    # Axis labels and title
    metric_name = mean_tag.split('/')[-1].replace('_', ' ').capitalize()
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

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    print(f"[✓] Plot saved to: {save_path}")

def plot_eval_bar_groups(
    eval_root: str,
    group1: list[str],
    group2: list[str] = None,
    metric_tag1: str = "eval/success_rate",
    metric_tag2: str = None,
    title: str = "Evaluation Metrics",
    twin_y: bool = False,
    label1: str = "Group 1",
    label2: str = "Group 2",
    color1: str = "royalblue",
    color2: str = "crimson",
    save_path: str = None,
    dpi: int = 300
):
    def compute_group_stats(group_dirs, metric_tag):
        means, stds, labels = [], [], []
        for cfg in group_dirs:
            group_path = os.path.join(eval_root, cfg)
            event_files = sorted(glob.glob(os.path.join(group_path, "events.out.tfevents.*")))
            run_means = []
            for event_file in event_files:
                df = load_scalar_all_files([event_file], metric_tag)
                if df is None or df.empty:
                    print(f"[!] No values for '{metric_tag}' in {event_file}")
                    continue
                val = df["value"].mean()
                run_means.append(val)
            if run_means:
                labels.append(cfg.replace("_", " "))
                means.append(np.mean(run_means))
                stds.append(np.std(run_means))
            else:
                print(f"[!] No valid runs for {cfg}")
        return labels, means, stds

    if group2 is None:
        group2 = []

    metric_tag2 = metric_tag2 or metric_tag1

    labels1, means1, stds1 = compute_group_stats(group1, metric_tag1)
    labels2, means2, stds2 = compute_group_stats(group2, metric_tag2)

    if not labels1 and not labels2:
        print("[!] No valid data to plot.")
        return

    x1 = np.arange(len(labels1))
    x2 = np.arange(len(labels1), len(labels1) + len(labels2))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x1, means1, yerr=stds1, capsize=6,
            color=color1, edgecolor="black", alpha=0.9, label=label1)
    ax1.set_xticks(np.concatenate([x1, x2]))
    ax1.set_xticklabels(labels1 + labels2)
    ax1.set_ylabel(metric_tag1.split("/")[-1].replace("_", " ").capitalize(), color=color1)
    ax1.tick_params(axis="y", colors=color1)

    if twin_y and labels2:
        ax2 = ax1.twinx()
        ax2.bar(x2, means2, yerr=stds2, capsize=6,
                color=color2, edgecolor="black", alpha=0.9, label=label2)
        ax2.set_ylabel(metric_tag2.split("/")[-1].replace("_", " ").capitalize(), color=color2)
        ax2.tick_params(axis="y", colors=color2)
        ax2.set_ylim(bottom=0)
    elif not twin_y and labels2:
        ax1.bar(x2, means2, yerr=stds2, capsize=6,
                color=color2, edgecolor="black", alpha=0.9, label=label2)

    ax1.set_title(title)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
        print(f"[✓] Saved to {save_path}")


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
        eval_root="./evaluations/",
        group1=["ppo_with_curriculum", "ppo_without_curriculum", "sac_with_curriculum", "sac_without_curriculum"],
        group2=["her_with_curriculum", "her_without_curriculum"],
        metric_tag1=f"eval/{metric}",
        title=f"{metric.replace('_', ' ').capitalize()} of different models",
        color1="royalblue",
        color2="crimson" if twin else "royalblue",
        save_path=f"figs/trained/{metric}.png",
        twin_y=twin,
    )
    

train_metrics = [
    ("eval/avg_ball_velocity", "eval/std_ball_velocity", False),
    ("eval/avg_distance", "eval/std_distance", False),
    ("eval/avg_joint_usage", "eval/std_joint_usage", False),
    ("eval/avg_throw_duration", "eval/std_throw_duration", False),
    ("eval/mean_ep_length", None, False),
    ("eval/mean_reward", None, True),
    ("eval/success_rate", None, False),
    ("eval/avg_max_height", "eval/std_max_height", False),
    ("rollout/ep_rew_mean", None, True),
    ("rollout/ep_len_mean", None, False),
]

ppo_runs = ["PPO_with_curriculum", "PPO_without_curriculum"]
sac_runs = ["SAC_with_curriculum", "SAC_without_curriculum"]
her_runs = ["HER_with_curriculum", "HER_without_curriculum"]

all_combinations = product(
    combinations(
        zip(["PPO", "SAC", "HER"], [ppo_runs, sac_runs, her_runs]),
        2,
    ),
    train_metrics,
)

for ((tag1, run1), (tag2, run2)), (metric, std, twin_y) in tqdm(all_combinations):
    plot_dual_x_groups(
        "./runs",
        run1,
        run2,
        group1_label=tag1,
        group2_label=tag2,
        mean_tag=metric,
        save_path=f"figs/training/{metric.split('/')[-1]}/{tag1}_vs_{tag2}.png",
        twin_y=twin_y,
        smooth_window=1,
    )
