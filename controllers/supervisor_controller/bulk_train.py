import json
from pathlib import Path
from typing import Callable
import os
import re

import gym
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

from controllers.supervisor_controller.callbacks import (
    BallerEvalCallback,
    CurriculumCallback,
    DIFFICULTIES,
)
from controllers.supervisor_controller.environments import (
    HERBallerSupervisor,
    BallerSupervisor,
)

# Train Settings
TRAINING = True
TRAIN_ALG = "ppo"
USE_CURRICULUM = True

# Constants
TIME_LIMIT = 1_000
MODEL_MAP = {"her": SAC, "sac": SAC, "ppo": PPO}


def get_latest_checkpoint(checkpoint_dir: str | Path, extension: str):
    # Get all .zip files
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith(extension):
            match = re.search(r"(\d+)", f)
            if match:
                checkpoints.append((int(match.group()), f))

    if not checkpoints:
        return None

    # Get the file with the highest number
    _, latest_file = max(checkpoints, key=lambda x: x[0])
    return os.path.join(checkpoint_dir, latest_file)


def train(
    algo: str,
    env_creator: Callable[[], gym.Env],
    rew_fun: str,
    policy: str = "MlpPolicy",  # or "MultiInputPolicy"
    use_curriculum: bool = True,
    n_runs: int = 10,
    total_timesteps: int = 1_000_000,
    time_limit: int = TIME_LIMIT,
    base_path: str = "../../runs/",
    her_params: dict = None,  # only for HER
    curriculum_threshold: float = 35,
    seed: int = 0,
):
    """
    Trains a reinforcement learning agent using the specified algorithm and environment configuration.

    Args:
        algo (str): The name of the RL algorithm to use (must be a key in MODEL_MAP).
        env_creator (Callable[[], gym.Env]): A function that returns a Gym environment instance.
            It must accept a 'rew_fun' argument and return a configured environment.
        rew_fun (str): Identifier or name of the reward function used by the environment.
        policy (str, optional): Policy architecture to use (e.g., "MlpPolicy", "MultiInputPolicy"). Defaults to "MlpPolicy".
        use_curriculum (bool, optional): Whether to use curriculum learning. Defaults to True.
        n_runs (int, optional): Number of independent training runs to perform. Defaults to 10.
        total_timesteps (int, optional): Total number of timesteps for each training run. Defaults to 1,000,000.
        time_limit (int, optional): Time limit per episode (used in the TimeLimit wrapper). Defaults to TIME_LIMIT.
        base_path (str, optional): Base directory where training outputs (models, logs) will be saved. Defaults to "../../runs/".
        her_params (dict, optional): Parameters for Hindsight Experience Replay (only applicable if `algo == 'her'`).
        curriculum_threshold (float, optional): Reward threshold to advance difficulty in curriculum learning. Defaults to 35.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        None

    Side Effects:
        - Saves trained models, replay buffers, and logs under the specified `base_path`.
        - Resumes training from the latest checkpoint if available.
        - Uses callbacks for checkpointing, evaluation, and optional curriculum adjustment.

    Notes:
        - HER-specific configuration is only activated if `algo` is "her" and `her_params` is provided.
        - The environment must support setting a reward function via the `rew_fun` argument.
        - Automatically restores training state from checkpoints, including model weights, replay buffer, and curriculum difficulty.
    """

    # ENVIRONMENT
    env = env_creator(rew_fun=rew_fun)
    env = TimeLimit(env, time_limit)

    base_path = Path(base_path)

    for n_run in range(n_runs):
        print(f"Starting run {n_run}")
        run_path = base_path / f"run_{n_run}"
        models_path = run_path / "models"

        run_path.mkdir(parents=True, exist_ok=True)
        models_path.mkdir(parents=True, exist_ok=True)

        # CALLBACKS
        ckpt = CheckpointCallback(
            save_freq=50_000,
            save_path=str(models_path),
            name_prefix=f"{algo}",
            save_replay_buffer=algo != "ppo",
        )
        eval_cb = BallerEvalCallback(
            env,
            best_model_save_path=str(models_path),
            log_path=str(run_path / "logs"),
            eval_freq=10_000,
            n_eval_episodes=10,
            deterministic=True,
        )
        curriculum_path = models_path
        if use_curriculum:
            curr_cb = CurriculumCallback(
                env,
                threshold=curriculum_threshold,
                eval_freq=5_000,
                max_difficulty=len(DIFFICULTIES),
                starting_difficulty=0,
                save_path=str(curriculum_path),
                verbose=1,
            )
            callbacks = CallbackList([ckpt, eval_cb, curr_cb])
        else:
            callbacks = CallbackList([ckpt, eval_cb])

        # LOAD/INIT MODEL
        checkpoint_file = get_latest_checkpoint(run_path / "models", extension=".zip")
        if checkpoint_file:
            # Load model
            model_class = MODEL_MAP[algo]
            model = model_class.load(
                checkpoint_file, env=env, tensorboard_log=str(run_path / "logs")
            )

            # Calculate remaining timesteps
            current_timesteps = model.num_timesteps
            remaining_timesteps = max(0, total_timesteps - current_timesteps)

            if remaining_timesteps == 0:
                print(f"Run {n_run} already finished, skipping...")
                continue

            print("Continuing training from", checkpoint_file)

            # Load replay buffer
            replay_buffer_file = get_latest_checkpoint(
                run_path / "models", extension=".pkl"
            )
            if replay_buffer_file:
                model.load_replay_buffer(replay_buffer_file)

            # Load curriculum
            if use_curriculum:
                curriculum_file = curriculum_path / "curriculum_latest.json"
                if os.path.exists(curriculum_file):
                    with open(curriculum_file, "r") as f:
                        difficulty = json.load(f)["difficulty"]
                    curr_cb.set_difficulty(difficulty)
                    print("Difficulty set to", difficulty)
        else:
            model_class = MODEL_MAP[algo]
            init_kwargs = dict(
                policy=policy,
                env=env,
                tensorboard_log=str(run_path / "logs"),
                verbose=1,
                seed=seed,
            )
            # if using HER, add special replay buffer + kwargs:
            if algo == "her":
                init_kwargs.update(
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs={
                        "n_sampled_goal": her_params.get("n_sampled_goal", 4),
                        "goal_selection_strategy": her_params.get(
                            "goal_selection_strategy", "final"
                        ),
                        "copy_info_dict": True,
                    },
                    learning_starts=time_limit,
                )
            model = model_class(**init_kwargs)

            remaining_timesteps = total_timesteps

        # TRAIN
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
        )

        print(f"Run {n_run} is done!")


def eval_all_trained_models(
    algo: str,
    use_curriculum: bool,
    env_creator: Callable[[], gym.Env],
    base_path: str = "../../runs/",
    eval_path: str = "../../evaluations/",
    n_eval_episodes: int = 10,
    deterministic: bool = True,
):
    assert (
        algo in MODEL_MAP
    ), f"Unknown algo '{algo}' – must be one of {list(MODEL_MAP)}"

    # Instantiate and wrap the env
    env = env_creator(rew_fun="sparse" if algo == "her" else "shaped")
    env = TimeLimit(env, TIME_LIMIT)

    base_path = Path(base_path)
    flag = "with" if use_curriculum else "without"
    selected_path = base_path / f"{algo}_{flag}_curriculum"

    for n_run, run_path in enumerate(os.listdir(selected_path)):
        print(f"\n>> Evaluating run {n_run} of {algo} ({flag} curriculum)")

        run_path = selected_path / run_path
        model_path = get_latest_checkpoint(run_path / "models", extension=".zip")
        if model_path is None:
            print(f"Checkpoint not found {algo} {flag} curriculum")
            return

        # Choose and load the model
        model_class = MODEL_MAP[algo]
        model = model_class.load(model_path, env=env)

        # Configure logger so we get CSV / Tensorboard / stdout in save_path
        save_path = Path(eval_path) / f"{algo}_{flag}_curriculum"
        os.makedirs(save_path, exist_ok=True)
        new_logger = configure(str(save_path), ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        eval_cb = BallerEvalCallback(
            env,
            eval_freq=1,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
        )
        # manually bootstrap the callback
        eval_cb.model = model
        eval_cb.eval_env = model.get_env()
        # a fake "step" to trigger evaluation immediately
        eval_cb._on_step()


if __name__ == "__main__":
    if TRAINING:
        if TRAIN_ALG == "ppo":
            train(
                algo="ppo",
                env_creator=lambda **kw: BallerSupervisor(**kw),
                rew_fun="shaped",
                policy="MlpPolicy",
                use_curriculum=USE_CURRICULUM,
                n_runs=10,
                total_timesteps=1_000_000,
                base_path=f"../../runs/ppo_{'with' if USE_CURRICULUM else 'without'}_curriculum",
                curriculum_threshold=0.5,
            )
        elif TRAIN_ALG == "sac":
            train(
                algo="sac",
                env_creator=lambda **kw: BallerSupervisor(**kw),
                rew_fun="shaped",
                policy="MlpPolicy",
                use_curriculum=USE_CURRICULUM,
                n_runs=10,
                total_timesteps=100_000,
                base_path=f"../../runs/sac_{'with' if USE_CURRICULUM else 'without'}_curriculum",
                curriculum_threshold=0.5,
            )
        elif TRAIN_ALG == "her":
            train(
                algo="her",
                env_creator=lambda **kw: HERBallerSupervisor(**kw),
                rew_fun="sparse",
                policy="MultiInputPolicy",
                use_curriculum=USE_CURRICULUM,
                n_runs=10,
                total_timesteps=400_000,
                base_path=f"../../runs/her_{'with' if USE_CURRICULUM else 'without'}_curriculum",
                curriculum_threshold=0.5,
                her_params={"n_sampled_goal": 4, "goal_selection_strategy": "final"},
            )
        else:
            raise ValueError(f"Invalid algorithm: {TRAIN_ALG}")
    else:
        eval_all_trained_models(
            algo=TRAIN_ALG,
            use_curriculum=USE_CURRICULUM,
            env_creator=(
                (lambda **kw: HERBallerSupervisor(**kw))
                if TRAIN_ALG == "her"
                else (lambda **kw: BallerSupervisor(**kw))
            ),
            base_path="../../runs/",
            eval_path="../../evaluations/",
            n_eval_episodes=10,
            deterministic=True,
        )
