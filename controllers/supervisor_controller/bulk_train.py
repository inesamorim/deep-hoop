import json
from pathlib import Path
from typing import Callable
import os
import glob
import re

import gym
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

from controller import Robot
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
            match = re.search(r'(\d+)', f)
            if match:
                checkpoints.append((int(match.group()), f))

    if not checkpoints:
        return None

    # Get the file with the highest number
    _, latest_file = max(checkpoints, key=lambda x: x[0])
    return os.path.join(checkpoint_dir, latest_file)


def train(
    algo: str,
    env_creator: Callable[
        [], gym.Env
    ],
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
    Trains a reinforcement learning agent using a specified algorithm and environment.

    Args:
        algo (str): Name of the RL algorithm to use (must be a key in MODEL_MAP).
        env_creator (Callable[[], gym.Env]): A function that returns an instance of a Gym environment.
        rew_fun (str): Identifier for the reward function used by the environment.
        policy (str, optional): Policy type to use (e.g., "MlpPolicy", "MultiInputPolicy"). Defaults to "MlpPolicy".
        use_curriculum (bool, optional): Whether to use curriculum learning or not
        total_timesteps (int, optional): Total number of timesteps for training. Defaults to 1,000,000.
        time_limit (int, optional): Maximum number of steps per episode (used in TimeLimit wrapper). Defaults to TIME_LIMIT.
        model_path (str, optional): Path to save or load the model. Defaults to "./models/baller_latest.zip".
        tensorboard_log (str, optional): Directory for TensorBoard logs. Defaults to "./logs/".
        continue_training (bool, optional): Whether to continue training from an existing model. Defaults to False.
        her_params (dict, optional): Parameters specific to Hindsight Experience Replay (HER). Required if algo == "her".
        curriculum_threshold (float, optional): Performance threshold for progressing in curriculum learning. Defaults to 35.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        model: The trained RL model instance.

    Notes:
        - Saves the model at `model_path` after training.
        - Uses multiple callbacks for checkpointing, evaluation, and curriculum learning.
        - Supports HER-specific configuration when `algo` is set to "her".
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
            model = model_class.load(checkpoint_file, env=env, tensorboard_log=str(run_path / "logs"))

            # Calculate remaining timesteps
            current_timesteps = model.num_timesteps
            remaining_timesteps = max(0, total_timesteps - current_timesteps)

            if remaining_timesteps == 0:
                print(f"Run {n_run} already finished, skipping...")
                continue

            print("Continuing training from", checkpoint_file)

            # Load replay buffer
            replay_buffer_file = get_latest_checkpoint(run_path / "models", extension=".pkl")
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

if __name__ == "__main__":
    if TRAINING:
        if TRAIN_ALG == "ppo":
            ppo_model = train(
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
            sac_model = train(
                algo="sac",
                env_creator=lambda **kw: BallerSupervisor(**kw),
                rew_fun="shaped",
                policy="MlpPolicy",
                use_curriculum=USE_CURRICULUM,
                n_runs=10,
                total_timesteps=100_000,
                base_path=f"../../sac_{'with' if USE_CURRICULUM else 'without'}_curriculum",
                curriculum_threshold=0.5,
            )
        elif TRAIN_ALG == "her":
            her_model = train(
                algo="her",
                env_creator=lambda **kw: HERBallerSupervisor(**kw),
                rew_fun="sparse",
                policy="MultiInputPolicy",
                use_curriculum=USE_CURRICULUM,
                n_runs=10,
                total_timesteps=400_000,
                base_path=f"../../sac_{'with' if USE_CURRICULUM else 'without'}_curriculum",
                curriculum_threshold=0.5,
                her_params={"n_sampled_goal": 4, "goal_selection_strategy": "final"},
            )
        else:
            raise ValueError(f"Invalid algorithm: {TRAIN_ALG}")
    else:
        raise NotImplementedError("TODO: this")
