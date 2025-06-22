from typing import Callable
import os
import glob

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
CONTINUE_TRAINING = False
TRAIN_ALG = "ppo"

# Constants
TIME_LIMIT = 1_000
MODEL_MAP = {"her": SAC, "sac": SAC, "ppo": PPO}


def train(
    algo: str,
    env_creator: Callable[
        [], gym.Env
    ],
    rew_fun: str,
    policy: str = "MlpPolicy",  # or "MultiInputPolicy"
    use_curriculum: bool = True,
    total_timesteps: int = 1_000_000,
    time_limit: int = TIME_LIMIT,
    model_path: str = "./models/baller_latest.zip",
    tensorboard_log: str = "./logs/",
    continue_training: bool = False,
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

    # CALLBACKS
    ckpt = CheckpointCallback(
        save_freq=50_000, save_path="./models/", name_prefix=f"{algo}"
    )
    eval_cb = BallerEvalCallback(
        env,
        best_model_save_path="./models/best_" + algo,
        log_path="./logs/" + algo,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )
    if use_curriculum:
        curr_cb = CurriculumCallback(
            env,
            threshold=curriculum_threshold,
            eval_freq=5_000,
            max_difficulty=len(DIFFICULTIES),
            starting_difficulty=0,
            verbose=1,
        )
        callbacks = CallbackList([ckpt, eval_cb, curr_cb])
    else:
        callbacks = CallbackList([ckpt, eval_cb])

    # LOAD/INIT MODEL
    if continue_training and os.path.exists(model_path):
        model_class = MODEL_MAP[algo]
        model = model_class.load(model_path, env=env, tensorboard_log=tensorboard_log)
    else:
        model_class = MODEL_MAP[algo]
        init_kwargs = dict(
            policy=policy,
            env=env,
            tensorboard_log=tensorboard_log,
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

    # TRAIN
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=not continue_training,
    )
    model.save(model_path)
    return model


def evaluate(
    algo: str,
    model_path: str,
    env_ctor: Callable[..., gym.Env],
    rew_fun: str,
    save_path: str,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    difficulty: int = None,
):
    """
    Evaluates a trained model over a specified number of episodes.

    Args:
        algo (str): The name of the algorithm used to train the model.
        model_path (str): Path to the saved model file.
        env_ctor (Callable[..., gym.Env]): A callable that returns a Gym environment instance.
        rew_fun (str): Identifier for the reward function used by the environment.
        save_path (str): Directory path to save evaluation logs and metrics.
        n_eval_episodes (int, optional): Number of episodes to run for evaluation. Defaults to 10.
        deterministic (bool, optional): Whether to use deterministic actions during evaluation. Defaults to True.
        difficulty (int, optional): Difficulty level to evaluate on. If None, uses the maximum available difficulty.

    Raises:
        ValueError: If the specified algorithm is not recognized (i.e., not found in MODEL_MAP).

    Notes:
        - Wraps the environment with a time limit and sets a fixed difficulty level.
        - Uses `BallerEvalCallback` to perform evaluation and log results.
        - Configures logging to output to stdout, CSV, and TensorBoard.
        - Contains a workaround for Webots simulator by resetting `Robot.created` to None after evaluation.

    """
    # Instantiate and wrap the env
    env = env_ctor(rew_fun=rew_fun)
    env = TimeLimit(env, TIME_LIMIT)
    # set to hardest level by default
    if difficulty is None:
        difficulty = len(DIFFICULTIES) - 1
    env.set_difficulty(difficulty)

    # Choose and load the model
    if algo not in MODEL_MAP:
        raise ValueError(f"Unknown algo '{algo}' – must be one of {list(MODEL_MAP)}")
    model_class = MODEL_MAP[algo]
    model = model_class.load(model_path, env=env)

    # Configure logger so we get CSV / Tensorboard / stdout in save_path
    os.makedirs(save_path, exist_ok=True)
    new_logger = configure(save_path, ["stdout", "csv", "tensorboard"])
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

    env.close()
    del env

    # webots only allows 1 environment per process by checking if this variable is None
    # setting it back to None seems to work as a workaround
    # lets hope it's not something important ¯\_(ツ)_/¯
    Robot.created = None


def find_model_path(base_dir: str) -> str:
    """Return the first baller_*.zip under base_dir, or raise."""
    matches = glob.glob(os.path.join(base_dir, "baller_*.zip"))
    if not matches:
        raise FileNotFoundError(f"No model found in {base_dir}")
    return matches[0]


def run_all_trained_models():
    for algo in ["her"]:
        for with_curr in (False,):
            flag = "with" if with_curr else "without"
            model_dir = os.path.join(
                "..", "..", "trained_models", f"{algo}_{flag}_curriculum"
            )
            try:
                model_path = find_model_path(model_dir)
            except FileNotFoundError as e:
                print(f"Skipping {algo} ({flag}): {e}")
                continue

            save_path = os.path.join("evaluation", f"{algo}_{flag}_curriculum")
            os.makedirs(save_path, exist_ok=True)

            if algo == "her":
                env_ctor = lambda rew_fun: HERBallerSupervisor(rew_fun)
                rew_fun = "sparse"
            else:
                env_ctor = lambda rew_fun: BallerSupervisor(rew_fun)
                rew_fun = "shaped"

            print(
                f"\n>> Evaluating {algo} ({'with' if with_curr else 'without'} curriculum)"
            )
            evaluate(
                algo=algo,
                model_path=model_path,
                env_ctor=env_ctor,
                rew_fun=rew_fun,
                save_path=save_path,
                n_eval_episodes=100,
                deterministic=True,
                difficulty=None,  # defaults to max difficulty
            )


if __name__ == "__main__":
    if TRAINING:
        if TRAIN_ALG == "ppo":
            ppo_model = train(
                algo="ppo",
                env_creator=lambda **kw: BallerSupervisor(**kw),
                rew_fun="shaped",
                policy="MlpPolicy",
                total_timesteps=1_000_000,
                curriculum_threshold=0.5,
                continue_training=CONTINUE_TRAINING,
            )
        elif TRAIN_ALG == "sac":
            sac_model = train(
                algo="sac",
                env_creator=lambda **kw: BallerSupervisor(**kw),
                rew_fun="shaped",
                policy="MlpPolicy",
                total_timesteps=1_000_000,
                curriculum_threshold=0.5,
                continue_training=CONTINUE_TRAINING,
            )
        elif TRAIN_ALG == "her":
            her_model = train(
                algo="her",
                env_creator=lambda **kw: HERBallerSupervisor(**kw),
                rew_fun="sparse",
                policy="MultiInputPolicy",
                total_timesteps=1_000_000,
                curriculum_threshold=0.5,
                her_params={"n_sampled_goal": 4, "goal_selection_strategy": "final"},
                continue_training=CONTINUE_TRAINING,
            )
        else:
            raise ValueError(f"Invalid algorithm: {TRAIN_ALG}")
    else:
        run_all_trained_models()
