from collections import defaultdict
from dataclasses import dataclass
import os

import gym
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


@dataclass
class Difficulty:
    hoop_pos: gym.spaces.Box
    hoop_size: gym.spaces.Box


# List of difficulties
# defines ranges for hoop position and radius
DIFFICULTIES = [
    # Easy
    Difficulty(
        hoop_pos=gym.spaces.Box(
            low=np.array([0, 2, 1.7]),
            high=np.array([0, 2, 1.7]),
            dtype=np.float32,
        ),
        hoop_size=gym.spaces.Box(
            low=np.array([0.7]),
            high=np.array([0.7]),
            dtype=np.float32,
        ),
    ),
    # Medium
    Difficulty(
        hoop_pos=gym.spaces.Box(
            low=np.array([-0.3, 2, 1.7]),
            high=np.array([0.3, 2.4, 1.8]),
            dtype=np.float32,
        ),
        hoop_size=gym.spaces.Box(
            low=np.array([0.5]),
            high=np.array([0.6]),
            dtype=np.float32,
        ),
    ),
    # Hard
    Difficulty(
        hoop_pos=gym.spaces.Box(
            low=np.array([-1, 2, 1.8]),
            high=np.array([1, 3.5, 2]),
            dtype=np.float32,
        ),
        hoop_size=gym.spaces.Box(
            low=np.array([0.45]),
            high=np.array([0.45]),
            dtype=np.float32,
        ),
    ),
]


# Callback that implements simple curriculum which scales difficulty based on agent performance
class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        threshold: float,
        eval_freq: int,
        max_difficulty: int,
        starting_difficulty: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.threshold = threshold
        self.eval_freq = eval_freq
        self.max_difficulty = max_difficulty
        self.current_difficulty = starting_difficulty

    def _on_training_start(self):
        self.training_env.env_method("set_difficulty", self.current_difficulty)
        self.logger.record("curriculum/difficulty", self.current_difficulty)
        self.logger.dump(self.num_timesteps)  # Write to disk

    def _on_step(self) -> bool:
        # Evaluate every 5000 steps
        if self.n_calls % self.eval_freq == 0:
            if self.current_difficulty == self.max_difficulty - 1:
                # Already at max difficulty
                return True

            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=10, deterministic=True
            )
            if self.verbose > 0:
                print(
                    f"Eval reward at difficulty {self.current_difficulty}: {mean_reward}"
                )

            # If performance is good, increase difficulty
            if (
                mean_reward > self.threshold
                and self.current_difficulty < self.max_difficulty - 1
            ):
                self.current_difficulty += 1
                if self.verbose > 0:
                    # Log difficulty change
                    print(f"Increasing difficulty to {self.current_difficulty}")
                    self.logger.record("curriculum/difficulty", self.current_difficulty)

                    log_dir = self.logger.dir or "./"
                    log_file = os.path.join(log_dir, "curriculum.txt")
                    with open(log_file, "a") as f:
                        f.write(
                            f"Step {self.num_timesteps}: Increased difficulty to {self.current_difficulty} with eval reward {mean_reward}\n"
                        )

                # Update training and evaluation environments
                self.training_env.env_method("set_difficulty", self.current_difficulty)
                self.eval_env.set_difficulty(self.current_difficulty)

        return True


# Superclass of EvalCallback that records additional metrics
class BallerEvalCallback(EvalCallback):
    def __init__(self, eval_env, *args, **kwargs):
        super().__init__(eval_env, *args, **kwargs)
        self.metrics = defaultdict(list)

    def _on_step(self) -> bool:
        # Calculate original metrics
        result = super()._on_step()

        if self.n_calls % self.eval_freq == 0:
            n_episodes = self.n_eval_episodes
            (
                successes,
                distances,
                throw_duration,
                throw_vel,
                joint_usage,
                max_heights,
            ) = ([], [], [], [], [], [])

            # Run n episodes
            for _ in range(n_episodes):
                obs = self.eval_env.reset()
                done = False

                released_ball, ball_vel, ball_heights = [], [], []
                joint_use = 0

                # Episode loop
                while not done:
                    action, _ = self.model.predict(
                        obs, deterministic=self.deterministic
                    )
                    obs, reward, done, info = self.eval_env.step(action)

                    info = info[0]
                    released_ball.append(info["released_ball"])
                    ball_vel.append(info["ball_vel_norm"])
                    ball_heights.append(info["ball_pos"])
                    joint_use += np.sum(np.abs(action))

                # Extract episode metrics
                successes.append(info.get("is_success", 0.0))
                distances.append(info.get("distance_to_goal", 0.0))
                max_heights.append(np.max(ball_heights))
                joint_usage.append(joint_use)

                released_ball = np.asarray(released_ball)
                indices = np.where(released_ball == 1)[0]
                if indices.size > 0:
                    release_time = indices[0]
                else:
                    release_time = len(released_ball)
                throw_duration.append(release_time)

                throw_vel.append(
                    np.mean(ball_vel[release_time:])
                    if len(ball_vel[release_time:]) > 0
                    else 0.0
                )

            # Compute and log means
            self.metrics["success_rate"].append(np.mean(successes))
            self.metrics["avg_distance"].append(np.mean(distances))
            self.metrics["std_distance"].append(np.std(distances))
            self.metrics["avg_throw_duration"].append(np.mean(throw_duration))
            self.metrics["std_throw_duration"].append(np.std(throw_duration))
            self.metrics["avg_ball_velocity"].append(np.mean(throw_vel))
            self.metrics["std_ball_velocity"].append(np.std(throw_vel))
            self.metrics["avg_joint_usage"].append(np.mean(joint_usage))
            self.metrics["std_joint_usage"].append(np.std(joint_usage))
            self.metrics["avg_max_height"].append(np.mean(max_heights))
            self.metrics["std_max_height"].append(np.std(max_heights))

            for key, val in self.metrics.items():
                self.logger.record(f"eval/{key}", val[-1])

            # Write to disk
            self.logger.dump(self.num_timesteps)

        return result
