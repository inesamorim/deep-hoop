from collections import defaultdict
from typing import Callable
import os

import gym
import numpy as np
from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from stable_baselines3 import PPO, DDPG, TD3, SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback, BaseCallback
from gym.wrappers import TimeLimit
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from dataclasses import dataclass

from controller import PositionSensor, Motor, Supervisor

JOINT_NAMES = [f"joint{i}" for i in range(1, 4)]
JOINT_SENSOR_NAMES = [f"joint{i}_sensor" for i in range(1, 4)]
HAND_NAMES = [f"finger_{i}_joint_1" for i in [1, 2, "middle"]]
HAND_SENSOR_NAMES = [f"finger_{i}_joint_1_sensor" for i in [1, 2, "middle"]]

ACTION_SCALE = [
    (-4, 4),
    (-4, 0),
    (-4, 0),
    (-4, 0),
]

@dataclass
class Difficulty:
    hoop_pos: gym.spaces.Box
    hoop_size: gym.spaces.Box

DIFFICULTIES = [
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


def rew_shaped(
    ball_pos: list[float],
    ball_vel: list[float],
    hoop_pos: list[float],
    ball_last_dist: float,
    passing_radius: float,
) -> float:
    ball_pos = np.asarray(ball_pos)
    ball_vel = np.asarray(ball_vel)
    hoop_pos = np.asarray(hoop_pos)
    rel_pos = hoop_pos - ball_pos

    # Delta-distance
    d = np.linalg.norm(rel_pos)
    r_dd = ball_last_dist - d

    # Raw velocity toward hoop
    dir_vec = rel_pos / (d + 1e-6)
    r_vel = np.dot(ball_vel, dir_vec)

    # Time penalty
    r_time = -0.01

    # Success boost
    passed_hoop = is_ball_passing(ball_pos, hoop_pos, passing_radius)

    reward = 0 * r_dd + 0.5 * r_vel + r_time + 10 * passed_hoop
    print(f"{r_dd=}, {r_vel=}, {passed_hoop=}, {reward=}")
    return reward


def rew_sparse(
    ball_pos: list[float],
    hoop_pos: list[float],
    passing_radius: float,
) -> float:
    return 1 if is_ball_passing(ball_pos, hoop_pos, passing_radius) else 0


def rew_sparse_dist(
    ball_pos: list[float],
    ball_vel: list[float],
    hoop_pos: list[float],
    passing_radius: float,
) -> float:
    ball_pos = np.asarray(ball_pos)
    ball_vel = np.asarray(ball_vel)
    hoop_pos = np.asarray(hoop_pos)
    dist = np.linalg.norm(hoop_pos - ball_pos)

    return -dist if is_done(ball_pos, ball_vel, hoop_pos, passing_radius) else 0


def is_done(ball_pos: list[float], ball_vel: list[float], hoop_pos: list[float], passing_radius: float) -> bool:
    if is_ball_passing(ball_pos, hoop_pos, passing_radius):
        # Hit the hoop
        return True

    if ball_vel[2] < -0.1 and ball_pos[2] < hoop_pos[2]:
        # Missed hoop
        return True

    if ball_pos[2] <= 0.2:
        # Hit ground
        return True

    return False


def is_ball_passing(ball_center: list[float], hoop_center: list[float], passing_radius: float) -> bool:
    # Compute distance between ball center and passing zone center (XY distance)
    dist = np.sqrt((ball_center[0] - hoop_center[0]) ** 2 +
                   (ball_center[1] - hoop_center[1]) ** 2)

    # Ball passes through if it's within the passing radius and at the correct Z level
    return dist <= passing_radius and abs(ball_center[2] - hoop_center[2]) < 0.05  # Allowing small Z tolerance


def to_unit(x: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    return 2 * (x - a) / (b - a) - 1


def from_unit(x: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    return ((x + 1) / 2) * (b - a) + a


class BallerSupervisor(RobotSupervisorEnv):
    def __init__(self, rew_fun: str, timestep: int | None=None):
        super().__init__(timestep=timestep)

        self.rew_fun = rew_fun
        self.cur_difficulty = 0

        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(15,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.timestep = int(self.getBasicTimeStep())
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.hoop = self.getFromDef("HOOP")
        self.ball = self.getFromDef("BALL")

        # Joint Motors
        self.joints = []
        for joint_name in JOINT_NAMES:
            joint: Motor = self.getDevice(joint_name)
            joint.setPosition(float("inf"))
            joint.setVelocity(0)
            self.joints.append(joint)

        # Joint Sensors
        self.joint_sensors = []
        for joint_name in JOINT_SENSOR_NAMES:
            sensor: PositionSensor = self.getDevice(joint_name)
            sensor.enable(self.timestep)
            self.joint_sensors.append(sensor)

        # Hand Motors
        self.hand = []
        for hand_name in HAND_NAMES:
            joint: Motor = self.getDevice(hand_name)
            joint.setPosition(float("inf"))
            joint.setVelocity(0)
            self.hand.append(joint)

        # Hand Sensors
        self.hand_sensors = []
        for hand_name in HAND_SENSOR_NAMES:
            sensor: PositionSensor = self.getDevice(hand_name)
            sensor.enable(self.timestep)
            self.hand_sensors.append(sensor)

        self.steps_per_episode = 200  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved

    def set_difficulty(self, difficulty: int):
        self.cur_difficulty = difficulty

    def get_observations(self):
        obs = []

        ball_pos = np.asarray(self.ball.getPosition())
        hoop_pos = np.asarray(self.hoop.getPosition())

        # Relative pos
        relative_pos = hoop_pos - ball_pos
        relative_pos = to_unit(relative_pos, -5, 5)
        obs.extend(relative_pos)

        # Joint angles
        for sensor, motor in zip(self.joint_sensors, self.joints):
            value = sensor.getValue()
            scaled = to_unit(value, motor.min_position, motor.max_position)
            obs.append(scaled)

        # Hand angles
        for sensor, motor in zip(self.hand_sensors, self.hand):
            value = sensor.getValue()
            scaled = to_unit(value, motor.min_position, motor.max_position)
            obs.append(scaled)

        # Ball Acceleration and Velocity
        ball_cur_vel = np.array(self.ball.getVelocity()[:3])
        ball_accel = (ball_cur_vel - self.ball_last_vel) / self.timestep
        obs.extend(to_unit(ball_accel, -0.5, 0.5))
        obs.extend(to_unit(ball_cur_vel, -5, 5))

        # Update ball min distance
        dist = np.linalg.norm(relative_pos)
        if dist < self.closest_dist:
            self.closest_dist = dist # TODO: see if there is a way to improve this
            self.closest_pos = ball_pos

        print("Obs:", obs)
        return obs

    def get_default_observation(self):
        return self.get_observations()

    def get_reward(self, action=None):
        ball_pos = np.asarray(self.ball.getPosition())
        ball_vel = np.asarray(self.ball.getVelocity()[:3])
        hoop_pos = np.asarray(self.hoop.getPosition())

        if self.rew_fun == "shaped":
            return rew_shaped(ball_pos, ball_vel, hoop_pos, self.ball_last_dist, self.passing_radius)
        elif self.rew_fun == "sparse":
            return rew_sparse(ball_pos, hoop_pos, self.passing_radius)
        elif self.rew_fun == "sparse_dist":
            return rew_sparse_dist(ball_pos, ball_vel, hoop_pos, self.passing_radius)
        else:
            raise ValueError("invalid reward function:", self.rew_fun)

    def is_ball_passing(self):
        ball_center = self.ball.getPosition()
        hoop_pos = self.hoop.getPosition()
        return is_ball_passing(ball_center, hoop_pos, self.passing_radius)

    def is_done(self):
        ball_pos = self.ball.getPosition()
        ball_vel = self.ball.getVelocity()
        hoop_pos = self.hoop.getPosition()
        return is_done(ball_pos, ball_vel, hoop_pos, self.passing_radius)

    def solved(self):
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 195.0:  # Last 100 episodes' scores average value
                return True
        return False

    def get_info(self):
        hoop_pos = np.asarray(self.hoop.getPosition())
        ball_pos = np.asarray(self.ball.getPosition())
        dist = np.linalg.norm(hoop_pos - ball_pos)
        ball_vel = np.asarray(self.ball.getVelocity()[:3])

        return {
            "is_success": self.is_ball_passing(),
            "distance_to_goal": dist,
            "released_ball": self.released_ball,
            "ball_vel_norm": np.linalg.norm(ball_vel),
        }

    def render(self, mode='human'):
        pass

    def apply_action(self, action):
        print("Action:", action)

        ball_pos = np.asarray(self.ball.getPosition())
        hoop_pos = np.asarray(self.hoop.getPosition())
        hand_pos = np.asarray(self.getFromDef("HAND").getPosition())

        if ball_pos[2] >= hoop_pos[2]:
            self.was_higher_than_hoop = True

        # Check if released ball
        if np.linalg.norm(ball_pos - hand_pos) > 0.2:
            self.released_ball = True

        # Update ball velocity
        # yes this must be done here
        self.ball_last_vel = np.asarray(self.ball.getVelocity()[:3])
        self.ball_last_dist = np.linalg.norm(hoop_pos - ball_pos)

        # Set joint velocities
        for i in range(len(self.joints)):
            vel = from_unit(action[i], *ACTION_SCALE[i])
            self.joints[i].setPosition(float("inf"))
            self.joints[i].setVelocity(vel)

        # Release ball
        release_ball = from_unit(action[-1], *ACTION_SCALE[-1])
        for i in range(len(self.hand)):
            self.hand[i].setPosition(float("inf"))
            self.hand[i].setVelocity(release_ball)

    def reset(self):
        self.was_higher_than_hoop = False
        self.ball_last_vel = np.array([0, 0, 0])

        self.released_ball = False # if robot is still holding the ball
        self.passed_hoop = False # ball has passed the hoop
        self.closest_dist = 99999

        # Reset simulation
        obs = super().reset()

        # Set hoop pos
        difficulty =  DIFFICULTIES[self.cur_difficulty]
        hoop_trans = self.hoop.getField("translation")
        new_pos = difficulty.hoop_pos.sample()
        hoop_trans.setSFVec3f(new_pos.tolist())

        # Set hoop size
        hoop_radius = self.hoop.getField("majorRadius")
        new_radius = difficulty.hoop_size.sample()
        hoop_radius.setSFFloat(float(new_radius[0]))

        # We need to step so the new settings are registered
        if super(Supervisor, self).step(self.timestep) == -1:
            exit()

        hoop_pos = np.asarray(self.hoop.getPosition())
        ball_pos = np.asarray(self.ball.getPosition())
        dist = np.linalg.norm(hoop_pos - ball_pos)

        assert all(hoop_pos == new_pos), f"{hoop_pos=}, {new_pos=}"

        self.ball_last_dist = dist

        # Get the outer radius (distance from center to tube middle)
        outer_radius = self.hoop.getField("majorRadius").getSFFloat()
        # Get the inner radius (tube thickness)
        inner_radius = self.hoop.getField("minorRadius").getSFFloat()
        self.passing_radius = outer_radius - inner_radius

        self.closest_pos = ball_pos
        self.passing_center = hoop_pos

        return obs

class HERBallerSupervisor(BallerSupervisor):
    def __init__(self, rew_fun: str):
        super().__init__(rew_fun=rew_fun)
        self.observation_space = gym.spaces.Dict({
            "observation": self.observation_space,
            "achieved_goal": gym.spaces.Box(0, 1, (3,)),
            "desired_goal": gym.spaces.Box(0, 1, (3,)),
        })

    def get_observations(self):
        obs_orig = super().get_observations()

        return {
            "observation": obs_orig,
            "achieved_goal": self.ball.getPosition(),
            "desired_goal": self.hoop.getPosition(),
        }

    def get_default_observation(self):
        return self.get_observations()

    def get_info(self):
        return super().get_info() | {
            "ball_last_dist": self.ball_last_dist,
            "ball_vel": self.ball.getVelocity()[:3],
            "passing_radius": self.passing_radius,
        }

    def get_reward(self, action=None):
        return super().get_reward(action)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return np.asarray(
            [self._compute_reward(ball, hoop, info)
             for ball, hoop, info in zip(achieved_goal, desired_goal, info)]
        )

    def _compute_reward(self, ball_pos: list[float], hoop_pos: list[float], info: dict):
        if self.rew_fun == "shaped":
            raise NotImplementedError("im lazy and we probably are not going to use 'shaped' with HER")
        elif self.rew_fun == "sparse":
            return rew_sparse(ball_pos, hoop_pos, info["passing_radius"])
        elif self.rew_fun == "sparse_dist":
            return rew_sparse_dist(ball_pos, info["ball_vel"], hoop_pos, info["passing_radius"])
        else:
            raise ValueError("invalid reward function:", self.rew_fun)


class CurriculumCallback(BaseCallback):
    def __init__(self, eval_env, threshold: float, eval_freq: int, max_difficulty: int, starting_difficulty: int = 0, verbose: int=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.threshold = threshold
        self.eval_freq = eval_freq
        self.max_difficulty = max_difficulty
        self.current_difficulty = starting_difficulty

    def _on_training_start(self):
        self.logger.record("curriculum/difficulty", self.current_difficulty)

    def _on_step(self) -> bool:
        # Evaluate every 5000 steps
        if self.n_calls % self.eval_freq == 0:
            if self.current_difficulty == self.max_difficulty - 1:
                # Already at max difficulty
                return True

            mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=5, deterministic=True)
            if self.verbose > 0:
                print(f"Eval reward at difficulty {self.current_difficulty}: {mean_reward}")

            # If performance is good, increase difficulty
            if mean_reward > self.threshold and self.current_difficulty < self.max_difficulty - 1:
                self.current_difficulty += 1
                if self.verbose > 0:
                    print(f"Increasing difficulty to {self.current_difficulty}")
                    self.logger.record("curriculum/difficulty", self.current_difficulty)

                    log_dir = self.logger.dir or "./"
                    log_file = os.path.join(log_dir, "curriculum.txt")
                    with open(log_file, "a") as f:
                        f.write(f"Step {self.num_timesteps}: Increased difficulty to {self.current_difficulty} with eval reward {mean_reward}\n")

                # Update training and evaluation environments
                self.training_env.env_method("set_difficulty", self.current_difficulty)
                self.eval_env.set_difficulty(self.current_difficulty)

        return True


class BallerEvalCallback(EvalCallback):
    def __init__(self, eval_env, *args, **kwargs):
        super().__init__(eval_env, *args, **kwargs)
        self.metrics = defaultdict(list)

    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.n_calls % self.eval_freq == 0:
            n_episodes = self.n_eval_episodes
            successes, distances, throw_duration, throw_vel, joint_usage = [], [], [], [], []

            for _ in range(n_episodes):
                obs = self.eval_env.reset()
                done = False

                released_ball, ball_vel = [], []
                joint_use = 0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    joint_use += np.sum(action)
                    obs, reward, done, info = self.eval_env.step(action)

                    info = info[0]
                    released_ball.append(info["released_ball"])
                    ball_vel.append(info["ball_vel_norm"])

                # Extract metrics
                successes.append(info.get("is_success", 0.0))
                distances.append(info.get("distance_to_goal", 0.0))
                joint_usage.append(joint_use)

                released_ball = np.asarray(released_ball)
                indices = np.where(released_ball == 1)[0]
                if indices.size > 0:
                    release_time = indices[0]
                else:
                    release_time = len(released_ball)
                throw_duration.append(release_time)

                throw_vel.append(np.mean(ball_vel[release_time:]) if len(ball_vel) > 0 else 0.0)

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

            if self.logger:
                for key, val in self.metrics.items():
                    self.logger.record(f"eval/{key}", val[-1])

        return result


def train_her():
    env = HERBallerSupervisor(rew_fun="sparse", )
    env = TimeLimit(env, TIME_LIMIT)

    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=f"./models/", name_prefix="baller")
    eval_callback = BallerEvalCallback(env, best_model_save_path="./models/her",
                                 log_path="./logs/her", eval_freq=10_000,
                                 n_eval_episodes=5, deterministic=True,
                                 render=False)
    curriculum_callback = CurriculumCallback(
        eval_env=env,
        threshold=0.5,
        eval_freq=5_000,
        max_difficulty=len(DIFFICULTIES),
        starting_difficulty=0, # len(DIFFICULTIES)
        verbose=1,
    )
    callback = CallbackList([checkpoint_callback, eval_callback, curriculum_callback])

    # Available strategies (cf paper): future, final, episode
    goal_selection_strategy = "final" # equivalent to GoalSelectionStrategy.FUTURE

    model_path = f"./models/baller_100000_steps.zip"
    # numberSteps = 50_000
    # model_path = f"./models/baller_{numberSteps}_steps"

    model_class = SAC  # works also with SAC, DDPG and TD3
    if CONTINUE_TRAINING:
        model = model_class.load(model_path, env=env, tensorboard_log="./logs/her")
    else:
        action_noise = None # OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=0.1 * np.ones(env.action_space.shape[0]))
        # Initialize the model
        model = model_class(
            "MultiInputPolicy",
            env,
            action_noise=action_noise,
            replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy=goal_selection_strategy,
                copy_info_dict=True,
            ),
            learning_starts=TIME_LIMIT,
            tensorboard_log="./logs/her",
            verbose=1,
        )

    model.learn(
        total_timesteps=10_000_000,
        callback=callback,
        reset_num_timesteps=not CONTINUE_TRAINING  # Only reset if new training
    )
    return model


def train_PPO():
    def make_env():
        env = BallerSupervisor(rew_fun="shaped")
        env = TimeLimit(env, TIME_LIMIT)
        # env = Monitor(env)
        return env

    # env = DummyVecEnv([make_env])
    # env = VecNormalize(env, True, norm_obs=False, norm_reward=True)
    env = make_env()

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=f"./models/", name_prefix="baller")
    eval_callback = BallerEvalCallback(env, best_model_save_path="./models/",
                                 log_path="./logs/", eval_freq=10_000,
                                 n_eval_episodes=5, deterministic=True,
                                 render=False)
    curriculum_callback = CurriculumCallback(
        eval_env=env,
        threshold=35,
        eval_freq=5_000,
        max_difficulty=len(DIFFICULTIES),
        starting_difficulty=0,  # len(DIFFICULTIES)
        verbose=1,
    )
    callback = CallbackList([checkpoint_callback, eval_callback, curriculum_callback])

    model_path = f"./models/baller_100000_steps.zip"
    # numberSteps = 50_000
    # model_path = f"./models/baller_{numberSteps}_steps"

    if CONTINUE_TRAINING:
        VecNormalize.load()
        model = TD3.load(model_path, env=env, tensorboard_log="./logs/")
    else:
        action_noise = None # NormalActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=0.1 * np.ones(env.action_space.shape[0]))
        model = SAC("MlpPolicy", env, action_noise=action_noise, tensorboard_log="./logs/", verbose=1)
        # model = PPO("MlpPolicy", env, tensorboard_log="./logs/", verbose=1)

    # Start/Continue training
    model.learn(
        total_timesteps=1_0000_000,
        callback=callback,
        reset_num_timesteps=not CONTINUE_TRAINING  # Only reset if new training
    )

    model.save(f"./models/final")

    return model


def fun(model_class, model_path: str, env_class):
    env = env_class("shaped", False)
    env = TimeLimit(env, TIME_LIMIT)

    model = model_class.load(model_path, env=env, tensorboard_log="./logs/")
    env = model.get_env()

    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)


TRAINING = True
MODEL_TO_LOAD = (PPO, "./models/baller_150000_steps", BallerSupervisor)
# MODEL_TO_LOAD = (SAC, "./models/final_her_sac", HERBallerSupervisor)
CONTINUE_TRAINING = False
TRAIN_PPO = False

TIME_LIMIT = 1_000


if TRAINING:
    if TRAIN_PPO:
        train_PPO()
    else:
        train_her()
else:
    fun(*MODEL_TO_LOAD)