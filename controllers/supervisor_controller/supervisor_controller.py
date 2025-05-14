import gym
import numpy as np
from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from gym.wrappers import TimeLimit

from controller import PositionSensor, Motor, Supervisor

JOINT_NAMES = [f"joint{i}" for i in range(1, 4)]
JOINT_SENSOR_NAMES = [f"joint{i}_sensor" for i in range(1, 4)]
HAND_NAMES = [f"finger_{i}_joint_1" for i in [1, 2, "middle"]]
HAND_SENSOR_NAMES = [f"finger_{i}_joint_1_sensor" for i in [1, 2, "middle"]]

# TODO:
# - parar o robo depois de lançar e só simular bola?

class BallerSupervisor(RobotSupervisorEnv):
    def __init__(self, timestep: int | None=None):
        super().__init__(timestep=timestep)
        # self.observation_space = gym.spaces.Box(-3, 3, (14,))
        # self.action_space = gym.spaces.Box(-1, 1, (6,))
        low = np.array([
            -10.0, -10.0, -10.0, # rel_pos
            -2.792, -3.9369, -0.7854, # -1.9198, -1.7453, # joints
            0.0495, 0.0495, 0.0495, # hand joints
            -10.0, -10.0, -10.0, # ball acceleration
            -10.0, -10.0, -10.0, # ball velocity
        ])
        high = np.array([
            10.0, 10.0, 10.0,
            2.792, 0.7854, 3.9269, # 2.967, 1.7453,
            1.2218, 1.2218, 1.2218,
            10.0, 10.0, 10.0,
            10.0, 10.0, 10.0,
        ])
        self.observation_space = gym.spaces.Box(low, high)
        low = np.array([-4] * 4)
        high = np.array([4, 0, 0, 0])
        self.action_space = gym.spaces.Box(low, high)

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

        # Get the outer radius (distance from center to tube middle)
        outer_radius = self.hoop.getField("majorRadius").getSFFloat()
        # Get the inner radius (tube thickness)
        inner_radius = self.hoop.getField("minorRadius").getSFFloat()
        self.passing_radius = outer_radius-inner_radius

        self.steps_per_episode = 200  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved


    def get_observations(self):
        obs = []

        ball_pos = self.ball.getPosition()
        hoop_pos = self.hoop.getPosition()

        # Relative pos
        relative_pos = [hoop_pos[i] - ball_pos[i] for i in range(len(hoop_pos))]
        obs.extend(relative_pos)

        # Joint angles
        for sensor in self.joint_sensors:
            obs.append(sensor.getValue())

        # Hand angles
        for sensor in self.hand_sensors:
            obs.append(sensor.getValue())

        # Ball Acceleration and Velocity
        ball_cur_vel = np.array(self.ball.getVelocity()[:3])
        ball_accel = (ball_cur_vel - self.ball_last_vel) / self.timestep
        obs.extend(ball_accel)
        obs.extend(ball_cur_vel)

        # Update ball min distance
        dist = np.linalg.norm(relative_pos)
        if dist < self.closest_dist:
            self.closest_dist = dist # TODO: see if there is a way to improve this
            self.closest_pos = ball_pos

        return obs

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action=None):
        return self.rew2()

    def rew2(self):
        ball_pos = np.asarray(self.ball.getPosition())
        ball_vel = np.asarray(self.ball.getVelocity()[:3])
        hoop_pos = np.asarray(self.hoop.getPosition())
        rel_pos = hoop_pos - ball_pos

        # Delta-distance
        d = np.linalg.norm(rel_pos)
        r_dd = self.ball_last_dist - d

        # Raw velocity toward hoop
        dir_vec = rel_pos / (d + 1e-6)
        r_vel = np.dot(ball_vel, dir_vec)

        # Time penalty
        r_time = -0.01

        reward = 10 * r_dd + 0.5 * r_vel + r_time + self.passed_hoop
        print(f"{r_dd=}, {r_vel=}, {reward=}")
        return reward

    def rew0(self):
        return self.released_ball * (999 * self.passed_hoop - self.closest_dist)

    def rew1(self):
        # Move a bola lentamente da direção do hoop para max r_velo

        ball_pos = np.array(self.ball.getPosition())
        hoop_pos = np.array(self.hoop.getPosition())
        rel_pos = hoop_pos - ball_pos
        dist = np.linalg.norm(rel_pos)

        # how well the ball is heading toward the hoop
        r_velo = np.dot(self.ball_last_vel, rel_pos) / (
                np.linalg.norm(self.ball_last_vel) * dist + 1e-6)

        # Time penalty = move fast?
        r_time = -0.01

        # print("dist:", foo)
        # print("Reward:", 999 * self.passed_hoop + (4 - foo) / 4)
        print("r_velo:", r_velo)
        print("dist:", - self.closest_dist / 4)
        print("Reward:", 999 * self.passed_hoop - dist / 4 + r_velo + r_time)
        return 999 * self.passed_hoop - dist / 4 + 0.9 * r_velo + r_time

    def is_ball_passing(self):
        ball_center = self.ball.getPosition() #returns x,y,z coordenates of center
        return self._is_ball_passing(ball_center, self.passing_center)

    def _is_ball_passing(self, ball_center: list[float], hoop_center: list[float]):
        # Compute distance between ball center and passing zone center (XY distance)
        dist = np.sqrt((ball_center[0] - hoop_center[0]) ** 2 +
                       (ball_center[1] - hoop_center[1]) ** 2)

        # Ball passes through if it's within the passing radius and at the correct Z level
        return dist <= self.passing_radius and abs(ball_center[2] - hoop_center[2]) < 0.05  # Allowing small Z tolerance

    def is_done(self):
        ball_pos = self.ball.getPosition()

        if self.is_ball_passing():
            # Hit the hoop
            return True

        if self.was_higher_than_hoop and ball_pos[2] < self.hoop.getPosition()[2]:
            # Missed hoop
            return True

        # ball_vel = self.ball.getVelocity()
        # if not self.was_higher_than_hoop and self.released_ball and ball_vel[2] < 0:
        #     # Wasnt high enough
        #     return True

        if ball_pos[2] <= 0.2:
            # Hit ground
            return True

        return False

    def solved(self):
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 195.0:  # Last 100 episodes' scores average value
                return True
        return False

    def get_info(self):
        return {}

    def render(self, mode='human'):
        pass

    def apply_action(self, action):
        print("Action:", action)
        if self.ball.getPosition()[2] >= self.hoop.getPosition()[2]:
            self.was_higher_than_hoop = True

        # Check if released ball
        ball_pos = self.ball.getPosition()
        hand_pos = self.getFromDef("HAND").getPosition()
        if 0.2 < np.sqrt((ball_pos[0] - hand_pos[0]) ** 2 +
                    (ball_pos[1] - hand_pos[1]) ** 2 +
                    (ball_pos[2] - hand_pos[2]) ** 2):
            self.released_ball = True

        # Update ball velocity
        # yes this must be done here
        self.ball_last_vel = np.asarray(self.ball.getVelocity()[:3])
        ball_pos = np.asarray(self.ball.getPosition())
        hoop_pos = np.asarray(self.hoop.getPosition())
        rel_pos = hoop_pos - ball_pos
        self.ball_last_dist = np.linalg.norm(rel_pos)

        # Set joint velocities
        for i in range(len(self.joints)):
            self.joints[i].setPosition(float("inf"))
            self.joints[i].setVelocity(action[i])

        # Release ball
        release_ball = action[-1]
        for i in range(len(self.hand)):
            self.hand[i].setPosition(float("inf"))
            self.hand[i].setVelocity(release_ball)

    def reset(self):
        hoop_pos = np.asarray(self.hoop.getPosition())
        ball_pos = np.asarray(self.ball.getPosition())
        dist = np.linalg.norm(hoop_pos - ball_pos)

        self.was_higher_than_hoop = False
        self.ball_last_vel = np.array([0, 0, 0])
        self.ball_last_dist = dist

        self.released_ball = False # if robot is still holding the ball
        self.passed_hoop = False # ball has passed the hoop
        self.closest_dist = 99999
        self.closest_pos = ball_pos
        self.passing_center = hoop_pos

        return super().reset()

class HERBallerSupervisor(BallerSupervisor):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict({
            "observation": self.observation_space,
            "achieved_goal": gym.spaces.Box(0, 1, (3,)),
            "desired_goal": gym.spaces.Box(0, 1, (3,)),
        })

    def get_observations(self):
        obs_orig = super().get_observations()

        # if self.released_ball:
        #     while super(Supervisor, self).step(self.timestep) != -1:
        #         if self.is_done():
        #             break

        return {
            "observation": obs_orig,
            "achieved_goal": self.ball.getPosition(),
            "desired_goal": self.hoop.getPosition(),
        }

    def get_default_observation(self):
        return {
            "observation": [0.0 for _ in range(self.observation_space["observation"].shape[0])],
            "achieved_goal": self.ball.getPosition(),
            "desired_goal": self.hoop.getPosition(),
        }

    def is_done(self):
        return super().is_done()
        if not self.released_ball:
            return False

        while super(Supervisor, self).step(self.timestep) != -1:
            ball_pos = self.ball.getPosition()

            if self.is_ball_passing():
                # Hit the hoop
                return True

            if self.was_higher_than_hoop and ball_pos[2] < self.hoop.getPosition()[2]:
                # Missed hoop
                return True

            # ball_vel = self.ball.getVelocity()
            # if not self.was_higher_than_hoop and self.released_ball and ball_vel[2] < 0:
            #     # Wasnt high enough
            #     return True

            if ball_pos[2] <= 0.2:
                # Hit ground
                return True

        return True

    def get_info(self):
        return {
            "ball_last_dist": self.ball_last_dist,
            "ball_vel": self.ball.getVelocity()[:3],
        }

    def get_reward(self, action=None):
        ball_pos = np.asarray(self.ball.getPosition())
        ball_vel = np.asarray(self.ball.getVelocity()[:3])
        hoop_pos = np.asarray(self.hoop.getPosition())
        dist = -np.linalg.norm(hoop_pos - ball_pos)

        # if self.released_ball:
        #     while super(Supervisor, self).step(self.timestep) != -1:
        #         if self.is_done():
        #             break

        # if self.is_done():
        #     return -dist

        return int(self.is_ball_passing())
        # return self.rew(ball_pos, ball_vel, hoop_pos, self.ball_last_dist)

    def rew(
        self,
        ball_pos: list[float],
        ball_vel: list[float],
        hoop_pos: list[float],
        ball_last_dist: float,
    ):
        ball_pos = np.asarray(ball_pos)
        ball_vel = np.asarray(ball_vel)
        hoop_pos = np.asarray(hoop_pos)
        rel_pos = hoop_pos - ball_pos

        # Delta-distance
        d = np.linalg.norm(rel_pos)
        r_dd = 0 # ball_last_dist - d

        # Raw velocity toward hoop
        dir_vec = rel_pos / (d + 1e-6)
        r_vel = np.dot(ball_vel, dir_vec)

        # Time penalty
        r_time = -0.01

        # Success boost
        passed_hoop = self._is_ball_passing(ball_pos, hoop_pos)

        reward = 10 * r_dd + 0.5 * r_vel + r_time + passed_hoop
        print(f"{r_dd=}, {r_vel=}, {reward=}")
        return reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        return np.asarray(
            [int(self._is_ball_passing(ball, hoop))
             for ball, hoop in zip(achieved_goal, desired_goal)]
        )
        # return np.asarray(
        #     [self.rew(
        #         ball,
        #         info["ball_vel"],
        #         hoop,
        #         info["ball_last_dist"],
        #     ) for ball, hoop, info in zip(achieved_goal, desired_goal, info)]
        # )


def train_her():
    from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3

    env = HERBallerSupervisor()
    env = TimeLimit(env, TIME_LIMIT)

    checkpointCallback = CheckpointCallback(save_freq=50_000, save_path=f"./models/", name_prefix="baller")
    eval_callback = EvalCallback(env, best_model_save_path="./models/her",
                                 log_path="./logs/her", eval_freq=10_000,
                                 n_eval_episodes=5, deterministic=True,
                                 render=False)
    callback = CallbackList([checkpointCallback, eval_callback])

    # Available strategies (cf paper): future, final, episode
    goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

    model_path = f"./models/baller_100000_steps.zip"
    # numberSteps = 50_000
    # model_path = f"./models/baller_{numberSteps}_steps"

    from stable_baselines3 import SAC
    model_class = SAC  # works also with SAC, DDPG and TD3
    if CONTINUE_TRAINING:
        model = model_class.load(model_path, env=env, tensorboard_log="./logs/her")
    else:
        from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape[0]),
                                         sigma=0.1 * np.ones(env.action_space.shape[0]))
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
    env = BallerSupervisor()
    env = TimeLimit(env, TIME_LIMIT)

    # Set up callbacks
    checkpointCallback = CheckpointCallback(save_freq=50_000, save_path=f"./models/", name_prefix="baller")
    callback = CallbackList([checkpointCallback])
    model_path = f"./models/baller_100000_steps.zip"
    # numberSteps = 50_000
    # model_path = f"./models/baller_{numberSteps}_steps"

    if CONTINUE_TRAINING:
        model = TD3.load(model_path, env=env, tensorboard_log="./logs/")
    else:
        from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
        action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=0.1 * np.ones(env.action_space.shape[0]))
        model = DDPG("MlpPolicy", env, action_noise=action_noise, tensorboard_log="./logs/", verbose=1)

    # Start/Continue training
    model.learn(
        total_timesteps=1_0000_000,
        callback=callback,
        reset_num_timesteps=not CONTINUE_TRAINING  # Only reset if new training
    )

    model.save(f"./models/final")

    return model


CONTINUE_TRAINING = False
TRAIN_PPO = False

TIME_LIMIT = 1_000

if TRAIN_PPO:
    train_PPO()
else:
    train_her()

exit()


env = BallerSupervisor()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

exit()

env = BallerSupervisor()
print(env.get_observations())
while env.step([1, 0,0,0,0]) != -1:
    print(env.get_observations())




solved = False
episode_count = 0
episode_limit = 2000
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and episode_count < episode_limit:
    observation = env.reset()  # Reset robot and get starting observation
    env.episode_score = 0

    for step in range(env.steps_per_episode):
        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        selected_action, action_prob = agent.work(observation, type_="selectAction")
        # Step the supervisor to get the current selected_action's reward, the new observation and whether we reached
        # the done condition
        new_observation, reward, done, info = env.step([selected_action])

        # Save the current state transition in agent's memory
        trans = Transition(observation, selected_action, action_prob, reward, new_observation)
        agent.store_transition(trans)

        if done:
            # Save the episode's score
            env.episode_score_list.append(env.episode_score)
            agent.train_step(batch_size=step + 1)
            solved = env.solved()  # Check whether the task is solved
            break

        env.episode_score += reward  # Accumulate episode reward
        observation = new_observation  # observation for next step is current step's new_observation

    print("Episode #", episode_count, "score:", env.episode_score)
    episode_count += 1  # Increment episode counter

if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")

observation = env.reset()
env.episode_score = 0.0
while True:
    selected_action, action_prob = agent.work(observation, type_="selectActionMax")
    observation, _, done, _ = env.step([selected_action])
    if done:
        pass
