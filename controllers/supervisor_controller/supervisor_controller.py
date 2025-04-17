import gym
import numpy as np
from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from controller import PositionSensor, Motor, Supervisor

JOINT_NAMES = [f"joint{i}" for i in range(1, 6)]
JOINT_SENSOR_NAMES = [f"joint{i}_sensor" for i in range(1, 6)]

JOINT_LIMITS = [
    (-2.792, 2.792),
    (-3.9369, 0.7854),
    (-0.7854, 3.9269),
    (-1.9198, 2.967),
    (-1.7453, 1.7453),
    # (-4.6425, 4.6425),
]

# TODO:
# - rewards
# - parar o robo depois de lançar e só simular bola?

class BallerSupervisor(RobotSupervisorEnv):
    def __init__(self, timestep: int | None=None):
        super().__init__(timestep=timestep)
        self.observation_space = gym.spaces.Box(0, 1, (14,))
        self.action_space = gym.spaces.Box(-1, 1, (6,))

        self.timestep = int(self.getBasicTimeStep())
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.hoop = self.getFromDef("HOOP")
        self.ball = self.getFromDef("BALL")

        self.joints = []
        for joint_name in JOINT_NAMES:
            joint: Motor = self.getDevice(joint_name)
            joint.setPosition(float("inf"))
            joint.setVelocity(0)
            self.joints.append(joint)

        self.joint_sensors = []
        for joint_name in JOINT_SENSOR_NAMES:
            sensor: PositionSensor = self.getDevice(joint_name)
            sensor.enable(self.timestep)
            self.joint_sensors.append(sensor)

        self.was_higher_than_hoop = False
        self.ball_last_vel = np.array([0, 0, 0])

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

        # Ball Acceleration and Velocity
        ball_cur_vel = np.array(self.ball.getVelocity()[:3])
        ball_accel = (ball_cur_vel - self.ball_last_vel) / self.timestep
        obs.extend(ball_accel)
        obs.extend(ball_cur_vel)

        self.ball_last_vel = ball_cur_vel

        return obs

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action=None):
        # Reward is +1 for every step the episode hasn't ended
        return 1

    def is_done(self):
        ball_pos = self.ball.getPosition()

        if self.was_higher_than_hoop and ball_pos[2] < self.hoop.getPosition()[2]:
            # Was high enough
            return True

        if ball_pos[2] <= 0.2:
            # Wasn't high enough
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
        if self.ball.getPosition()[2] >= self.hoop.getPosition()[2]:
            self.was_higher_than_hoop = True

        for i in range(len(self.joints)):
            self.joints[i].setPosition(float("inf"))
            self.joints[i].setVelocity(action[i])

        release_ball = action[-1]
        # TODO: release the ball

    def reset(self):
        self.was_higher_than_hoop = False
        self.ball_last_vel = np.array([0, 0, 0])

        return super().reset()


CONTINUE_TRAINING = False

env = BallerSupervisor()

# Set up callbacks
checkpointCallback = CheckpointCallback(save_freq=50_000, save_path=f"./models/", name_prefix="baller")
callback = CallbackList([checkpointCallback])
model_path = f"./models/final"
# numberSteps = 50_000
# model_path = f"./models/baller_{numberSteps}_steps"

if CONTINUE_TRAINING:
    model = PPO.load(model_path, env=env)
else:
    model = PPO("MlpPolicy", env, verbose=1)

# Start/Continue training
model.learn(
    total_timesteps=1_000_000,
    callback=callback,
    reset_num_timesteps=not CONTINUE_TRAINING  # Only reset if new training
)

model.save(f"./models/final")

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
        observation = env.reset()