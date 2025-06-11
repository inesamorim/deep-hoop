from controller import PositionSensor, Motor, Supervisor
import gym
import numpy as np
from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from callbacks import DIFFICULTIES


# Define names of robot devices
JOINT_NAMES = [f"joint{i}" for i in range(1, 4)]
JOINT_SENSOR_NAMES = [f"joint{i}_sensor" for i in range(1, 4)]
HAND_NAMES = [f"finger_{i}_joint_1" for i in [1, 2, "middle"]]
HAND_SENSOR_NAMES = [f"finger_{i}_joint_1_sensor" for i in [1, 2, "middle"]]

# Range of values for each action
ACTION_SCALE = [
    (-4, 4),
    (-4, 0),
    (-4, 0),
    (-4, 0),
]


# Dense reward based on how directly and quickly the ball is moving toward the hoop,
# with a large bonus if the ball passes through the hoop.
def rew_shaped(
    ball_pos: list[float],
    ball_vel: list[float],
    hoop_pos: list[float],
    passing_radius: float,
) -> float:
    ball_pos = np.asarray(ball_pos)
    ball_vel = np.asarray(ball_vel)
    hoop_pos = np.asarray(hoop_pos)
    rel_pos = hoop_pos - ball_pos

    # Raw velocity toward hoop
    d = np.linalg.norm(rel_pos)
    dir_vec = rel_pos / (d + 1e-6)
    r_vel = np.dot(ball_vel, dir_vec)

    # Time penalty
    r_time = -0.01

    # Success boost
    passed_hoop = is_ball_passing(ball_pos, hoop_pos, passing_radius)

    reward = 0.5 * r_vel + r_time + 10 * passed_hoop
    return reward


# 1 if success else 0
def rew_sparse(
    ball_pos: list[float],
    hoop_pos: list[float],
    passing_radius: float,
) -> float:
    return 1 if is_ball_passing(ball_pos, hoop_pos, passing_radius) else 0


# Gives a reward proportional to the distance between the ball and the hoop at the end of the episode
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


# If the episode has ended
def is_done(
    ball_pos: list[float],
    ball_vel: list[float],
    hoop_pos: list[float],
    passing_radius: float,
) -> bool:
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


# If ball is passing though the hoop
def is_ball_passing(
    ball_center: list[float], hoop_center: list[float], passing_radius: float
) -> bool:
    # Compute distance between ball center and passing zone center (XY distance)
    dist = np.sqrt(
        (ball_center[0] - hoop_center[0]) ** 2 + (ball_center[1] - hoop_center[1]) ** 2
    )

    # Ball passes through if it's within the passing radius and at the correct Z level
    return (
        dist <= passing_radius
        and abs(ball_center[2] - hoop_center[2]) < 0.05  # Allowing small Z tolerance
    )


# From [a, b] to [-1, 1]
def to_unit(x: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    return 2 * (x - a) / (b - a) - 1


# From [-1, 1] to [a, b]
def from_unit(x: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    return ((x + 1) / 2) * (b - a) + a


# Custom reinforcement learning environment where a robot learns to throw a ball through a hoop
# It manages robot joints and sensors, sets up observation and action spaces,
# computes rewards based on different strategies, and resets the environment with varying difficulty.
class BallerSupervisor(RobotSupervisorEnv):
    def __init__(self, rew_fun: str, timestep: int | None = None):
        super().__init__(timestep=timestep)

        self.rew_fun = rew_fun
        self.cur_difficulty = 0

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(15,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.timestep = int(self.getBasicTimeStep())
        self.robot = (
            self.getSelf()
        )  # Grab the robot reference from the supervisor to access various robot methods
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

    def set_difficulty(self, difficulty: int):
        self.cur_difficulty = difficulty

    def get_observations(self):
        obs = []

        ball_pos = np.asarray(self.ball.getPosition())
        hoop_pos = np.asarray(self.hoop.getPosition())

        # Relative pos
        relative_pos = hoop_pos - ball_pos
        relative_pos = to_unit(relative_pos, -5, 5)  # normalize
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

        # print("Obs:", obs)
        return obs

    def get_default_observation(self):
        return self.get_observations()

    def get_reward(self, action=None):
        ball_pos = np.asarray(self.ball.getPosition())
        ball_vel = np.asarray(self.ball.getVelocity()[:3])
        hoop_pos = np.asarray(self.hoop.getPosition())

        if self.rew_fun == "shaped":
            return rew_shaped(ball_pos, ball_vel, hoop_pos, self.passing_radius)
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
            "ball_pos": ball_pos,
        }

    def render(self, mode="human"):
        pass

    def apply_action(self, action):
        # print("Action:", action)

        ball_pos = np.asarray(self.ball.getPosition())
        hand_pos = np.asarray(self.getFromDef("HAND").getPosition())

        # Check if released ball
        if np.linalg.norm(ball_pos - hand_pos) > 0.2:
            self.released_ball = True

        # Update ball velocity
        # yes this must be done here
        self.ball_last_vel = np.asarray(self.ball.getVelocity()[:3])

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
        self.ball_last_vel = np.array([0, 0, 0])

        self.released_ball = False  # if robot is still holding the ball
        self.passed_hoop = False  # ball has passed the hoop

        # Reset simulation
        obs = super().reset()

        # Set hoop pos
        difficulty = DIFFICULTIES[self.cur_difficulty]
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

        # Get the outer radius (distance from center to tube middle)
        outer_radius = self.hoop.getField("majorRadius").getSFFloat()
        # Get the inner radius (tube thickness)
        inner_radius = self.hoop.getField("minorRadius").getSFFloat()
        self.passing_radius = outer_radius - inner_radius

        return obs


# Extends BallerSupervisor to support Hindsight Experience Replay (HER)
# It modifies the observation space to include achieved_goal and desired_goal
# and provides a custom compute_reward method for goal-based learning.
class HERBallerSupervisor(BallerSupervisor):
    def __init__(self, rew_fun: str):
        super().__init__(rew_fun=rew_fun)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": self.observation_space,
                "achieved_goal": gym.spaces.Box(0, 1, (3,)),
                "desired_goal": gym.spaces.Box(0, 1, (3,)),
            }
        )

    def get_observations(self):
        obs_orig = super().get_observations()
        return {
            "observation": obs_orig,
            "achieved_goal": self.ball.getPosition(),
            "desired_goal": self.hoop.getPosition(),
        }

    def get_info(self):
        return super().get_info() | {
            "ball_vel": self.ball.getVelocity()[:3],
            "passing_radius": self.passing_radius,
        }

    def get_reward(self, action=None):
        return super().get_reward(action)

    # Determines the way HER calculates rewards of relabeled goals
    def compute_reward(self, achieved_goal, desired_goal, info):
        return np.asarray(
            [
                self._compute_reward(ball, hoop, info)
                for ball, hoop, info in zip(achieved_goal, desired_goal, info)
            ]
        )

    def _compute_reward(self, ball_pos: list[float], hoop_pos: list[float], info: dict):
        if self.rew_fun == "shaped":
            raise NotImplementedError(
                "im lazy and we probably are not going to use 'shaped' with HER"
            )
        elif self.rew_fun == "sparse":
            return rew_sparse(ball_pos, hoop_pos, info["passing_radius"])
        elif self.rew_fun == "sparse_dist":
            return rew_sparse_dist(
                ball_pos, info["ball_vel"], hoop_pos, info["passing_radius"]
            )
        else:
            raise ValueError("invalid reward function:", self.rew_fun)
