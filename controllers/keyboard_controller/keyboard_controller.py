from controller import Robot, Keyboard

# Create robot instance
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get keyboard instance
keyboard = Keyboard()
keyboard.enable(timestep)

print("""
PUMA 560 Keyboard Control:
--------------------------
Use the following keys to control each joint:
- Joint 1:  Q (increase) / A (decrease)
- Joint 2:  W (increase) / S (decrease)
- Joint 3:  E (increase) / D (decrease)
- Joint 4:  R (increase) / F (decrease)
- Joint 5:  T (increase) / G (decrease)

Robotiq3f Gripper Control:
------------------------
- Finger 1:     U (close) / J (open)
- Finger 2:     I (close) / K (open)
- Middle Finger: O (close) / L (open)

Check the console to see current joint and finger motor positions.
""")

# Define joints and their limits
joints = [
    robot.getDevice("joint1"),
    robot.getDevice("joint2"),
    robot.getDevice("joint3"),
    robot.getDevice("joint4"),
    robot.getDevice("joint5"),
    #robot.getDevice("joint6")
]

joint_limits = [
    (-2.792, 2.792),
    (-3.9369, 0.7854),
    (-0.7854, 3.9269),
    (-1.9198, 2.967),
    (-1.7453, 1.7453)
    #(-4.6425, 4.6425)
]

# Initialize joint positions
joint_positions = [0] * len(joints)

# Define Robotiq motors
finger_motors = {
    "finger_1_joint_1": robot.getDevice("finger_1_joint_1"),
    "finger_2_joint_1": robot.getDevice("finger_2_joint_1"),
    "finger_middle_joint_1": robot.getDevice("finger_middle_joint_1"),
    "finger_1_link_0": robot.getDevice("finger_1_link_0"),
    "finger_2_link_0": robot.getDevice("finger_2_link_0"),
    "finger_middle_link_1": robot.getDevice("finger_middle_link_1"),
    "finger_1_link_0": robot.getDevice("finger_1_link_0"),
    "finger_2_link_0": robot.getDevice("finger_2_link_0"),
    "finger_middle_link_1": robot.getDevice("finger_middle_link_1")
}

finger_positions = {name: 0.0 for name in finger_motors}


finger_limits = {
    "finger_1_joint_1": (0.0495, 1.2218),
    "finger_2_joint_1": (0.0495, 1.2218),
    "finger_middle_joint_1": (0.0495, 1.2218)
}


# Define key bindings for movement
key_bindings = {
    'Q': ("puma", 0, 0.05), 'A': ("puma", 0, -0.05),
    'W': ("puma", 1, 0.05), 'S': ("puma", 1, -0.05),
    'E': ("puma", 2, 0.05), 'D': ("puma", 2, -0.05),
    'R': ("puma", 3, 0.05), 'F': ("puma", 3, -0.05),
    'T': ("puma", 4, 0.05), 'G': ("puma", 4, -0.05),
    'U': ("gripper", "finger_1_joint_1", 0.05), 'J': ("gripper", "finger_1_joint_1", -0.05),
    'I': ("gripper", "finger_2_joint_1", 0.05), 'K': ("gripper", "finger_2_joint_1", -0.05),
    'O': ("gripper", "finger_middle_joint_1", 0.05), 'L': ("gripper", "finger_middle_joint_1", -0.05),
    'Z': ("gripper_all", None, 0.05),  # Close all fingers
    'X': ("gripper_all", None, -0.05)  # Open all fingers
}


while robot.step(timestep) != -1:
    key = keyboard.getKey()
    if key == -1:
        continue

    key_char = chr(key).upper()
    if key_char in key_bindings:
        control_type, index_or_name, delta = key_bindings[key_char]

        if control_type == "puma":
            min_limit, max_limit = joint_limits[index_or_name]
            new_pos = joint_positions[index_or_name] + delta
            if min_limit <= new_pos <= max_limit:
                joint_positions[index_or_name] = new_pos
                joints[index_or_name].setPosition(new_pos)

        elif control_type == "gripper":
            min_limit, max_limit = finger_limits[index_or_name]
            new_pos = finger_positions[index_or_name] + delta
            if min_limit <= new_pos <= max_limit:
                finger_positions[index_or_name] = new_pos
                finger_motors[index_or_name].setPosition(new_pos)

        elif control_type == "gripper_all":
            for name in finger_positions:
                min_limit, max_limit = finger_limits[name]
                new_pos = finger_positions[name] + delta
                if min_limit <= new_pos <= max_limit:
                    finger_positions[name] = new_pos
                    finger_motors[name].setPosition(new_pos)

    print("Joint positions:", joint_positions)
    print("Gripper positions:", finger_positions)


    # Debugging: print current joint positions
    #print("Joint positions:", joint_positions)
