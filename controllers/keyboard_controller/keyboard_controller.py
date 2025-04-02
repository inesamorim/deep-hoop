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

Press the corresponding keys to move the robotic arm.
Joint positions will be displayed in the console.
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

# Define key bindings for movement
key_bindings = {
    'Q': (0, 0.05), 'A': (0, -0.05),
    'W': (1, 0.05), 'S': (1, -0.05),
    'E': (2, 0.05), 'D': (2, -0.05),
    'R': (3, 0.05), 'F': (3, -0.05),
    'T': (4, 0.05), 'G': (4, -0.05)
    #'Y': (5, 0.05), 'H': (5, -0.05)
}

while robot.step(timestep) != -1:
    key = keyboard.getKey()
    if key == -1:
        continue

    key_char = chr(key).upper()
    if key_char in key_bindings:
        joint_index, delta = key_bindings[key_char]
        min_limit, max_limit = joint_limits[joint_index]

        # Update position while respecting limits
        new_position = joint_positions[joint_index] + delta
        if min_limit is None or max_limit is None or (min_limit <= new_position <= max_limit):
            joint_positions[joint_index] = new_position
            joints[joint_index].setPosition(new_position)

    # Debugging: print current joint positions
    #print("Joint positions:", joint_positions)
