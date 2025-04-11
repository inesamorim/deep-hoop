from deepbots.robots.controllers.csv_robot import CSVRobot
from controller import Motor


class BallerRobot(CSVRobot):
    def __init__(self):
        super().__init__()
        self.joints = [
            self.getDevice("joint1"),
            self.getDevice("joint2"),
            self.getDevice("joint3"),
            self.getDevice("joint4"),
            self.getDevice("joint5"),
        ]

    def create_message(self):
        # Read the sensor value, convert to string and save it in a list
        message = []
        return message

    def use_message_data(self, message):
        action = int(message[0])  # Convert the string message into an action integer



# Create the robot controller object and run it
robot_controller = BallerRobot()
robot_controller.run()  # Run method is implemented by the framework, just need to call it