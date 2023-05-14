"""Action state."""

from utils.motion_library import MotionLibrary
from utils.gait_manager import GaitManager
import numpy as np

class Action():
    def __init__(self, robot, time_step):
        # self.joints = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RHipYawPitch", "RHipRoll",
        # "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", 
        # "LElbowRoll", "LHipYawPitch", "LHipRoll",
        # "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll"] // will use it later
        self.joints = ["LHipYawPitch","LHipRoll","LHipPitch","LKneePitch","LAnklePitch","LAnkleRoll","RHipYawPitch","RHipRoll","RHipPitch","RKneePitch","RAnklePitch","RAnkleRoll"]

        self.actions = ["Forwards", "Backwards", "SideStepLeft", "SideStepRight", "TurnLeft20", "TurnRight20", "Shove"]
        robot.library = MotionLibrary()
        robot.library.add('Shove', './Shove.motion', loop=False)
        self.robot = robot
        self.time_step = time_step
        self.gait_manager = GaitManager(self.robot, self.time_step)
        self.last_action = -1

    def execute_action(self, action):
        # self.robot.library.play(self.actions[action])
        # self.robot.library.stop(self.actions[action])
        self.robot.getDevice('HeadPitch').setPosition(0.3)

        desired_radius = action[0]*0.1
        heading_angle = 0.0
        self.gait_manager.update_theta()
        self.gait_manager.command_to_motors(desired_radius=desired_radius, heading_angle=heading_angle)
        # self.robot.library.play(self.actions[action])
        
        # while not self.robot.library.isOver(self.actions[self.last_action]):
        #     self.robot.step(time_step)

        ##################### Discarded ###########################################
        # This approach has been discarded as it referes to the two bandits problem

        # if action == self.last_action:
        #     if self.robot.library.isOver(self.actions[action]):
        #         self.robot.library.play(self.actions[action])
        # else:
        #     if (self.last_action != -1):
        #         if not self.robot.library.isOver(self.actions[self.last_action]):
        #             self.robot.library.stop(self.actions[self.last_action])
        #         self.robot.library.play(self.actions[action])
        #     else:
        #         self.robot.library.play(self.actions[action])

        ###########################################################################
                
        # self.last_action = action
        # for index, joint in enumerate(self.joints):
        #     self.robot.getDevice(joint).setPosition(float('inf'))
        #     self.robot.getDevice(joint).setVelocity(action[index])

    def stop_last_action(self):
        self.robot.library.stop(self.actions[self.last_action])

    def reset_gait_manager(self):
        del self.gait_manager
        self.gait_manager = GaitManager(self.robot, self.time_step)