"""Observation state."""

#from utils.camera import Camera
from controller import Supervisor
from utils.fall_detection import FallDetection
from utils.image_processing import ImageProcessing
from utils.camera import Camera
import numpy as np
import math
from pytransform3d import transformations as tr
from pytransform3d.rotations import quaternion_from_matrix, euler_from_quaternion
from pytransform3d.transform_manager import TransformManager
import cv2
#from utils.image_processing import ImageProcessing

# The idea is to eventually swap to conv network using only images as state

class Observation():
    def __init__(self, wrestler):
        self.robot = Supervisor.getFromDef(Supervisor, 'WRESTLER_RED').getFromProtoDef('HEAD_SLOT')
        self.oponent = Supervisor.getFromDef(Supervisor, 'WRESTLER_BLUE').getFromProtoDef('HEAD_SLOT')
        self.camera = Camera(wrestler, 'CameraTop')
        self.tm = TransformManager()
        self.image_processing = ImageProcessing()
        self.wrestler = wrestler
        # self.joints_sensors = ["LHipYawPitchS","LHipRollS","LHipPitchS","LKneePitchS","LAnklePitchS","LAnkleRollS","RHipYawPitchS","RHipRollS","RHipPitchS","RKneePitchS","RAnklePitchS","RAnkleRollS"]
        # for joint_sensor in self.joints_sensors:
        #     self.wrestler.getDevice(joint_sensor).enable(30)

    # def oponent_observation(self):
    #     size, y, x = ImageProcessing.locate_opponent(self.get_image())
    #     print(size)
    #     return self.normalize_oponent_position(y, x, self.camera.getHeight(), self.camera.getWidth())

    # def normalize_oponent_position(self, y, x, height, width): # [-1.0, 1.0]
    #     if (x == None or y == None):
    #         return [0.0, 0.0, 0.0]
    #     x_obs = x * 2 / width -1
    #     y_obs = y * 2 / height -1 
    #     return [1.0, x_obs, y_obs] #visible, x_pos, y_pos

    def get_observation_image(self):
        img = self.camera.get_image()
        
        # those spikes are then smoothed out using a Gaussian blur to get blurry blobs
        # we apply a threshold to get a binary image of potential robot locations
        #laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=3)

        # blur = cv2.GaussianBlur(laplacian, (0, 0), 2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (84, 84))
        
        return resized
    
    def image_to_predict(self):
        img = self.camera.get_image()
        
        # those spikes are then smoothed out using a Gaussian blur to get blurry blobs
        # we apply a threshold to get a binary image of potential robot locations
        #laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=3)

        # blur = cv2.GaussianBlur(laplacian, (0, 0), 2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (84, 84))
        array_image = np.expand_dims(resized, axis=2)
        return array_image


    def get_position_state(self):
        position, oponent_position = self.get_robots_position()
        orientation, oponent_orientation = self.get_robots_orientation()

        world_to_robot = tr.transform_from_pq(np.hstack((np.array([position[0], position[1], position[2]]),
         np.array(quaternion_from_matrix(np.reshape(orientation, (3, 3)))))))

        world_to_oponent = tr.transform_from_pq(np.hstack((np.array([oponent_position[0], oponent_position[1], oponent_position[2]]),
         np.array(quaternion_from_matrix(np.reshape(oponent_orientation, (3, 3)))))))

        self.tm.add_transform("oponent", "world", world_to_robot)
        self.tm.add_transform("robot", "world", world_to_oponent)

        p = self.mat_to_position(self.tm.get_transform("robot", "oponent"))

        #o = self.mat_
        
        if position[2] > 1.0: position[2] = 1.0
        if oponent_position[2] > 1.0: oponent_position[2] = 1.0
        
        yaw_2_oponent = self.yaw_to_oponent(p[0], p[1])

        yaw_normalized = yaw_2_oponent / np.pi
        
        obs = [position[0], position[1], position[2], oponent_position[0], oponent_position[1], oponent_position[2], p[0], p[1], yaw_normalized]

        return obs

    def get_observation_state(self):
        
        return self.get_position_state()

    def mat_to_position(self, m):
        p = m[:3,3]
        return p
    
    def mat_to_yaw(self, m):
        r = m[:3, :3]
        yaw = np.arctan2(r[1,0], r[0,0])
        return yaw

    def get_robots_position(self):
        position = self.robot.getPosition()
        oponent_position = self.oponent.getPosition()
        return position, oponent_position
    
    def get_robots_orientation(self):
        position = self.robot.getOrientation()
        oponent_position = self.oponent.getOrientation()
        return position, oponent_position
    
    def yaw_to_oponent(self, relative_x, relative_y):
        angle = np.arctan2(relative_y, relative_x)
        return angle
    
    def get_joint_states(self):
        joint_positions = []
        for joint_sensor in self.joints_sensors:
            joint_positions.append(self.wrestler.getDevice(joint_sensor).getValue())
        
        normalized_joints = self.normalize_joint_positions(joint_positions)
        return np.array(normalized_joints)
    
    def normalize_joint_positions(self, joint_positions):
        # Define joint limits
        limits = {
            "LHipYawPitch": {"high": 0.7408, "low": -1.1453},
            "LHipRoll": {"high": 0.7904, "low": -0.3794},
            "LHipPitch": {"high": 0.4840, "low": -1.7739},
            "LKneePitch": {"high": 2.1125, "low": -0.0923},
            "LAnklePitch": {"high": 0.9227, "low": -1.1895},
            "LAnkleRoll": {"high": 0.7690, "low": -0.3978},
            "RHipYawPitch": {"high": 0.7408, "low": -1.1453},
            "RHipRoll": {"high": 0.4147, "low": -0.7383},
            "RHipPitch": {"high": 0.4856, "low": -1.7723},
            "RKneePitch": {"high": 2.1201, "low": -0.1030},
            "RAnklePitch": {"high": 0.9320, "low": -1.1864},
            "RAnkleRoll": {"high": 0.3886, "low": -1.1864}
        }

        # Normalize joint positions
        normalized_joints = []
        for i, joint_name in enumerate(limits.keys()):
            # Get the joint position value and limits
            pos = joint_positions[i]
            high = limits[joint_name]["high"]
            low = limits[joint_name]["low"]

            # Normalize the joint position
            normalized_pos = (2 * (pos - low) / (high - low)) - 1
            normalized_pos = np.clip(normalized_pos, -1.0, 1.0)
            # Append the normalized joint position to the output list
            normalized_joints.append(normalized_pos)

        return normalized_joints
    
    def detect_robot_position(self):
        area,_,_,_ = self.image_processing.locate_opponent(self.camera.get_image())
        return area