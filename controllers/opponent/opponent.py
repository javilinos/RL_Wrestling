# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimalist controller example for the Robot Wrestling Tournament.
   Demonstrates how to play a simple motion file."""

from controller import Robot, Motion, Supervisor
import sys
import os
sys.path.append('..') # adding the utils folder to get access to some custom helper functions, have a look at it
from utils.accelerometer import Accelerometer
from utils.finite_state_machine import FiniteStateMachine
from utils.motion_library import MotionLibrary
from utils.current_motion_manager import CurrentMotionManager
from utils.camera import Camera
from utils.fall_detection import FallDetection  # David's fall detection is implemented in this class
from utils.running_average import RunningAverage
from utils.image_processing import ImageProcessing as IP
from utils.gait_manager import GaitManager
from threading import Thread
from participant.shared import res, reset
import cv2
import random
from multiprocessing import shared_memory
import time

time.sleep(5)

shm_b = shared_memory.SharedMemory("env_reset")
shm_b.buf[0] = 0

class Bob ():
    def __init__(self, robot):
        # to load all the motions from the motion folder, we use the Motion_library class:
        self.robot = robot
        self.library = MotionLibrary()
        self.reset()

    def run(self):
        global shm_b
        # to control a motor, we use the setPosition() function:
        self.RShoulderPitch.setPosition(1.3)
        self.LShoulderPitch.setPosition(1.3)
        # for more motor control functions, see the documentation: https://cyberbotics.com/doc/reference/motor
        # to see the list of available devices, see the NAO documentation: https://cyberbotics.com/doc/guide/nao
        
        while shm_b.buf[0] == 0:
            self.robot.step(self.time_step)
            # if self.robot.getTime() == 1: # We wait a bit for the robot to stabilise
            #     # to play a motion from the library, we use the play() function as follows:
            #     self.library.play('Forwards50')

    def reset(self):
        
        self.time_step = int(self.robot.getBasicTimeStep())
        # we initialize the shoulder pitch motors using the Robot.getDevice() function:
        self.RShoulderPitch = self.robot.getDevice("RShoulderPitch")
        self.LShoulderPitch = self.robot.getDevice("LShoulderPitch")
        
        shm_b.buf[0] = 0

class Charlie ():
    def __init__(self, robot):
        self.robot = robot
        self.library = MotionLibrary()
        self.library.add('Shove', './Shove.motion', loop=True)
        self.reset()

    def run(self):
        global shm_b
        self.library.play('Stand')
        
        while shm_b.buf[0] == 0:
            self.robot.step(self.time_step)
            # When the robot is done standing for stabilization, it moves forwards
            if self.library.get('Stand').isOver():
                self.library.play('ForwardLoop')  # walk forward
                self.library.play('Shove')        # play the shove motion
        
    def reset(self):
        self.time_step = int(self.robot.getBasicTimeStep())

        # there are 7 controllable LEDs on the NAO robot, but we will use only the ones in the eyes

        # adding a custom motion to the library
        
        shm_b.buf[0] = 0

class David ():
    def __init__(self, robot):
        self.robot = robot
        # retrieves the WorldInfo.basicTimeTime (ms) from the world file
        self.reset()

    def run(self):
        global shm_b
        
        self.leds['right'].set(0x0000ff)
        self.leds['left'].set(0x0000ff)
        self.current_motion.set(self.library.get('Stand'))
        self.fsm.transition_to('BLOCKING_MOTION')
        
        while shm_b.buf[0] == 0:
            self.robot.step(self.time_step)
            self.detect_fall()
            self.fsm.execute_action()

    def detect_fall(self):
        '''Detect a fall and update the FSM state.'''
        [acc_x, acc_y, _] = self.accelerometer.get_new_average()
        if acc_x < -7:
            self.fsm.transition_to('FRONT_FALL')
        elif acc_x > 7:
            self.fsm.transition_to('BACK_FALL')
        if acc_y < -7:
            # Fell to its right, pushing itself on its back
            self.RShoulderRoll.setPosition(-1.2)
        elif acc_y > 7:
            # Fell to its left, pushing itself on its back
            self.LShoulderRoll.setPosition(1.2)

    def pending(self):
        # waits for the current motion to finish before doing anything else
        if self.current_motion.is_over():
            self.fsm.transition_to('DEFAULT')

    def walk(self):
        if self.current_motion.get() != self.library.get('ForwardLoop'):
            self.current_motion.set(self.library.get('ForwardLoop'))

    def front_fall(self):
        self.current_motion.set(self.library.get('GetUpFront'))
        self.fsm.transition_to('BLOCKING_MOTION')

    def back_fall(self):
        self.current_motion.set(self.library.get('GetUpBack'))
        self.fsm.transition_to('BLOCKING_MOTION')

    def reset(self):
        self.time_step = int(self.robot.getBasicTimeStep())
        # the Finite State Machine (FSM) is a way of representing a robot's behavior as a sequence of states
        self.fsm = FiniteStateMachine(
            states=['DEFAULT', 'BLOCKING_MOTION', 'FRONT_FALL', 'BACK_FALL'],
            initial_state='DEFAULT',
            actions={
                'BLOCKING_MOTION': self.pending,
                'DEFAULT': self.walk,
                'FRONT_FALL': self.front_fall,
                'BACK_FALL': self.back_fall
            }
        )
        self.accelerometer = Accelerometer(self.robot, self.time_step)
        self.leds = {
            'right': self.robot.getDevice('Face/Led/Right'),
            'left': self.robot.getDevice('Face/Led/Left')
        }

        # Shoulder roll motors for getting up from a side fall
        self.RShoulderRoll = self.robot.getDevice('RShoulderRoll')
        self.LShoulderRoll = self.robot.getDevice('LShoulderRoll')

        # load motion files
        self.current_motion = CurrentMotionManager()
        self.library = MotionLibrary()
        shm_b.buf[0] = 0

class Eve ():
    NUMBER_OF_DODGE_STEPS = 10

    def __init__(self, robot):

        # retrieves the WorldInfo.basicTimeTime (ms) from the world file
        self.robot = robot
        self.reset()

    def run(self):
        global shm_b
        while shm_b.buf[0] == 0:
            self.robot.step(self.time_step)
            self.opponent_position.update_average(
                self._get_normalized_opponent_horizontal_position())
            self.fall_detector.check(bool(shm_b.buf[0]))
            self.fsm.execute_action()

    def choose_action(self):
        if self.opponent_position.average < -0.4:
            self.current_motion.set(self.motions['TurnLeft'])
        elif self.opponent_position.average > 0.4:
            self.current_motion.set(self.motions['TurnRight'])
        else:
            # dodging by alternating between left and right side steps to avoid easily falling off the ring
            if self.dodging_direction == 'left':
                if self.counter < self.NUMBER_OF_DODGE_STEPS:
                    self.current_motion.set(self.motions['SideStepLeft'])
                    self.counter += 1
                else:
                    self.dodging_direction = 'right'
            elif self.dodging_direction == 'right':
                if self.counter > 0:
                    self.current_motion.set(self.motions['SideStepRight'])
                    self.counter -= 1
                else:
                    self.dodging_direction = 'left'
            else:
                return
        self.fsm.transition_to('BLOCKING_MOTION')

    def pending(self):
        # waits for the current motion to finish before doing anything else
        if self.current_motion.is_over():
            self.fsm.transition_to('CHOOSE_ACTION')

    def _get_normalized_opponent_horizontal_position(self):
        """Returns the horizontal position of the opponent in the image, normalized to [-1, 1]
            and sends an annotated image to the robot window."""
        img = self.camera.get_image()
        largest_contour, vertical, horizontal = self.locate_opponent(img)
        output = img.copy()
        if largest_contour is not None:
            cv2.drawContours(output, [largest_contour], 0, (255, 255, 0), 1)
            output = cv2.circle(output, (horizontal, vertical), radius=2,
                                color=(0, 0, 255), thickness=-1)
        self.camera.send_to_robot_window(output)
        if horizontal is None:
            return 0
        return horizontal * 2 / img.shape[1] - 1

    def locate_opponent(self, img):
        """Image processing demonstration to locate the opponent robot in an image."""
        # we suppose the robot to be located at a concentration of multiple color changes (big Laplacian values)
        laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=3)
        # those spikes are then smoothed out using a Gaussian blur to get blurry blobs
        blur = cv2.GaussianBlur(laplacian, (0, 0), 2)
        # we apply a threshold to get a binary image of potential robot locations
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        # the binary image is then dilated to merge small groups of blobs together
        closing = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        # the robot is assumed to be the largest contour
        largest_contour = IP.get_largest_contour(closing)
        if largest_contour is not None:
            # we get its centroid for an approximate opponent location
            vertical_coordinate, horizontal_coordinate = IP.get_contour_centroid(
                largest_contour)
            return largest_contour, vertical_coordinate, horizontal_coordinate
        else:
            # if no contour is found, we return None
            return None, None, None
    
    def reset(self):
        global shm_b
        self.time_step = int(self.robot.getBasicTimeStep())

        self.fsm = FiniteStateMachine(
            states=['CHOOSE_ACTION', 'BLOCKING_MOTION'],
            initial_state='CHOOSE_ACTION',
            actions={
                'CHOOSE_ACTION': self.choose_action,
                'BLOCKING_MOTION': self.pending
            }
        )

        self.camera = Camera(self.robot)

        # arm motors for getting up from a side fall
        self.RShoulderRoll = self.robot.getDevice("RShoulderRoll")
        self.LShoulderRoll = self.robot.getDevice("LShoulderRoll")

        self.fall_detector = FallDetection(self.time_step, self.robot)
        self.current_motion = CurrentMotionManager()
        # load motion files
        self.motions = {
            'SideStepLeft': Motion('../motions/SideStepLeftLoop.motion'),
            'SideStepRight': Motion('../motions/SideStepRightLoop.motion'),
            'TurnRight': Motion('../motions/TurnRight20.motion'),
            'TurnLeft': Motion('../motions/TurnLeft20.motion'),
        }
        self.opponent_position = RunningAverage(dimensions=1)
        self.dodging_direction = 'left'
        self.counter = 0
        
        shm_b.buf[0] = 0

class Fatima ():
    global res
    SMALLEST_TURNING_RADIUS = 0.1
    SAFE_ZONE = 0.75
    TIME_BEFORE_DIRECTION_CHANGE = 200  # 8000 ms / 40 ms

    def __init__(self, robot: Supervisor, *args, **kwargs):
        self.robot = robot
        self.reset()

    def run(self):
        global shm_b
        shm_b.buf[0] = 0
        while shm_b.buf[0] == 0:
            self.robot.step(self.time_step)
            # We need to update the internal theta value of the gait manager at every step:
            t = self.robot.getTime()
            self.gait_manager.update_theta()
            if 0.3 < t < 2:
                self.start_sequence()
                
            elif t > 2:
                self.fall_detector.check(bool(shm_b.buf[0]))
                self.walk()
            

    def start_sequence(self):
        """At the beginning of the match, the robot walks forwards to move away from the edges."""
        self.gait_manager.command_to_motors(heading_angle=0)

    def walk(self):
        """Dodge the opponent robot by taking side steps."""
        normalized_x = self._get_normalized_opponent_x()
        # We set the desired radius such that the robot walks towards the opponent.
        # If the opponent is close to the middle, the robot walks straight.
        desired_radius = (self.SMALLEST_TURNING_RADIUS / normalized_x) if abs(normalized_x) > 1e-3 else None
        # TODO: position estimation so that if the robot is close to the edge, it switches dodging direction
        if self.counter > self.TIME_BEFORE_DIRECTION_CHANGE:
            self.heading_angle = - self.heading_angle
            self.counter = 0
        self.counter += 1
        self.gait_manager.command_to_motors(desired_radius=desired_radius, heading_angle=self.heading_angle)

    def _get_normalized_opponent_x(self):
        """Locate the opponent in the image and return its horizontal position in the range [-1, 1]."""
        img = self.camera.get_image()
        _, _, horizontal_coordinate = IP.locate_opponent(img)
        if horizontal_coordinate is None:
            return 0
        return horizontal_coordinate * 2 / img.shape[1] - 1
    
    def reset(self):
        global shm_b
        self.time_step = int(self.robot.getBasicTimeStep())

        self.camera = Camera(self.robot)
        self.fall_detector = FallDetection(self.time_step, self.robot)
        self.gait_manager = GaitManager(self.robot, self.time_step)
        self.heading_angle = 3.14 / 2
        # Time before changing direction to stop the robot from falling off the ring
        self.counter = 0
        shm_b.buf[0] = 0
        
    
class AllInOneRobot(Supervisor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.time_step = int(self.getBasicTimeStep())
        self.opponents = {1: Bob, 2:Charlie, 3:David, 4: Eve, 5:Fatima}
        
    def run(self):
        op_id = random.randint(1, 1)
        print (f"oponent {op_id} loaded")
        oponnent = self.opponents[op_id](self)     
        oponnent.run()
        oponnent.reset()
        time.sleep(0.5)
        del oponnent

# create the Robot instance and run main loop
#oponents = {1: Bob, 2: Charlie, 3: David, 4: Fatima}
# while shm_b != 1:
#     pass

opponent = AllInOneRobot()

while(True):
    opponent.run()

shm_b.close()
print("oponent should have ended")
