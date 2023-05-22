"""Reward class"""

import math
import numpy as np
from controller import Supervisor

class Reward:
    def __init__(self, wrestler, initial_x, initial_y):
        self.min = [initial_x, initial_y]
        self.max = [initial_x, initial_y]
        self.ko_count = 0
        self.op_ko_count = 0
        self.time_step = int(wrestler.getBasicTimeStep())
        self.coverage_gained = 0.0
        self.n_steps = 0.0

    def calculate_reward(self, pos_x, pos_y, pos_z, op_pos_x, op_pos_y, op_pos_z, rel_x, rel_y, rel_yaw_2_oponent):

        done = False
        reward = 0.0
        self.n_steps += 1

################################# Continuous reward #####################################

        position = [pos_x, pos_y]
        box = [0] * 3
        if abs(position[0]) < 1 and abs(position[1]) < 1:  # inside the ring
            coverage = 0
            for j in range(2):
                if position[j] < self.min[j]:
                    self.min[j] = position[j]
                elif position[j] > self.max[j]:
                    self.max[j] = position[j]
                box[j] = self.max[j] - self.min[j]
                coverage += box[j] * box[j]
            coverage = math.sqrt(coverage)
            
            if coverage > self.coverage_gained:
                # reward = coverage - self.coverage_gained
                # reward = reward * 10
                if (coverage - self.coverage_gained) > 0.0:
                    reward = 0.05
                else:
                    reward = -0.01

                self.coverage_gained = coverage      

        distance = math.sqrt(rel_x**2 + rel_y**2)
        distance = distance / 2.45
        distance = -(1 - distance)
        distance = np.clip(distance, -1.0, 0.0)
        reward += distance * 0.01

        rel_yaw = 1 - abs(rel_yaw_2_oponent)
        rel_yaw = rel_yaw*2 - 1
        reward -= rel_yaw * 0.01

        # distance_to_center = math.sqrt(position[0]**2 + position[1]**2)
        # distance_to_center = distance_to_center/math.sqrt(2)
        # distance_to_center = -np.clip(distance_to_center, 0.0, 1.0)
        # reward += distance_to_center * 0.01

        #print(reward)
        # position below threshold (0.9) or robot exploded (any coordinate above 1.5)
        # if pos_z < 0.9:
        #     self.ko_count = self.ko_count + self.time_step
        #     reward -= (self.ko_count*0.00001)
        # else:
        #     self.ko_count = 0

#################################### Terminal reward ################################

        if abs(position[0]) > 1.5 or abs(position[1]) > 1.5 or pos_z < 0.9:
            reward = -20
            self.ko_count = 0
            self.coverage_gained = 0.0
            done = True

        if abs(op_pos_x) > 1.5 or abs(op_pos_y) > 1.5 or op_pos_z < 0.9:
            #reward = 10
            self.ko_count = 0
            self.coverage_gained = 0.0
            done = True

        #if distance > -0.2 and rel_yaw > 0.9:
        if distance < -0.87:
            reward = -20
            self.op_ko_count = 0
            self.coverage_gained = 0.0
            done = True

        if self.n_steps > 2048:
            reward = self.coverage_gained*2
            print(reward)
            done = True
        #reward += 0.1

        return reward, done
    
    def calculate_action_reward(self, state: list, action):
        action_reward = 0.0
        if state[2]>=0.9:
            if action == 0:
                action_reward += 0.01
            # elif action == 4 or action == 5:
            #     action_reward += -0.3
        # elif state[2] < 0.9:
        #     action_reward = -0.1
        #     if action == 4 or action == 5:
        #         action_reward += 0.1
        #     elif action == 0:
        #         action_reward += -0.3

        return action_reward            

    
    def clean_reward(self):
        reward = 0.0
        self.n_steps = 0.0
        self.coverage_gained = 0.0

    def clean_min_max(self, initial_x, initial_y):
        self.min = [initial_x, initial_y]
        self.max = [initial_x, initial_y]
