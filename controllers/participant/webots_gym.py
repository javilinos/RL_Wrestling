from gymnasium import Env
from gymnasium import spaces
import numpy as np
from controller import Supervisor
from action import Action
from observation import Observation
from reward import Reward
import random
from multiprocessing import shared_memory
import time

class Environment(Env):
    def __init__(self, observation: Observation, action: Action, reward: Reward, robot):
        super().__init__()
        # self.action_space = spaces.Box(np.array((0, 0, 0, 0, 0, 0, 0, 0, 0)),
        #  np.array((1, 1, 1, 1, 1, 1, 1, 1, 1)), dtype=np.int8)
        self.num_frames = 1
        self.frames = []
        self.action_space = spaces.Box(np.array((-1.0)), np.array((1.0)), dtype=np.float32)
        # self.observation_space = spaces.Box(np.array((-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)), np.array((1.0, 1.0, 1.0, 1.0, 1.0, 1.0)), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.num_frames, 84, 84), dtype=np.uint8)
        self.render_mode = False
        self.robot = robot 
        self.observation = observation
        self.action = action
        self.reward = reward
        self.time_step = int(self.robot.getBasicTimeStep())
        self.n_steps = 0
        self.n_episodes = 0
        self.last_action = -1
        self.shm_a = shared_memory.SharedMemory(create=True, size=1, name="env_reset")
        time.sleep(1)
        self.shm_a.buf[0] = 0
        self.node_red = Supervisor.getFromDef(Supervisor, 'WRESTLER_RED')
        self.trans_field_red = self.node_red.getField("translation")
        self.rot_field_red = self.node_red.getField("rotation")
        self.node_blue = Supervisor.getFromDef(Supervisor, 'WRESTLER_BLUE')
        self.trans_field_blue = self.node_blue.getField("translation")
        self.rot_field_blue = self.node_blue.getField("rotation")
        
    def step(self, action):
        
        # if self.last_action != action:
        #     self.action.stop_last_action()

        self.robot.step(self.time_step)
        self.action.execute_action(action)
        obs_state = self.observation.get_observation_state()
        self.frames.append(self.observation.get_observation_image())
        #self.observation.get_joint_states()
        self.frames.pop(0)
        rew, done = self.reward.calculate_reward(
            obs_state[0], obs_state[1], obs_state[2], obs_state[3], obs_state[4], obs_state[5], obs_state[6], obs_state[7], obs_state[8])
        self.n_steps += 1
        # rew+=self.reward.calculate_action_reward(obs_state, action)
        if self.n_steps > 2048:
            done = True
            rew = 10
            self.n_steps = 0

        obs = np.stack(self.frames, axis=0)
        
        return obs, rew, done, False, {}

    def reset(self):
        self.n_episodes+=1
        self.n_steps = 0
        self.shm_a.buf[0] = 1
        Supervisor.simulationReset(self.robot)
        self.robot.step(self.time_step)
        
        x_red = random.uniform(-0.85, -0.2)
        y_red = random.uniform(-0.85, 0.85)
        yaw_red = random.uniform(0.0, 3.13)
        x_blue = random.uniform(0.85, 0.2)
        y_blue = random.uniform(-0.85, 0.85)
        yaw_blue = random.uniform(0.0, 3.13)
        initial_red_trans = [x_red, y_red, 0.834]
        initial_red_rot = [0.0, 0.0, 1.0, yaw_red]
        initial_blue_trans = [x_blue, y_blue, 0.834]
        initial_blue_rot = [0.0, 0.0, 1.0, yaw_blue]
        if self.n_episodes % 2 == 0:
            self.trans_field_red.setSFVec3f(initial_red_trans)
            self.rot_field_red.setSFRotation(initial_red_rot)
            self.trans_field_blue.setSFVec3f(initial_blue_trans)
            self.rot_field_blue.setSFRotation(initial_blue_rot)
        else:
            self.trans_field_red.setSFVec3f(initial_blue_trans)
            self.rot_field_red.setSFRotation(initial_blue_rot)
            self.trans_field_blue.setSFVec3f(initial_red_trans)
            self.rot_field_blue.setSFRotation(initial_red_rot)

        self.node_red.resetPhysics()
        self.node_blue.resetPhysics()
        self.action.reset_gait_manager()
        obs_state = self.observation.get_observation_state()
        self.frames = [self.observation.get_observation_image()] * self.num_frames
        
        self.reward.clean_reward()
        self.reward.clean_min_max(obs_state[0], obs_state[1])
        obs = np.stack(self.frames, axis=0)
        
        return obs, {}
    
    def end(self):
        self.shm_a.close()
        self.shm_a.unlink()

