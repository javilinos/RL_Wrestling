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
import sys
from time import time, sleep
sys.path.append('..')
import numpy as np
import cloudpickle
from time import time
import threading

from utils.fall_detection import FallDetection

from controller import Robot, Supervisor
from observation import Observation
from reward import Reward
from action import Action

from webots_gym import Environment
from sb3_contrib import RecurrentPPO
import torch as th

import gymnasium as gym
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space

# We provide a set of utilities to help you with the development of your controller. You can find them in the utils folder.
# If you want to see a list of examples that use them, you can go to https://github.com/cyberbotics/wrestling#demo-robot-controllers

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, wrestler, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.wrestler = wrestler
        #self.update_pub = rospy.Publisher('/environment/flightmare/net_update', EmptyMsg, queue_size=1)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.wrestler.simulationReset()
        

    def _on_rollout_start(self) -> None:

        self.wrestler.simulationSetMode(2)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        
        return True

    def _on_rollout_end(self) -> None:

        self.wrestler.simulationSetMode(0)
        #self.update_pub.publish()

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.training_env.end()
        self.wrestler.simulationSetMode(0)


class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space, features_dim: int = 1024, normalized_image: bool = False) -> None:
        print("using custom cnn")
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        # self.cnn = nn.Sequential(
        #     nn.Conv3d(1, 128, kernel_size=(1, 4, 4), stride=(1, 4, 4)),
        #     nn.ReLU(),
        #     nn.Conv3d(128, 164, kernel_size=(4, 4, 4), stride=(2, 2, 2)),
        #     nn.ReLU(),
        #     nn.Conv3d(164, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
        #     nn.ReLU(),
        #     #nn.AdaptiveAvgPool3d((1, 1, 1)), # added this layer to reduce output size to [256, 256, 1, 1, 1]
        #     nn.Flatten()
        # )

        # # Compute shape by doing one forward pass
        # with th.no_grad():
        #     n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        # self.linear = nn.Sequential(
        #     nn.Linear(n_flatten, features_dim),
        #     nn.ReLU()
        # )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.linear(self.cnn(observations))
        return x

class Wrestler(Robot):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time_step = int(self.getBasicTimeStep())
        self.action_node = Action(self, self.time_step)

    def run(self):
        #################################################################################
        ############################# TRAINING ##########################################
        #################################################################################
        # observation = Observation(self)
        # obs = observation.get_observation_state()
        # reward = Reward(self, obs[0], obs[1])
        # cb = CustomCallback(self)
        
        # env = Environment(observation, self.action_node, reward, self)
        # env = Monitor(env)
        # checkpoint_callback = CheckpointCallback(save_freq=20000, save_path="/home/javilinos/checkpoints/PPO_10", name_prefix="runner_model")
        # #checkpoint_callback = CheckpointCallback(save_freq=20000, save_path="/home/javilinos/checkpoints/PPO_1", name_prefix="rl_model")
        # model = RecurrentPPO("CnnLstmPolicy", env, tensorboard_log="/home/javilinos/PPO", verbose=1, ent_coef=0.01, learning_rate=3e-05, gae_lambda=0.95, clip_range=0.2, n_steps=1024, batch_size=256, n_epochs=10, normalize_advantage=True, use_sde=True, sde_sample_freq=8, policy_kwargs=dict(
        #     ortho_init = True,
        #     share_features_extractor = True,
        #     normalize_images = True,
        #     activation_fn=th.nn.ReLU,
        #     features_extractor_class=CustomCNN
        #     )
        # )
        # print("Starting mission...")
        # model.learn(total_timesteps=4000000, callback=[cb, checkpoint_callback])


        #################################################################################
        ############################# TESTING ###########################################
        #################################################################################
        
        print ("Initializing Fall detector")
        fall_detector = FallDetection(self.time_step, self)
        print ("Initializing Observation")
        observation = Observation(self)
        print ("Initializing Action")
        lstm_states = None
        num_envs = 1
        # Episode start signals are used to reset the lstm states
        episode_starts = np.ones((num_envs,), dtype=bool)
        print ("Initializing RL model")
        with open("model.pkl", "rb") as f:
            rl_model = cloudpickle.load(f)

        t1 = self.getTime()
        runned = False
        while self.step(self.time_step) != -1 :  # mandatory function to make the simulation run
            t2 = self.getTime()
            if (t2-t1) < 4:
                self.action_node.execute_action([0.0])
                self.action_node.arms_to_running_position()
                runned = True
            else:
                if runned:
                    self.action_node.stand()
                    self.action_node.reset_gait_manager()
                    runned = False
                else:
                    if (fall_detector.check()):
                        self.action_node.reset_gait_manager()
                    # area = observation.detect_robot_position()
                    # if area is not None and area < 200 and area > 150:
                    #     self.action_node.hit_front_robot()
                    # else:
                    #     self.action_node.arms_to_normal_position()
                    obs = observation.image_to_predict()
                    # if (observation.get_l_sensor() < 0.3 or observation.get_r_sensor() < 0.3):
                    #     self.action_node.hit_front_robot()

                    self.action_node.arms_to_running_position()
                    action, lstm_states = rl_model.predict(obs, state=lstm_states, episode_start=episode_starts)
                    episode_starts = np.zeros((num_envs,), dtype=bool)
                    self.action_node.execute_action(action)

        #################################################################################
        ############################# EVALUATING ########################################
        #################################################################################

        # n_step = 1000000
        # observation = Observation(self)
        # obs = observation.get_observation_state()
        # reward = Reward(self, obs[0], obs[1])
        # action = Action(self, self.time_step)
        # cb = CustomCallback(self)
        # results = []
        # env = Environment(observation, action, reward, self)
        # env = Monitor(env)
        # done = True
        # while (n_step < 3200000):
        #     if (done):
        #         model = RecurrentPPO.load(f"/home/javilinos/checkpoints/PPO_7/attacker_model_2180000_steps.zip")
        #         done = False
        #         mean_reward, std_reward = evaluate_policy(model, env, 25)
        #         done = True
        #         result = f"model: {2200000} -> mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}"
        #         print (result)
        #         results.append(result)
        #         n_step += 20000
        # print(results)
# create the Robot instance and run main loop
wrestler = Wrestler()
wrestler.run()
