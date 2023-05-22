from sb3_contrib import RecurrentPPO
import torch as th
import cloudpickle
import pickle
import time

rl_model = RecurrentPPO.load("runner_model")

with open("model.pkl", "wb") as f:
    cloudpickle.dump(rl_model, f)

print("cloudpickle is life")