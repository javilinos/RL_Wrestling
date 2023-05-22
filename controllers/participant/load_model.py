from sb3_contrib import RecurrentPPO
import torch as th
import cloudpickle
import pickle

with open("model.pkl", "rb") as f:
    loaded_model = cloudpickle.load(f)

print(loaded_model.action_space)