import numpy as np
import torch
from torch import nn, optim
import pickle

from typing import List
import numpy as np
from citylearn.agents.rbc import RBC, BasicRBC, OptimizedRBC
from citylearn.agents.rlc import RLC
from citylearn.rl import PolicyNetwork, ReplayBuffer, SoftQNetwork

# This class loads a pretrained citylearn.agents.sac.SAC agent for evaluation.

class LudwigSACAgent:

    def __init__(self):
        with open("agent.pickle", "rb") as f:
            self.agent = pickle.load(f)
        self.action_space = {}
    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        actions = self.agent.select_actions(observation[agent_id])
        return actions