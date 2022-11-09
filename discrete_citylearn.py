import numpy as np
import gym
from citylearn.citylearn import CityLearnEnv


class DiscreteSingle(gym.ActionWrapper):
    def __init__(self, env, n_discrete):
        assert n_discrete % 2 == 1 , "Please use an odd n_discrete to preserve the middle 0 action"
        super().__init__(env)
        self.n_discrete = n_discrete
        self.action_space = gym.spaces.Discrete(n_discrete)
        self._observation_space = env.buildings[0].observation_space

    # expects a list of discrete actions and returns a list of singleton lists of continuous actions
    def action(self, actions):
        return [ [ (a-((self.n_discrete-1)/2))
                    /
                    ((self.n_discrete-1)/2)
                 ] for a in actions]