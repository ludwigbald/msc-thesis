import numpy as np
import gym
from citylearn.citylearn import CityLearnEnv

class OneHotHour(gym.ObservationWrapper):
    def __init__(self, env, hour_index=3):
        super().__init__(env)
        self.hour_index = hour_index
        
        single_observation_space = env.buildings[0].observation_space

        lows = np.concatenate((np.zeros((24,)), np.delete(single_observation_space.low, hour_index)))
        highs = np.concatenate((np.ones((24,)), np.delete(single_observation_space.high, hour_index)))
        
        self._observation_space = gym.spaces.Box(lows, highs)

    def observation(self, obss):
        obs_out=[]
        for obs in obss:

            hours = np.zeros((24,))
            hours[obs[self.hour_index] - 1]=1
            obs_out.append(np.concatenate((hours, np.delete(obs, self.hour_index))))
        return obs_out


class DiscreteSingle(gym.ActionWrapper):
    def __init__(self, env, n_discrete):
       # assert n_discrete % 2 == 1 , "Please use an odd n_discrete to preserve the middle 0 action"
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