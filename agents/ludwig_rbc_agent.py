import numpy as np

def rbc_policy(observation, action_space):
    """
    Simple rule based policy based on day or night time
    """
    hour = observation[2] # Hour index is 2 for all observations
    solar = observation[21] # solar generation in kWh
    load = observation[20] # non-shiftable load

    action = 0.0
    if 16 <= hour <= 20:
        # Afternoon (high prices): release/sell all stored energy
        action = -0.2
    elif (1 <= hour <= 15) or (21 <= hour <= 24):
        # Rest of Day: store enough solar power to get the battery near full
        action = 0.6*solar/6.4 # where 6.4 kWh is battery capacity
        # action should be the share of battery size (which is given in kWh)
    



    action = np.array([action], dtype=action_space.dtype)
    assert action_space.contains(action)
    return action

class LudwigRBCAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """
    def __init__(self):
        self.action_space = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return rbc_policy(observation, self.action_space[agent_id])