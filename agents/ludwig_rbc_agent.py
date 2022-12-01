import numpy as np

def rbc_policy(observation, action_space):
    """
    complicated rule based policy based on day or night time
    """
    # all observations are from the last timestep. in particular, solar-load is the last timestep's electricity use!
    hour = observation[2] # Hour index is 2 for all observations
    solar = observation[21] # solar generation in kWh
    load = observation[20] # non-shiftable load
    #price = observation[24]
    #net_electricity = observation[23]


    action = 0.0

    if (16 <= hour <= 24) or (1 <= hour <= 10):
        # Afternoon (high prices 16-20): try to use up the stored energy
        action = min(1,
                     max((solar-load)/6.4,
                          -1
                        ))
    elif (11 <= hour <= 15):
        # Rest of Day: store enough solar power to get the battery near full
        action = min(1,
                     max((solar-load)/6.4,
                          0.24
                        ))

    action = np.array([action], dtype=action_space.dtype)

    #uncomment to discretize the action in 9 possible values:
    action = np.round(action * 4)/4

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