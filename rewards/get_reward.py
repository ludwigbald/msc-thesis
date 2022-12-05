from typing import List
import numpy as np

###########################################################################
#####                Specify your reward function here                #####
###########################################################################

def get_reward(electricity_consumption: List[float], carbon_emission: List[float], electricity_price: List[float], agent_ids: List[int]) -> List[float]:
        """CityLearn Challenge user reward calculation.

        Parameters
        ----------
        electricity_consumption: List[float]
            List of each building's/total district electricity consumption in [kWh].
        carbon_emission: List[float]
            List of each building's/total district carbon emissions in [kg_co2].
        electricity_price: List[float]
            List of each building's/total district electricity price in [$].
        agent_ids: List[int]
            List of agent IDs matching the ordering in `electricity_consumption`, `carbon_emission` and `electricity_price`.

        Returns
        -------
        rewards: List[float]
            Agent(s) reward(s) where the length of returned list is either = 1 (central agent controlling all buildings) 
            or = number of buildings (independent agent for each building).
        """

        # *********** BEGIN EDIT ***********
        # Replace with custom reward calculation

        # print(electricity_consumption)
        # print(electricity_price)
        # print(carbon_emission)
        # breakpoint()
        # we need to normalize by the total values, if we would not make use of the battery.

        total_co2_no_battery = np.array([1117.6211690517193, 1043.3168421031762, 707.0223402007878, 1080.9376721533065, 791.53366612883]).mean() /8760
        total_cost_no_battery = np.array([2250.8700563860034, 1966.1250130131689, 1315.5171066205235, 1707.2781431365536, 1540.8814011827963]).mean() /8760

        carbon_emission   = np.array(carbon_emission  ).clip(min=0) / total_co2_no_battery
        electricity_price = np.array(electricity_price).clip(min=0) / total_cost_no_battery

        reward = 2+ (carbon_emission + electricity_price)*-1
        # ************** END ***************
        return reward