# This python script loads the weights for all tuning runs and re-evaluates them on building 1.



# read command line arguments
# - seed
# - agent
# - path to data
# - number of episodes

import argparse
import json
import os
import time
import re

import pandas as pd
import numpy as np
import gym
import torch

import ast
import pickle
import matplotlib.pyplot as plt


from agents.dqn.dqn import DQN
from agents.uadqn.uadqn import UADQN
from agents.common.networks.mlp import MLP

from citylearn.citylearn import CityLearnEnv
from discrete_citylearn import DiscreteSingle
from discrete_citylearn import OneHotHour

row_dict = {}
rows=[]

# set up the environment:
class Constants:
    episodes = 1 # for evaluation only
    test_schema_path = os.path.join("data/citylearn_challenge_2022_phase_1", "test_schema.json")
    n_discrete = 9
    hour_index=0


def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }

def create_env(schema_path=Constants.test_schema_path, n_discrete=Constants.n_discrete, hour_index=Constants.hour_index):
    base_env = CityLearnEnv(schema_path)
    return gym.wrappers.NormalizeObservation(OneHotHour(DiscreteSingle(base_env, n_discrete), hour_index))

def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations }
    return obs_dict

# evaluation function: runs the trained agent for one episode without training
def evaluate():
    print("Starting local evaluation")

    env = create_env(schema_path=Constants.test_schema_path)
    observations = env.reset()

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = [agent.predict(torch.Tensor(obs)) for obs in observations]
    agent_time_elapsed += time.perf_counter()- step_start

    episodes_completed = 0
    num_steps = 0
    episode_metrics = []
    rewards = []
    actionss = []
    obss = []
    while True:

        observations, reward, done, _ = env.step(actions)
        rewards.append(reward)
        actionss.append(actions)
        obss.append(observations[0])

        if done:
            episodes_completed += 1
            metrics_t = env.evaluate()
            metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1]}
            if np.any(np.isnan(metrics_t)):
                raise ValueError("Episode metrics are nan, please contant organizers")
            episode_metrics.append(metrics)
            print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}" )
            print(np.mean(rewards, axis=0)/2, np.mean(rewards)/2)

            row_dict["carbon_score"]=np.sum(np.clip(env.buildings[0].net_electricity_consumption_emission, a_min=0, a_max=None))/ env.buildings[0].net_electricity_consumption_without_storage_emission.clip(min=0).sum()
            row_dict["cost_score"]=np.sum(np.clip(env.buildings[0].net_electricity_consumption_price, a_min=0, a_max=None))/ env.buildings[0].net_electricity_consumption_without_storage_price.clip(min=0).sum()
            print(row_dict)
            observations=env.reset()

            step_start = time.perf_counter()
            agent_time_elapsed += time.perf_counter()- step_start
        else:
            step_start = time.perf_counter()
            actions = [agent.act(torch.Tensor(obs)) for obs in observations]
            agent_time_elapsed += time.perf_counter()- step_start
        
        num_steps += 1
        if num_steps % 1000 == 0:
            print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

        if episodes_completed >= Constants.episodes:
            break
    
    

    if len(episode_metrics) > 0:
        print("Average Price Cost:", np.mean([e['price_cost'] for e in episode_metrics]))
        print("Average Emmision Cost:", np.mean([e['emmision_cost'] for e in episode_metrics]))
        print("Average of both costs:", (np.mean([e['price_cost'] for e in episode_metrics]) + np.mean([e['emmision_cost'] for e in episode_metrics]))/2)
    print(f"Total time taken by agent: {agent_time_elapsed}s")
    

# run experiment (save intermediary checkpoints)
if __name__ == '__main__':
    # set up environment:
    notes = "This is a re-evaluation run, executed on a single building."
    env = create_env(schema_path=Constants.test_schema_path)
    #nb_steps = 8760*args["episodes"]

    

    # read results of validation runs:
    validation_results="experiments/tuning/validation_results/"
    # read results
    run_folders = [validation_results+s for s in os.listdir(validation_results) if s.startswith("2022")]
    run_folders.sort(key=lambda f: int(re.findall(r"[\d]+\Z", f)[0]))

   
    for run_folder in run_folders:
        with open(run_folder+"/experimental-setup", "r") as setup:
            setup_dict = ast.literal_eval(setup.read())
            row_dict["agent"] = "UADQN" if re.search(r"UADQN", run_folder) else "DQN"
            row_dict["seed"] = setup_dict["seed"]

            if (row_dict["agent"] == "DQN"):
                row_dict["action_selection"] = setup_dict["action_selection"]
            else:
                row_dict["action_selection"] = None



        if (row_dict["agent"] == "DQN"):
            agent = DQN( env,
                        MLP,
                        replay_start_size=setup_dict["minibatch_size"],
                        replay_buffer_size=10000,
                        gamma=0.99,
                        update_target_frequency=setup_dict["update_target_frequency"],
                        minibatch_size=setup_dict["minibatch_size"],
                        learning_rate=setup_dict["learning_rate"],
                        action_selection = setup_dict["action_selection"], #"softmax"|"egreedy"
                        initial_exploration_rate=0.02,
                        final_exploration_rate=0.02,
                        final_exploration_step=1000,#8760,#1000
                        adam_epsilon=setup_dict["adam_epsilon"],
                        update_frequency=1,
                        logging=False,
                        log_folder_details=None,
                        render = False,
                        loss='mse',
                        seed=row_dict["seed"],
                        notes=notes)
            try:
                agent.load(os.path.join(run_folder, "network.pth"))
            except (FileNotFoundError):
                continue

        elif (row_dict["agent"] == "UADQN"):
            agent = UADQN( env,
                            MLP,
                            n_quantiles=20,
                            weight_scale=np.sqrt(2),
                            noise_scale=1,
                            epistemic_factor=1,
                            aleatoric_factor=0, #risk aversion, 0 means risk-neutral
                            kappa=10,
                            replay_start_size=setup_dict["minibatch_size"],
                            replay_buffer_size=10000,
                            gamma=0.99,
                            update_target_frequency=setup_dict["update_target_frequency"],
                            minibatch_size=setup_dict["minibatch_size"],
                            learning_rate=setup_dict["learning_rate"],
                            adam_epsilon=setup_dict["adam_epsilon"],
                            update_frequency=1,
                            logging=False,
                            log_folder_details=None,
                            seed = row_dict["seed"],
                            notes=notes)
            agent.load(run_folder+"/")

        
        evaluate()
        rows.append(row_dict)
        row_dict = {}
    dataframe=pd.DataFrame(rows)
    dataframe.to_csv("result.csv")
        

