# This python script takes command line parameters and runs an experiment on CityLearn
# It saves the results in the form ./results/<agent>/<seed>/<timestamp.filetype>


# read command line arguments
# - seed
# - agent
# - path to data
# - number of episodes

import argparse
import json
import os
import time

import numpy as np
import gym
import torch

import pickle
import matplotlib.pyplot as plt


from agents.dqn.dqn import DQN
from agents.uadqn.uadqn import UADQN
from agents.common.networks.mlp import MLP

from citylearn.citylearn import CityLearnEnv
from discrete_citylearn import DiscreteSingle



parser = argparse.ArgumentParser(description='Run Experiment on CityLearn')
parser.add_argument('--agent', dest="agent", type=str, required=False, default="DQN",
                    help='the agent.') #TODO: how to specify agent? runscript?
parser.add_argument('--seed', dest="seed", type=int, required=False, default=42,
                    help='random seed for the training process')
parser.add_argument('--episodes', dest="episodes", type=int, required=False, default=1,
                    help='number of episodes to train for')                    
parser.add_argument('--data-path', dest="data_path", type=str, required=False, default="data/citylearn_challenge_2022_phase_1",
                    help='path to dataset')
parser.add_argument("--dry-run", dest="dry_run", action="store_true", required=False,
                    help="if set, don't save anything")
parser.add_argument("--n_discrete", dest="n_discrete", type=int, default=7, required=False,
                    help="the number of discrete actions possible")

args = vars(parser.parse_args())

# set up the environment:
class Constants:
    episodes = 1 # for evaluation only
    schema_path = os.path.join(args["data_path"], "schema.json")

    train_schema_path = os.path.join(args["data_path"], "test_schema.json")

    test_schema_path = os.path.join(args["data_path"], "test_schema.json")
    n_discrete = args["n_discrete"]

# save experiment parameters
if args["dry_run"]:
    print("Proceeding experiment without keeping logs. Variables:")
    print(args)
else:
    timestamp=time.strftime("%Y%m%d-%H%M%S")
    results_path=os.path.join("results", str(args["agent"]), str(args["seed"]), timestamp)
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path,"args.json"), mode="w") as f:
        json.dump(args, f, indent="")
    print("Saved experimental conditions")



def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }

def create_env(schema_path=Constants.schema_path, n_discrete=Constants.n_discrete):
    base_env = CityLearnEnv(schema_path)
    return DiscreteSingle(base_env, n_discrete)

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
    while True:

        observations, reward, done, _ = env.step(actions)
        rewards.append(reward)
        actionss.append(actions)

        if done:
            episodes_completed += 1
            metrics_t = env.evaluate()
            metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1]}
            if np.any(np.isnan(metrics_t)):
                raise ValueError("Episode metrics are nan, please contant organizers")
            episode_metrics.append(metrics)
            print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}" )

            np.savetxt("soc.csv", env.buildings[0].electrical_storage.soc, delimiter=",")
            np.savetxt("actions.csv", actionss, delimiter=",")
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
    notes = "This is a test run"
    env = create_env(Constants.train_schema_path)

    nb_steps = 8760*args["episodes"]


    if (args["agent"] == "DQN"):
        agent = DQN( env,
                    MLP,
                    replay_start_size=256,
                    replay_buffer_size=10000,
                    gamma=0.99,
                    update_target_frequency=50,
                    minibatch_size=256,
                    learning_rate=3e-4,
                    initial_exploration_rate=1,
                    final_exploration_rate=0.02,
                    final_exploration_step=8760,#1000
                    adam_epsilon=1e-8,
                    update_frequency=1,
                    logging=True,
                    log_folder_details="CityLearn-DQN",
                    render = False,
                    loss='mse',
                    seed=args["seed"],
                    notes=notes)
    elif (args["agent"] == "UADQN"):
        agent = UADQN( env,
                MLP,
                n_quantiles=20,
                weight_scale=3,
                noise_scale=1,
                epistemic_factor=1,
                aleatoric_factor=0,
                kappa=10,
                replay_start_size=256,
                replay_buffer_size=50000,
                gamma=0.99,
                update_target_frequency=50,
                minibatch_size=256,
                learning_rate=1e-3,
                adam_epsilon=1e-8,
                update_frequency=1,
                logging=True,
                log_folder_details="CityLearn-UADQN",
                seed = args["seed"],
                notes=notes)
    else:
        print("This agent does not exist: ", args["agent"])


    # train for an episode
    agent.learn(timesteps=nb_steps, verbose=True)
    
    scores = np.array(agent.logger.memory['Episode_score'])

    evaluate()

# save raw results for later analysis.
# - training loss
# - uncertainties?