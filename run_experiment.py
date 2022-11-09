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
from citylearn.citylearn import CityLearnEnv

parser = argparse.ArgumentParser(description='Run Experiment on CityLearn')
parser.add_argument('--agent', dest="agent", type=str, required=False, default="test",
                    help='the agent.') #TODO: how to specify agent? runscript?
parser.add_argument('--seed', dest="seed", type=int, required=False, default=42,
                    help='random seed for the training process')
parser.add_argument('--episodes', dest="episodes", type=int, required=False, default=1,
                    help='number of episodes to train for')                    
parser.add_argument('--data-path', dest="data_path", type=str, required=False, default="data/citylearn_challenge_2022_phase_1",
                    help='path to dataset')
parser.add_argument("--dry-run", dest="dry_run", action="store_true", required=False,
                    help="if set, don't save anything")

args = vars(parser.parse_args())


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

# set up the environment:
class Constants:
    episodes = 1 # for evaluation only
    schema_path = os.join(args["data_path"], "schema.json")

def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }

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
    
    env = CityLearnEnv(schema=Constants.schema_path)
    agent = OrderEnforcingAgent()

    obs_dict = env_reset(env)

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(obs_dict)
    agent_time_elapsed += time.perf_counter()- step_start

    episodes_completed = 0
    num_steps = 0
    episode_metrics = []
    rewards = []
    while True:

        observations, reward, done, _ = env.step(actions)
        rewards.append(reward)

        if done:
            episodes_completed += 1
            metrics_t = env.evaluate()
            metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1]}
            if np.any(np.isnan(metrics_t)):
                raise ValueError("Episode metrics are nan, please contant organizers")
            episode_metrics.append(metrics)
            print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )
            print(np.mean(rewards, axis=0)/2, np.mean(rewards)/2)

            np.savetxt("soc.csv", env.buildings[0].electrical_storage.soc, delimiter=",")
            obs_dict = env_reset(env)

            step_start = time.perf_counter()
            actions = agent.register_reset(obs_dict)
            agent_time_elapsed += time.perf_counter()- step_start
        else:
            step_start = time.perf_counter()
            actions = agent.compute_action(observations)
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
    # train for an episode
    pass
    
    done = False
    while not done:
        # get actions from agent
        # save results from environment step
        # agent learn
        # other?
        pass
        
    
    # catch keyboard exception
    
    # after logging interval: save agent snapshot

# save raw results for later analysis.
# - training loss
# - uncertainties?