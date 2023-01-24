import numpy as np
import pandas as pd
import time

from agents.orderenforcingwrapper import OrderEnforcingAgent
from citylearn.citylearn import CityLearnEnv


class Constants:
    episodes = 1
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'

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

def discretize(actions, n_discrete):
    if n_discrete != 0:
        d = np.floor(n_discrete/2)
        return [np.round(action * d)/d for action in actions]
    else:
        return actions

def evaluate(n_discrete, result):
    print("Starting local evaluation")
    
    env = CityLearnEnv(schema=Constants.schema_path)
    agent = OrderEnforcingAgent()

    obs_dict = env_reset(env)

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = discretize(agent.register_reset(obs_dict), n_discrete)
    agent_time_elapsed += time.perf_counter()- step_start

    episodes_completed = 0
    num_steps = 0
    interrupted = False
    episode_metrics = []
    rewards = []
    actionss = []
    obss = []
    try:
        while True:

            observations, reward, done, _ = env.step(actions)
            rewards.append(reward)
            actionss.append(actions[0])
            obss.append(observations[0])

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
                np.savetxt("actions.csv", actionss, delimiter=",")
                np.savetxt("rewards.csv", [r[0] for r in rewards], delimiter=",")
                np.savetxt("obs.csv", obss, delimiter=",")
                obs_dict = env_reset(env)

                step_start = time.perf_counter()
                actions = discretize(agent.register_reset(obs_dict), n_discrete)
                agent_time_elapsed += time.perf_counter()- step_start
            else:
                step_start = time.perf_counter()
                actions = discretize(agent.compute_action(observations), n_discrete)
                agent_time_elapsed += time.perf_counter()- step_start
            
            num_steps += 1
            if num_steps % 1000 == 0:
                print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

            if episodes_completed >= Constants.episodes:
                break
    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True
    
    if not interrupted:
        print("=========================Completed=========================")

    if len(episode_metrics) > 0:
        print("Average Price Cost:", np.mean([e['price_cost'] for e in episode_metrics]))
        print("Average Emmision Cost:", np.mean([e['emmision_cost'] for e in episode_metrics]))
        print("Average of both costs:", (np.mean([e['price_cost'] for e in episode_metrics]) + np.mean([e['emmision_cost'] for e in episode_metrics]))/2)
    print(f"Total time taken by agent: {agent_time_elapsed}s")
    return (n_discrete,
            np.mean([e['price_cost'] for e in episode_metrics]),
            np.mean([e['emmision_cost'] for e in episode_metrics]))
    

if __name__ == '__main__':
    complete_start = time.perf_counter()

    n_discretes = []
    cost_scores = []
    carbon_scores = []

    result = pd.DataFrame(columns=["n_discrete", "cost_score", "carbon_score"])
    for n_discrete in range(0, 21):
        the_n_discrete, cost_score, carbon_score = evaluate(n_discrete, result)
        n_discretes.append(the_n_discrete)
        cost_scores.append(cost_score)
        carbon_scores.append(carbon_score)
    result = pd.DataFrame(zip(n_discretes, cost_scores, carbon_scores))
    result.to_csv("result.csv")
    print("The entire evaluation took: {elapsed}s".format(elapsed = (time.perf_counter()- complete_start)))