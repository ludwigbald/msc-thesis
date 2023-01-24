# Uncertainty-Aware Reinforcement Learning for Demand Response in Energy Systems
_M.Sc. thesis by Ludwig Bald_

Supervised by [Nicole Ludwig](https://www.mlsustainableenergy.com/), 2022-2023 at Uni Tübingen, Germany

- [📖 Thesis (pdf)](thesis/paper.pdf)
- [💻 Slides (pdf)](thesis-talk.pdf)

This repository contains all code, experiments, results, figures, and necessary sources for my master's thesis.
It contains modified data and code from the [2022 CityLearn challenge](https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge) and from the research paper [Estimating Risk and Uncertainty in Deep Reinforcement Learning](https://arxiv.org/abs/1905.09638), which proposes the UA-DQN algorithm.

In order to run an experiment, you need to install the packages from `requirements.txt`.
By default, the script `local_evaluation.py` runs the Rule-Based agent on the 5 public buildings from the 2022 CityLearn challenge.
The script `run_experiment.py` can be used to run DQN and UA-DQN. Hyperparameters and the Dataset can be specified as command line arguments or in the script.
The schema file `data/citylearn_challenge_2022/test_schema.json` specifies the setup used throughout the thesis.
Reward definitions can be changed in `rewards/get_reward.py`.

Explorative Data analysis is in the `exploration` folder, and experiments, along with their results and analysis in code are in the `experiments` folder.

LaTeX sources for my thesis are in the `thesis` folder.