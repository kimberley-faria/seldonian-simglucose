#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 19:37:00 2022

@author: ctra
"""

from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
from seldonian.RL.Env_Description.Env_Description import Env_Description
from seldonian.RL.Env_Description.Spaces import Discrete_Space
from seldonian.spec import createRLSpec
from seldonian.dataset import RLDataSet
from seldonian.utils.io_utils import load_pickle

def main():
    episodes_file = './simglucose_episodes.pkl'
    episodes = load_pickle(episodes_file)
    dataset = RLDataSet(episodes=episodes)
    
    # Initialize policy
    NUM_STATES = 1
    observation_space = Discrete_Space(0, NUM_STATES - 1)
    
    NUM_ACTIONS = 20    # NUM_SETTINGS in generate_data.py
    action_space = Discrete_Space(0, NUM_ACTIONS - 1)
    env_description =  Env_Description(observation_space, action_space)
    policy = DiscreteSoftmax(hyperparam_and_setting_dict={},
        env_description=env_description)
    env_kwargs={'gamma':0.9}
    save_dir = '.'
    constraint_strs = ['J_pi_new >= -50']   # New risk is grater than -10 (
                                            # determined by looking at certain P choices)
    deltas=[0.05]

    spec = createRLSpec(
        dataset=dataset,
        policy=policy,
        constraint_strs=constraint_strs,
        deltas=deltas,
        env_kwargs=env_kwargs,
        save=True,
        save_dir='.',
        verbose=True)

if __name__ == '__main__':
    main()