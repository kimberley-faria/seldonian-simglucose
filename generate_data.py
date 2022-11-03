#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 18:32:23 2022

@author: ctra
"""

import gym
import numpy as np

from seldonian.dataset import Episode
from seldonian.utils.io_utils import save_pickle

###########################
# Settings for the BANDIT #
###########################

S_0 = 0

NUM_SETTINGS = 20
ACTIONS = np.arange(NUM_SETTINGS).astype(np.float32) / NUM_SETTINGS # Possible P / PID_SETTINGS

################################
# Running a single simglucose episode #
################################

TARGET = 160
MIN_INSULIN = 0
MAX_INSULIN = 30

def get_insulin(f_t, P):
    # This is a simple P controller based on Blogg Glucose (BG)
    bg = f_t[0]
    
    diff = bg - TARGET

    insulin = P * diff
    
    # Clip insulin between allowed amount
    return max(MIN_INSULIN, min(MAX_INSULIN, insulin))

# Features
def get_features(observation):
    # Features = (Blood Glucose)
    features = [observation[0]]
    
    return features
    
def run_simglucose(sg_env, P, verbose=False):
    observation = sg_env.reset()
    done = False
    t = 0
    total_risk = 0
    
    
    while not done:
        # Get features and predict insulin amount using the model (pi)
        f_t = get_features(observation)
        action = get_insulin(f_t, P)
        
        
        if verbose:
            print('t: {}, obs: {}, action: {}'.format(t, observation, action))
    
        observation, risk, done, info = sg_env.step(action)
        t += 1
        total_risk += risk
        
    # print("Episode finished after {} timesteps with total reward {}".format(t, total_reward))
    
    return total_risk

############################
# Generate BANDIT episodes #
############################

def generate_random_episode(sg_env):
    # Append the single state to the history
    observations = [S_0]

    # Choose a random action
    # In this case it is a random P setting for the PID controller
    A_0 = np.random.choice(ACTIONS)
    
    # Append the action to the history as well as the probability of choosing that action
    actions = [A_0]
    prob_actions = [1. / NUM_SETTINGS]
    
    # Evaluate the action - PID setting on the simglucose env
    R_0 = run_simglucose(sg_env, A_0)
    
    # Append the reward to the history
    rewards = [R_0]
    
    # print(observations, actions, rewards, prob_actions)
    
    # That's the end of the episode, so return the history
    return Episode(observations, actions, rewards, prob_actions)


# Initialize the simglucose environment required to run BANDIT episodes
env = gym.make('simglucose-adolescent2-v0')

print('observation space: {}'.format(env.observation_space))
print('high: {}'.format(env.observation_space.high))
print('low: {}'.format(env.observation_space.low))
print()
print('action space: {}'.format(env.action_space))
print('high: {}'.format(env.action_space.high))
print('low: {}'.format(env.action_space.low))

def main():
    episodes = []
    NUM_EPISODES = 1000
    
    for i in range(NUM_EPISODES):
        print('Generating Episode: {}'.format(i))
        episode = generate_random_episode(env)
        episodes.append(episode)
    
    EPISODES_FILE = './simglucose_episodes.pkl'
    save_pickle(EPISODES_FILE, episodes)

if __name__ == '__main__':
    main()