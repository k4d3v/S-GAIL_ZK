#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import time
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json

from model_sgail import TRPOAgent

import os
import sys
sys.setrecursionlimit(10000)
from datetime import datetime

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def norm(x, a_min, a_max):
    result = np.clip((x-a_min)/(a_max-a_min), 0.0, 1.0)
    return result

def playGame(finetune=1):
    """
    Main procedure. Initializes a TRPO agent and learns based on expert`s trajectories provided in dir Expert using S-GAIL algo.
    :param finetune: 1 if weights are already given and to be finetunned
    :return:
    """
    # Directory of demos, state and action dim
    demo_dir = "Expert/"
    param_dir = "params_MuJoCo/"
    state_dim = 6
    encode_dim = 2
    action_dim = 2
    actions_output_dim = 2

    GPU = 0

    time = datetime.now()
    new_dir_path = "log_mujoco/%d-%d %d:%d" %(time.month, time.day, time.hour, time.minute)
    os.mkdir(new_dir_path)
    print("Make directory: " + new_dir_path)

    # TODO: tf to pyTorch
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(GPU)
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    # define the model
    agent = TRPOAgent(sess, state_dim, encode_dim, action_dim, actions_output_dim, new_dir_path)

    print("agent_path: ", agent.config.trajectory_length)
    path_length = agent.config.trajectory_length

    seed = agent.config.seed
    file_path = new_dir_path + "/Main_seed.txt"
    f = open(file_path, "a")
    f.write("Seed:\n" + str(seed) + "\n")
    np.random.seed(seed)
    f.close()

    print("main seed:", seed)

    state_expert = np.load(demo_dir + "state_mujoco.npy")
    action_expert = np.load(demo_dir + "action_mujoco.npy")

    #
    # Normalize & Get Min-Max
    #
    state_expert_norm = min_max(state_expert,axis=0)
    action_expert_norm = min_max(action_expert, axis=0)

    state_max = np.max(state_expert, axis=0)
    state_min = np.min(state_expert, axis=0)
    action_max = np.max(action_expert, axis=0) # = [0.2112, 0.3219]
    action_min = np.min(action_expert, axis=0) # = [-0.1343, -0.0819]


    #
    # Now load the weight
    #
    print("Now we load the weight")
    try:
        if finetune:
            agent.generator.load_weights(
                param_dir + "generator_model_299.h5")
            agent.discriminator.load_weights(
                param_dir + "discriminator_model_299.h5")
            agent.baseline.model.load_weights(
                param_dir + "baseline_model_299.h5")
        else:
            agent.generator.load_weights(
                param_dir + "generator_bc_model_BC_norm.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("MuJoCo Agent learning start.")

    epoch = agent.learn(state_expert_norm, action_expert_norm, action_expert, state_max, state_min, action_max, action_min, new_dir_path)

    print("Finish.")

if __name__ == "__main__":
    playGame()
