#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import time
import json

from model_sgail_pyTorch import TRPOAgent

import gridworld

import os
import sys
sys.setrecursionlimit(10000)
from datetime import datetime

def playGame(finetune=1):
    """
    Main script. Performs s-gail on expert demos provided using TRPO.
    @param finetune: 1 if weights of generator and discriminator are already given and should be finetuned
    @return:
    """

    demo_dir = "Expert/"
    param_dir = "params/"
    state_dim = 2
    encode_dim = 3
    action_dim = 4

    GPU = 0

    #network initialize (get time)
    time = datetime.now()
    new_dir_path = "learnt_model/%d-%d %d:%d" %(time.month, time.day, time.hour, time.minute)
    os.mkdir(new_dir_path)
    print("Make directory: " + new_dir_path)

    # Configure tensorflow session and load on keras
    # TODO: Here is where pyTorch work starts

    # define the model
    agent = TRPOAgent(state_dim, encode_dim, action_dim, new_dir_path)

    seed = agent.config.seed
    file_path = new_dir_path + "/Main_seed.txt"
    f = open(file_path, "a")
    f.write("Seed:\n" + str(seed) + "\n")
    np.random.seed(seed)
    f.close()

    print("main seed:", seed)

    # Load expert demos
    state_expert = np.load(demo_dir + "state_5.npy")
    action_expert = np.load(demo_dir + "action_5.npy")
    
    # Now load the weight
    print("Now we load the weight")
    try:
        if finetune:
            agent.generator.load_weights(
                param_dir + "generator_model_188.h5")
            agent.discriminator.load_weights(
                param_dir + "discriminator_model_188.h5")
            agent.baseline.model.load_weights(
                param_dir + "baseline_model_188.h5")
        else:
            agent.generator.load_weights(
                param_dir + "weights/generator_bc_model_BC50_norm.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("Gridworld Agent learning start.")

    # Call agent learning procedure
    epoch = agent.learn(state_expert, action_expert, new_dir_path)

    print("Finish.")

if __name__ == "__main__":
    playGame()
