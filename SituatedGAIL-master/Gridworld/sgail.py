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

import gridworld

import os
import sys
sys.setrecursionlimit(10000)
from datetime import datetime

def playGame(finetune=1):
    '''
    Main script. Performs s-gail on expert demos provided using TRPO.
    @param finetune: 1 if weights are already given and should be finetuned
    @return:
    '''

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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(GPU)
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    # define the model
    agent = TRPOAgent(sess, state_dim, encode_dim, action_dim, new_dir_path)

    seed = agent.config.seed
    file_path = new_dir_path + "/Main_seed.txt"
    f = open(file_path, "a")
    f.write("Seed:\n" + str(seed) + "\n")
    np.random.seed(seed)
    f.close()

    print("main seed:", seed)

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

    epoch = agent.learn(state_expert, action_expert, new_dir_path)

    print("Finish.")

if __name__ == "__main__":
    playGame()
