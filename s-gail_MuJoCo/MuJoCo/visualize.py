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

import os
import sys
sys.setrecursionlimit(10000)
from datetime import datetime

def playGame(path, seed, epoch, finetune=1):

    from model_visualize import TRPOAgent

    state_dim = 6
    encode_dim = 2
    action_dim = 2
    actions_output_dim = 2

    np.random.seed(seed)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    #
    # Define the model
    #
    agent = TRPOAgent(sess, state_dim, encode_dim, action_dim, actions_output_dim, path)
    print("MuJoCo agent visualize start.")

    #
    # Visualize
    #
    epoch = agent.visualize(state_max, state_min, action_max, action_min, path, epoch)

    print("Finish.")

if __name__ == "__main__":
    args = sys.argv
    print(args)

    #
    # Get argment (path , seed, epoch of the model)
    #
    path = "log_mujoco/" + args[1]
    seed = int(args[2])
    epoch = int(args[3])

    playGame(path, seed, epoch)
