#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym
import numpy as np

import sys
import multiprocessing
import os.path as osp
from collections import defaultdict
import tensorflow as tf
import numpy as np

from keras.initializations import normal, identity, uniform
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Activation, Flatten, Input, merge, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import keras.backend as K
import json

from scipy.stats import multivariate_normal
from scipy.stats import mvn, norm

import copy

env = gym.make('Reacher-v1')

dtype = tf.float32

class TRPOAgent(object):

    def __init__(self, sess, state_dim, encode_dim, action_dim, actions_output_dim, filepath):
        """
        Inits a TRPO agent for visualization
        :param sess: tf object
        :param state_dim:
        :param encode_dim:
        :param action_dim:
        :param actions_output_dim:
        :param filepath: Where models are stored
        """
        self.sess = sess
        self.state_dim = state_dim
        self.encode_dim = encode_dim
        self.action_dim = action_dim
        self.actions_output_dim = actions_output_dim

        # tf placeholder vars
        # TODO: pyTorch
        self.state = state = tf.placeholder(dtype, shape=[None, state_dim])
        self.encodes = encodes = tf.placeholder(dtype, shape=[None, encode_dim])
        self.actions = actions = tf.placeholder(dtype, shape=[None, action_dim])
        self.policy = policy = tf.placeholder(dtype, shape=[None, action_dim])

        self.noise = noise = tf.placeholder(dtype, shape=[None, actions_output_dim])

        print ("Now we build trpo generator")
        self.generator = self.create_generator(state, encodes, action_dim)

        self.action_dist_mu = self.generator.outputs[0]

    #
    # Copy the Generator
    #
    def create_generator(self, state, encodes, action_dim):
        """
        Creates a G net
        :param state:
        :param encodes:
        :param action_dim:
        :return: The net object
        """
        # TODO: pyTorch
        K.set_learning_phase(1)

        s = Input(tensor=state)
        s = Dense(128)(s)
        s = LeakyReLU()(s)

        e = Input(tensor=encodes)
        e = Dense(128)(e)
        e = LeakyReLU()(e)

        h = merge([s, e], mode='sum')
        h = Dense(128)(h)
        h = LeakyReLU()(h)

        h = Dense(2, activation='tanh')(h)
        h = Lambda(lambda x: x * 0.3)(h)
        actions = Dense(2, activation='tanh')(h)

        model = Model(input=[state, encodes], output=actions)
        return model

    def act(self, state, encodes, *args):

        action_dist_mu = \
                self.sess.run(
                    self.action_dist_mu,
                    {self.state: state, self.encodes: encodes}
                )

        act = mu = copy.copy(action_dist_mu[0])
        output = np.zeros(self.actions_output_dim)

        policy = copy.copy(action_dist_mu)
        policy = policy * 0

        sigma = [[np.exp(-3.0),0.0],[0.0,np.exp(-3.0)]]

        output = np.random.multivariate_normal(mu, sigma)

        policy[0,0] = multivariate_normal.pdf(output,mu,sigma)

        return [act], [output], policy

    def delete(self, x, s_min, s_max):
        y = np.delete(x,[4,5,8,9,10])
        result = np.clip((y-s_min)/(s_max-s_min), 0.0, 1.0)
        return result

    def visualize(self, state_max, state_min, action_max, action_min, new_dir_path, epoch):
        """
        Loads weights and visualizes generated trajs.
        :param state_max:
        :param state_min:
        :param action_max:
        :param action_min:
        :param new_dir_path: Where models are stored (Together with weights)
        :param epoch:
        :return:
        """
        self.generator.load_weights(
        new_dir_path.strip("_V-function") + "/model_parametors/" + "generator_model_%d.h5" % (epoch))
        max_steps = 250

        #
        # Rendering
        #
        for i_episode in range(20):
            observation = env.reset()
            encode = np.zeros((1, self.encode_dim), dtype=np.float32)
            encode[0, 0] = 1
            print("task: ", i_episode % 2)
            steps = 0
            done = False
            for t in range(150):
                steps += 1
                env.render()
                state = self.delete(observation, state_min, state_max)

                act, action, policy = self.act([state], encode)

                observation, reward, done, info = env.step(action)

                if i_episode % 2 == 0:
                    if np.sum(abs(observation[-3:])) <= 0.018 and abs(action[0][0])< 5e-4 and abs(action[0][0])< 5e-4:
                        break

                elif i_episode % 2 == 1:
                    if np.sum(abs(observation[-3:])) >= 0.40 and abs(action[0][0])< 5e-4 and abs(action[0][0])< 5e-4:
                        break

        #env.monitor.close()