#!/usr/bin/env python
# -*- coding: utf-8 -*
from utils import *
import numpy as np
import time
import math
import argparse
import copy
from keras.initializations import normal, identity, uniform
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Activation, Input, merge, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import keras.backend as K
import json
from scipy.stats import multivariate_normal
from scipy.stats import mvn, norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import manifold

import os
import sys
sys.setrecursionlimit(10000)
from datetime import datetime


parser = argparse.ArgumentParser(description="TRPO")
parser.add_argument("--paths_per_collect", type=int, default=20)
parser.add_argument("--max_step_limit", type=int, default=200)
parser.add_argument("--min_step_limit", type=int, default=50)
parser.add_argument("--pre_step", type=int, default=0)
parser.add_argument("--n_iter", type=int, default=3000)
parser.add_argument("--gamma", type=float, default=.95)
parser.add_argument("--lam", type=float, default=.97)
parser.add_argument("--max_kl", type=float, default=0.01)
parser.add_argument("--cg_damping", type=float, default=0.1)
parser.add_argument("--lr_discriminator", type=float, default=1e-4)
parser.add_argument("--d_iter", type=int, default=1)
parser.add_argument("--clamp_lower", type=float, default=-0.01)
parser.add_argument("--clamp_upper", type=float, default=0.01)
parser.add_argument("--lr_baseline", type=float, default=1e-4)
parser.add_argument("--b_iter", type=int, default=1)
parser.add_argument("--g_iter", type=int, default=1)
parser.add_argument("--lr_posterior", type=float, default=1e-4)
parser.add_argument("--p_iter", type=int, default=1)                   
parser.add_argument("--buffer_size", type=int, default=60)             
parser.add_argument("--sample_size", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=256)             
parser.add_argument("--sample_size_new", type=int, default=60)
parser.add_argument("--trajectory_length", type=int, default=None)
parser.add_argument("--inner_loop", type=int, default=10)
parser.add_argument("--seed", type=int, default=1024)
parser.add_argument("--schedule", type=int, default=0)
parser.add_argument("--beta", type=float, default=.9)
parser.add_argument("--w", type=float, default=1e-3)
parser.add_argument("--step", type=int, default=1)

args = parser.parse_args()

class TRPOAgent(object):
    config = dict2(paths_per_collect = args.paths_per_collect,
                   max_step_limit = args.max_step_limit,
                   min_step_limit = args.min_step_limit,
                   pre_step = args.pre_step,
                   n_iter = args.n_iter,
                   gamma = args.gamma,
                   lam = args.lam,
                   max_kl = args.max_kl,
                   cg_damping = args.cg_damping,
                   lr_discriminator = args.lr_discriminator,
                   d_iter = args.d_iter,
                   clamp_lower = args.clamp_lower,
                   clamp_upper = args.clamp_upper,
                   lr_baseline = args.lr_baseline,
                   b_iter = args.b_iter,
                   lr_posterior = args.lr_posterior,
                   g_iter = args.g_iter,
                   p_iter = args.p_iter,
                   buffer_size = args.buffer_size,
                   sample_size = args.sample_size,
                   sample_size_new = args.sample_size_new,
                   batch_size = args.batch_size,
                   trajectory_length = args.trajectory_length,
                   seed = args.seed,
                   schedule = args.schedule,
                   beta = args.beta,
                   w = args.w,
                   step = args.step,
                   inner_loop = args.inner_loop)

    def __init__(self, sess, state_dim, encode_dim, action_dim, actions_output_dim, filepath):

        #
        # Initialize Network
        #
        self.dir_path = filepath

        self.seed = seed_initialize(filepath, self.config.seed)
        print(self.seed)

        self.sess = sess
        self.buffer = ReplayBuffer(self.config.buffer_size)
        self.buffer_new = ReplayBuffer(self.config.buffer_size)
        self.state_dim = state_dim
        self.encode_dim = encode_dim
        self.action_dim = action_dim
        self.actions_output_dim = actions_output_dim

        self.state = state = tf.placeholder(dtype, shape=[None, state_dim])
        self.state_5times = state_5times = tf.placeholder(dtype, shape=[None, state_dim*5]) ## 4時刻前までの環境を入力(10次元ベクトル)
        self.encodes = encodes = tf.placeholder(dtype, shape=[None, encode_dim])
        self.actions = actions = tf.placeholder(dtype, shape=[None, action_dim])
        self.actions_output = actions_output = tf.placeholder(dtype, shape=[None, actions_output_dim])
        self.policy = policy = tf.placeholder(dtype, shape=[None, action_dim])

        self.advants = advants = tf.placeholder(dtype, shape=[None])
        self.oldaction_dist_mu = oldaction_dist_mu = \
                tf.placeholder(dtype, shape=[None, action_dim])
        self.oldaction_dist_logstd = oldaction_dist_logstd = \
                tf.placeholder(dtype, shape=[None, action_dim])

        self.noise = noise = tf.placeholder(dtype, shape=[None, actions_output_dim])

        # Create neural network.
        print ("Now we build trpo generator")
        self.generator = self.create_generator(state, encodes, action_dim)
        print ("Now we build discriminator")
        self.discriminator = self.create_discriminator(state, actions_output, noise, encodes, policy)

        self.demo_idx = 0

        action_dist_mu = self.generator.outputs[0]
        action_dist_logstd = tf.placeholder(dtype, shape=[None, action_dim])

        eps = 1e-8
        self.action_dist_mu = action_dist_mu
        self.action_dist_logstd = action_dist_logstd
        N = tf.shape(state)[0]
        # compute probabilities of current actions and old actions
        log_p_n = gauss_log_prob(action_dist_mu, action_dist_logstd, actions)
        log_oldp_n = gauss_log_prob(oldaction_dist_mu, oldaction_dist_logstd, actions)

        ratio_n = tf.exp(log_p_n - log_oldp_n)
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advants) # Surrogate loss
        var_list = self.generator.trainable_weights

        kl = gauss_KL(oldaction_dist_mu, oldaction_dist_logstd,
                      action_dist_mu, action_dist_logstd) / Nf
        ent = gauss_ent(action_dist_mu, action_dist_logstd) / Nf

        self.losses = [surr, kl, ent]
        self.pg = flatgrad(surr, var_list)
        # KL divergence where first arg is fixed
        kl_firstfixed = gauss_selfKL_firstfixed(action_dist_mu,
                                                action_dist_logstd) / Nf
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)
        self.gf = GetFlat(self.sess, var_list)
        self.sff = SetFromFlat(self.sess, var_list)
        self.baseline = NNBaseline(sess, state, encodes, state_dim, encode_dim,
                                   self.config.lr_baseline, self.config.b_iter,
                                   self.config.batch_size, self.dir_path)
        self.sess.run(tf.global_variables_initializer())

    def create_generator(self, state, encodes, action_dim):

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

    def create_discriminator(self, state, action, noise, encode, policy):

        states = Input(tensor=state)
        actions = Input(tensor=action)
        noises = Input(tensor=noise)
        actions = merge([actions, noises], mode='sum')
        x = merge([states, actions], mode='concat')
        x = Dense(128)(x)
        x = LeakyReLU()(x)
        encodes = Input(tensor=encode)
        e = Dense(128)(encodes)
        e = LeakyReLU()(e)
        h = merge([x, e], mode='sum')
        h = Dense(128)(h)
        h = LeakyReLU()(h)
        p = Dense(1)(h)
        p = LeakyReLU()(p)

        policies = Input(tensor=policy)

        p_exp = Lambda(lambda x: K.exp(x))(p)
        policies_max = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(policies)
        p_and_actions = merge([p_exp, policies_max], mode='sum')

        p_reward = merge([p_exp, p_and_actions], mode=lambda p_reward: p_reward[0]/p_reward[1], output_shape=(1, ))

        model = Model(input=[state, action, noise, encode, policy], output=p_reward)
        adam = Adam(lr=self.config.lr_discriminator)
        model.compile(
            loss = 'binary_crossentropy', optimizer=adam #, metrics=['accuracy']
        )
        return model

    #
    # Multi variate normal distribution
    #
    def gaussian(self, x, mean, cov):
        A = 1 / ((2. * np.pi) ** (x.size/2.0))
        B = 1. / (np.linalg.det(cov) ** 0.5)
        C = -0.5 * np.dot(np.dot(x - mean, np.linalg.inv(cov)), x - mean)
        return A * B * np.exp(float(C))

    #
    # Action select
    #
    def act(self, state, encodes, logstds, *args):

        action_dist_mu = \
                self.sess.run(
                    self.action_dist_mu,
                    {self.state: state, self.encodes: encodes}
                    #{state, encodes}
                )

        act = mu = copy.copy(action_dist_mu[0])
        output = np.zeros(self.actions_output_dim)

        policy = copy.copy(action_dist_mu)
        policy = policy * 0

        sigma = [[np.exp(-3.0),0.0],[0.0,np.exp(-3.0)]]

        output = np.random.multivariate_normal(mu, sigma)

        policy[0,0] = multivariate_normal.pdf(output,mu,sigma)

        return [act], [output], policy

    def get_policy_encode(self, state, action, encodes):

        policy = np.zeros(self.action_dim, dtype=np.float32)

        G_policy = \
                self.sess.run(
                    self.action_dist_mu,
                    {self.state: state, self.encodes: encodes}
                    #{state, encodes}
                )

        mu =copy.copy(G_policy[0])

        sigma = [[np.exp(-3.0),0.0],[0.0,np.exp(-3.0)]]

        policy[0] = multivariate_normal.pdf(action,mu,sigma)

        return policy


    #
    # Learning 
    #
    def learn(self, state_expert, action_expert, action_expert_ori, s_max, s_min, a_max, a_min, new_dir_path):
        '''
        Impl. of S-GAIL learning procedure. In each iter. generates trajs from G and updates G (via TRPO) and D (Adam optimizer).
        :param state_expert:
        :param action_expert:
        :param action_expert_ori:
        :param s_max:
        :param s_min:
        :param a_max:
        :param a_min:
        :param new_dir_path:
        :return:
        '''
        # 
        # Information Writing
        #
        file_path = new_dir_path + "/Readme.txt"
        f_read = open(file_path, "a")

        f_read.write("paths_per_collect:\n" + str(self.config.paths_per_collect) + "\n")
        f_read.write("max_step_limit:\n" + str(self.config.max_step_limit) + "\n")
        f_read.write("min_step_limit:\n" + str(self.config.min_step_limit) + "\n")
        f_read.write("pre_step:\n" + str(self.config.pre_step) + "\n")
        f_read.write("n_iter:\n" + str(self.config.n_iter) + "\n")
        f_read.write("gamma:\n" + str(self.config.gamma) + "\n")
        f_read.write("lam:\n" + str(self.config.lam) + "\n")
        f_read.write("max_kl:\n" + str(self.config.max_kl) + "\n")
        f_read.write("cg_damping:\n" + str(self.config.cg_damping) + "\n")
        f_read.write("lr_discriminator:\n" + str(self.config.lr_discriminator) + "\n")
        f_read.write("d_iter:\n" + str(self.config.d_iter) + "\n")
        f_read.write("clamp_lower:\n" + str(self.config.clamp_lower) + "\n")
        f_read.write("clamp_upper:\n" + str(self.config.clamp_upper) + "\n")
        f_read.write("lr_baseline:\n" + str(self.config.lr_baseline) + "\n")
        f_read.write("b_iter:\n" + str(self.config.b_iter) + "\n")
        f_read.write("lr_posterior:\n" + str(self.config.lr_posterior) + "\n")
        f_read.write("lr_discriminator:\n" + str(self.config.lr_discriminator) + "\n")
        f_read.write("g_iter:\n" + str(self.config.g_iter) + "\n")
        f_read.write("p_iter:\n" + str(self.config.p_iter) + "\n")
        f_read.write("buffer_size:\n" + str(self.config.buffer_size) + "\n")
        f_read.write("sample_size:\n" + str(self.config.sample_size) + "\n")
        f_read.write("batch_size:\n" + str(self.config.batch_size) + "\n")
        f_read.write("trajectory_length:\n" + str(self.config.trajectory_length) + "\n")
        f_read.write("inner_loop:\n" + str(self.config.inner_loop) + "\n")
        f_read.write("seed:\n" + str(self.config.seed) + "\n")
        f_read.write("schedule:\n" + str(self.config.schedule) + "\n")
        f_read.write("beta_for_entropy:\n" + str(self.config.beta) + "\n")
        f_read.write("w:\n" + str(self.config.w) + "\n")
        f_read.write("step:\n" + str(self.config.step) + "\n")

        f_read.close()

        np.random.seed(self.seed)        
        config = self.config
        start_time = time.time()
        numeptotal = 0
        start_epoch = 0
        progress = 0
        Omax = self.config.sample_size

        state_d = state_expert
        actions_d = action_expert
        action_d_no_norm = action_expert_ori

        #
        # Learning rate
        #
        losses = []
        generator_loss = []
        generator_D = []
        generator_P = []
        posterior_loss = []
        baseline_loss = []

        #
        # Goal Agent
        #
        count_goalagent = 0
        count_goalagent_stack = []
        count_goalagent_new_stack = []
    

        for i in range(start_epoch, start_epoch+config.n_iter):

            progress += 1

            #
            # Load encode expert data
            #
            demo_dir = "Expert/"
            encodes_d = np.load(demo_dir + "encode_mujoco.npy")

            encode_labels = [(0,1) for num_expert in range(30)]
            encode_labels = list(np.array(encode_labels).flatten())

            #
            # Create Generator's trajectory
            #
            count_goalagent = 0
            Omax_sample = 60

            rollouts, count_goalagent = rollout_contin(
                self,
                self.state_dim,
                self.encode_dim,
                self.action_dim,
                s_max, 
                s_min,
                a_max, 
                a_min,
                config.max_step_limit,
                config.min_step_limit,
                Omax_sample,
                count_goalagent,
                i,
                )
            latent_labels = [r for r in range(self.encode_dim)]
            sub_labels = []
            for r in range(Omax_sample/self.encode_dim):
                sub_labels.extend(latent_labels)

            count_goalagent_stack.append(count_goalagent)

            #
            # Pool trajecoty
            #
            index_for_labels = 0

            for path in rollouts:
                path["labels"] = [sub_labels[index_for_labels]]
                index_for_labels +=1

            for path in rollouts:
                self.buffer.add(path)

            paths = self.buffer.get_sample(config.sample_size)

            #
            # Rearrange paths
            #
            logstds_n = np.concatenate([path["logstds"] for path in paths])
            state_n = np.concatenate([path["state"] for path in paths]) 
            encodes_n = np.concatenate([path["encodes"] for path in paths])
            actions_n = np.concatenate([path["act"] for path in paths])
            labels_n = np.concatenate([path["labels"] for path in paths])
            policies_n = np.concatenate([path["policies"] for path in paths])

            #
            # Pre-run Discriminator
            #
            numdetotal = state_d.shape[0]
            numnototal = state_n.shape[0]
            batch_size = config.batch_size
            start_d = 0
            start_n = 0
            d_iter = self.config.d_iter

            loss_sum = 0

            idx = np.arange(numnototal)
            np.random.shuffle(idx)
            train_val_ratio = 0.7

            #
            # Inner loop Discriminator & Generator
            #
            if i == start_epoch:
                new_labels = []
                new_labels = encode_labels
                if not os.path.exists(new_dir_path + "/new_labels"):
                    os.mkdir(new_dir_path + "/new_labels")
                np.save(new_dir_path + "/new_labels/iter_%d_labels.npy" %(i), np.array(new_labels))

            for iiter in range(self.config.inner_loop):
                if iiter % 10 == 0:
                    print("Inner: ", iiter)
                if iiter % 50 == 0:
                    print(new_dir_path)
                
                count_goalagent_new =0

                #
                # Create trajectory while over minimal batch_size
                #
                new_labels = encode_labels
                counter_batchsize = 0
                rollouts_new, count_goalagent_new = rollout_contin(
                    self,
                    self.state_dim,
                    self.encode_dim,
                    self.action_dim,
                    s_max, 
                    s_min,
                    a_max, 
                    a_min,
                    config.max_step_limit,
                    config.min_step_limit,
                    Omax,
                    count_goalagent_new,
                    i,
                    encode_list = new_labels
                    )
                counter_batchsize = len(np.concatenate([path["state"] for path in rollouts_new]))
                print("GOAL: ", count_goalagent_new)

                if iiter == self.config.inner_loop -1:
                    count_goalagent_new_stack.append(count_goalagent_new)

                index_for_labels = 0
                for path in rollouts_new:
                    path["labels"] = [new_labels[index_for_labels]]
                    index_for_labels +=1

                #
                # Pool trajectory (for buffer_new)
                #
                for path in rollouts_new:
                    self.buffer_new.add(path)

                paths_new = self.buffer_new.get_sample(config.sample_size)

                #
                # Create Trajectory
                #
                for path in paths_new:
                    path["mus"] = self.sess.run(
                        self.action_dist_mu,
                        {self.state: path["state"],
                         self.encodes: path["encodes"]}
                    )

                #
                # Rearrange paths_new
                #
                mus_new_n = np.concatenate([path["mus"] for path in paths_new])
                logstds_new_n = np.concatenate([path["logstds"] for path in paths_new])
                state_new_n = np.concatenate([path["state"] for path in paths_new]) 
                encodes_new_n = np.concatenate([path["encodes"] for path in paths_new])
                actions_new_n = np.concatenate([path["act"] for path in paths_new])
                actions_output_new_n = np.concatenate([path["actions"] for path in paths_new])
                labels_new_n = np.concatenate([path["labels"] for path in paths_new])
                policies_new_n = np.concatenate([path["policies"] for path in paths_new])

                #
                # Preparation for update Networks
                #
                numdetotal = state_d.shape[0]
                numnototal = state_new_n.shape[0]
                start_d = 0
                start_n = 0

                #
                # Get policy from expert data
                #
                policy_d = []
                for t in range(len(state_d)):
                    policy_d.append(self.get_policy_encode([state_d[t]], action_d_no_norm[t], [encodes_d[t]]))

                policy_d = np.array(policy_d)

                batch_size_agent = len(state_new_n)
                batch_size_expert = len(state_d)

                noise_expert = np.random.normal(0, 0.05, (batch_size_expert, 2))
                noise_agent = np.random.normal(0, 0.05, (batch_size_agent, 2))
                #noise_expert = np.zeros((batch_size_expert, 2), dtype=np.float16)
                #noise_agent = np.zeros((batch_size_agent, 2), dtype=np.float16)

                #
                # Update Discriminator (using Keras)
                #
                d_loss_real = self.discriminator.train_on_batch([state_d[start_d:start_d + batch_size_expert],actions_d[start_d:start_d + batch_size_expert],noise_expert[:batch_size_expert],encodes_d[start_d:start_d + batch_size_expert],policy_d[start_d:start_d + batch_size_expert]], np.ones(batch_size_expert))
                d_loss_fake = self.discriminator.train_on_batch([state_new_n[start_n:start_n + batch_size_agent],actions_new_n[start_n:start_n + batch_size_agent],noise_agent[:batch_size_agent],encodes_new_n[start_n:start_n + batch_size_agent],policies_new_n[start_n:start_n + batch_size_agent]], np.zeros(batch_size_agent))
                loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    l.set_weights(weights)

                if iiter == self.config.inner_loop - 1:
                    losses.append(loss)

                if iiter == self.config.inner_loop - 1:
                    path_idx = 0
                    start_w = 0

                    for path in paths_new:
                        file_path_world = new_dir_path + "/Generator_trajectories_resample"
                        if not os.path.exists(file_path_world):
                            os.mkdir(file_path_world)
                        file_path_world = new_dir_path + "/Generator_trajectories_resample/iter_%d_world_%d" % (i, path_idx)

                        np.save(file_path_world, path["state"])
                        np.save(file_path_world + "_actions.npy", actions_new_n[start_w:start_w +  path["length"]])
                        np.save(file_path_world + "_policies.npy", policies_new_n[start_w:start_w + path["length"]])
                        np.save(file_path_world + "_labels.npy", labels_new_n[path_idx])

                        path_idx += 1

                #
                # Update Generator 
                #
                path_i = 0
                for path in paths_new:

                    path["baselines"] = self.baseline.predict(path)

                    output_d = self.discriminator.predict(
                        [path["state"], path["actions"], noise_agent, path["encodes"], path["policies"]])

                    anti = 1.0 - copy.copy(output_d)

                    for an in range(path["length"]):
                        if anti[an] == 0.0:
                            anti[an] = 1e-10

                    #
                    # With scheduling
                    #
                    if i % self.config.step == 0:
                        if self.config.schedule == 1:
                            beta = self.config.beta - self.config.w * i
                    #
                    # Without scheduling
                    #
                    else:
                        beta = self.config.beta

                    if i % 50 == 0 and path_i == 0:
                        file_path2 = new_dir_path + "/bata.txt"
                        f_read2 = open(file_path2, "a")
                        f_read2.write(str(i) +  " : " + str(beta) + "\n")
                        f_read2.close()

                    path_i += 1

                    #output_d_for_reward = np.log(output_d + 1e-10) - np.log(anti)
                    output_d_for_reward = ((np.log(output_d + 1e-10) - np.log(anti)).flatten() + beta * np.log(np.max(path["policies"], axis=1) + 1e-10).flatten())

                    '''
                    if path["flag"] == 1:
                        output_d_for_reward = output_d_for_reward + np.ones(path["length"]) * 0.1 * np.exp(3*abs((float(config.max_step_limit)-float(path["length"]))/float(config.max_step_limit)))
                    '''

                    path["rewards"] = output_d_for_reward.flatten()

                    path["generator_d"] = output_d_for_reward.flatten()

                    path_baselines = np.append(path["baselines"], 0 if
                                               path["flag"] == 1 else
                                               path["baselines"][-1])

                    deltas = path["rewards"] + config.gamma * path_baselines[1:] -\
                            path_baselines[:-1]

                    path["advants"] = discount(deltas, config.gamma * config.lam)
                    path["returns"] = discount(path["rewards"], config.gamma)

                    path["baseline"] = copy.copy(path_baselines)

                if iiter == self.config.inner_loop - 1:
                    path_idx = 0
                    for path in paths_new:
                        file_path = new_dir_path + "/returns(resample)"
                        if not os.path.exists(file_path):
                            os.mkdir(file_path)
                        file_path = new_dir_path + "/returns(resample)/iter_%d_path_%d.txt" % (i, path_idx)
                        f = open(file_path, "w")

                        f.write("Baseline:\n" + np.array_str(path["baseline"]) + "\n")
                        f.write("Returns:\n" + np.array_str(path["returns"]) + "\n")
                        f.write("Advants:\n" + np.array_str(path["advants"]) + "\n")
                        f.write("Mus:\n" + np.array_str(path["mus"]) + "\n")
                        f.write("Actions:\n" + np.array_str(path["actions"]) + "\n")
                        f.write("States:\n" + np.array_str(path["state"]) + "\n")
                        f.write("Logstds:\n" + np.array_str(path["logstds"]) + "\n")

                        path_idx += 1

                        f.close()

                #
                # Standardize the advantage function to have mean=0 and std=1
                #
                advants_new_n = np.concatenate([path["advants"] for path in paths_new])
                advants_new_n /= (advants_new_n.std() + 1e-8)

                if iiter == self.config.inner_loop - 1:

                    #
                    # Calcurate Generator's reward(sum)
                    #
                    episoderewards = np.array([path["rewards"].sum() for path in paths_new])
                    generator_d = np.array([path["generator_d"].sum() for path in paths_new])

                    #
                    # Calcurate Generator's reward(average each epochs)
                    #
                    generator_loss.append(episoderewards.mean())
                    generator_D.append(generator_d.mean())

                #
                # Computing baseline function for next iter.
                #
                b_loss = self.baseline.fit(paths_new, batch_size_agent)
                if iiter == self.config.inner_loop - 1:
                    baseline_loss.append(b_loss)

                feed = {self.state: state_new_n,
                        self.encodes: encodes_new_n,
                        self.actions: actions_output_new_n,
                        self.advants: advants_new_n,
                        self.action_dist_logstd: logstds_new_n,
                        self.oldaction_dist_mu: mus_new_n,
                        self.oldaction_dist_logstd: logstds_new_n}

                thprev = self.gf()

                def fisher_vector_product(p):
                    feed[self.flat_tangent] = p
                    return self.sess.run(self.fvp, feed) + p * config.cg_damping

                g = self.sess.run(self.pg, feed_dict=feed)
                stepdir = conjugate_gradient(fisher_vector_product, -g)
                shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                if shs <= 0:
                    break 

                lm = np.sqrt(shs / config.max_kl)
                fullstep = stepdir / lm
                neggdotstepdir = -g.dot(stepdir)

                def loss(th):
                    self.sff(th)
                    return self.sess.run(self.losses[0], feed_dict=feed)

                theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
                self.sff(theta)

                surrafter, kloldnew, entropy = self.sess.run(
                    self.losses, feed_dict=feed
                )

            #
            # Display learn results
            #
            stats = {}
            numeptotal += len(episoderewards)
            stats["Total number of episodes"] = numeptotal
            stats["Average sum of rewards per episode"] = episoderewards.mean()
            stats["Entropy"] = entropy
            stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
            stats["KL between old and new distribution"] = kloldnew
            stats["Surrogate loss"] = surrafter
            print("\n********** Iteration {} **********".format(i))
            for k, v in stats.iteritems():
                print(k + ": " + " " * (40 - len(k)) + str(v))
            if entropy != entropy:
                exit(-1)

            #
            # parameter saving
            #
            param_dir = new_dir_path + "/model_parametors/"
            if not os.path.exists(new_dir_path + "/model_parametors"):
                os.mkdir(new_dir_path + "/model_parametors")

            print("Now we save model")
            self.generator.save_weights(
                param_dir + "generator_model_%d.h5" % i, overwrite=True)
            with open(param_dir + "generator_model_%d.json" % i, "w") as outfile:
                json.dump(self.generator.to_json(), outfile)

            self.discriminator.save_weights(
                param_dir + "discriminator_model_%d.h5" % i, overwrite=True)
            with open(param_dir + "discriminator_model_%d.json" % i, "w") as outfile:
                json.dump(self.discriminator.to_json(), outfile)

            self.baseline.model.save_weights(
                param_dir + "baseline_model_%d.h5" % i, overwrite=True)
            with open(param_dir + "baseline_model_%d.json" % i, "w") as outfile:
                json.dump(self.baseline.model.to_json(), outfile)

            outfile.close()

            #
            # Save for loss data
            #
            file_results = new_dir_path + "/results"
            if not os.path.exists(file_results):
                os.mkdir(file_results)
            file_results = new_dir_path + "/results/"

            #
            # Learning rate of Discriminator & Generator
            #
            losses_A = np.array(losses)
            generator_loss_A = np.array(generator_loss)
            np.save(file_results+ "reward_Generator.npy", generator_loss_A)
            np.save(file_results+ "loss_Discriminator.npy", losses_A)

            #
            # Learning rate of Baseline
            #
            baseline_loss_A = np.array(baseline_loss)
            np.save(file_results+ "baseline_loss.npy", baseline_loss_A)

            #
            # Generator's reward
            #
            generator_D_A = np.array(generator_D)
            np.save(file_results+ "generator_D.npy", generator_D_A)
            
            #
            # Number of goal agent
            #
            goal_agent_A = np.array(count_goalagent_stack)
            np.save(file_results+ "number_of_goal_agent.npy", goal_agent_A)

            #
            # Making directory
            #
            file_values = new_dir_path + "/value_function"
            if not os.path.exists(file_values):
                os.mkdir(file_values)

            for enc_dir in range(self.encode_dim):
                file_values_enc = new_dir_path + "/value_function/encode_%d" % (enc_dir)
                if not os.path.exists(file_values_enc):
                    os.mkdir(file_values_enc)

        #
        # Plot learning rate
        #
        file_results = new_dir_path + "/results"
        if not os.path.exists(file_results):
            os.mkdir(file_results)
        file_results = new_dir_path + "/results/"

        #
        # Learning rate of Discriminator & Generator
        #
        losses = np.array(losses)
        generator_loss = np.array(generator_loss)
        t = np.arange(progress)
        fig = plt.figure(figsize = (15,15))
        ax = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        ax.plot(t,losses)
        ax.set_title("Discriminator_loss")
        ax2.plot(t,generator_loss)
        ax2.set_title("Generator_reward")
        plt.savefig(new_dir_path + '/loss.png')
        np.save(file_results+ "reward_Generator.npy", generator_loss)
        np.save(file_results+ "loss_Discriminator.npy", losses)

        #
        # Learning rate of Baseline
        #
        baseline_loss = np.array(baseline_loss)
        t = np.arange(progress)
        fig = plt.figure(figsize = (15,15))
        ax = fig.add_subplot(1,1,1)
        ax.plot(t,baseline_loss)
        ax.set_title("baseline_loss")
        plt.savefig(new_dir_path + '/baseline_loss.png')
        np.save(file_results+ "baseline_loss.npy", baseline_loss)
        
        #
        # Number of goal agent
        #
        goal_agent = np.array(count_goalagent_stack)
        t = np.arange(progress)
        fig = plt.figure(figsize = (15,15))
        ax = fig.add_subplot(1,1,1)
        ax.plot(t, goal_agent)
        ax.set_title("Goal_agent")
        plt.savefig(new_dir_path + '/goal_agent.png')

        return progress