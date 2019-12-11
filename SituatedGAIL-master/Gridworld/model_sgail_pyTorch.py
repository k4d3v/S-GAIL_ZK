#!/usr/bin/env python
# -*- coding: utf-8 -*
import torch
from torch import nn
from torch.autograd import Variable
from utils_pyTorch import *
import numpy as np
import time
import math
import argparse
import copy
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import manifold

import gridworld

import os
import sys

sys.setrecursionlimit(10000)
from datetime import datetime

# import pyclustering
# from pyclustering.cluster import xmeans

# Configuration of hyperparams
parser = argparse.ArgumentParser(description="TRPO")
parser.add_argument("--paths_per_collect", type=int, default=10)  # Number of trajectory (20)
parser.add_argument("--max_step_limit", type=int, default=25)  # Max step per episode
parser.add_argument("--min_step_limit", type=int, default=0)  # Min step per episode
parser.add_argument("--n_iter", type=int, default=7)  # Epoch (300)
parser.add_argument("--gamma", type=float, default=.95)  # Discount factor (MDP)
parser.add_argument("--lam", type=float, default=.97)  # Lamda (TRPO)
parser.add_argument("--max_kl", type=float, default=0.01)  # Max of KL-divergence
parser.add_argument("--cg_damping", type=float, default=0.1)  # Damping (TRPO)
parser.add_argument("--lr_discriminator", type=float, default=1e-4)  # Learning rate of Discriminator
parser.add_argument("--lr_baseline", type=float, default=1e-4)  # Learning rate of Baseline
parser.add_argument("--b_iter", type=int, default=1)  # Inner loop of Baseline
parser.add_argument("--buffer_size", type=int, default=60)  # Size of Replaybuffer
parser.add_argument("--sample_size", type=int, default=60)  # Sample size (if batch learning, buffer size = sample size)
parser.add_argument("--batch_size", type=int, default=256)  # Batch size for network update (samplesize * trajectory length)
parser.add_argument("--inner_loop", type=int, default=50)  # Inner loop (100)
parser.add_argument("--seed", type=int, default=1024)  # Seed
parser.add_argument("--schedule", type=int, default=0)  # Beta scheduling
parser.add_argument("--beta", type=float, default=.9)  # Beta
parser.add_argument("--w", type=float, default=0)  # Scheduling weight of beta

args = parser.parse_args()


class TRPOAgent(object):
    """
    Represents the learning agent with its D and G nets and TRPO procedure
    """
    # Configuration of the agent containing all the hyperparams
    config = dict2(paths_per_collect=args.paths_per_collect,
                   max_step_limit=args.max_step_limit,
                   min_step_limit=args.min_step_limit,
                   n_iter=args.n_iter,
                   gamma=args.gamma,
                   lam=args.lam,
                   max_kl=args.max_kl,
                   cg_damping=args.cg_damping,
                   lr_discriminator=args.lr_discriminator,
                   lr_baseline=args.lr_baseline,
                   b_iter=args.b_iter,
                   buffer_size=args.buffer_size,
                   sample_size=args.sample_size,
                   batch_size=args.batch_size,
                   seed=args.seed,
                   schedule=args.schedule,
                   beta=args.beta,
                   w=args.w,
                   inner_loop=args.inner_loop)

    def __init__(self, state_dim, encode_dim, action_dim, filepath):
        """
        Initialize
        @param state_dim: dimension of state space
        @param encode_dim: See paper. Used for task var.
        @param action_dim: dimension of action space
        @param filepath: path where model should be saved
        """
        self.dir_path = filepath

        self.seed = seed_initialize(filepath, self.config.seed)
        print(self.seed)

        # Set variables from constructor
        self.buffer = ReplayBuffer(self.config.buffer_size)
        self.buffer_new = ReplayBuffer(self.config.buffer_size)
        self.state_dim = state_dim
        self.encode_dim = encode_dim
        self.action_dim = action_dim

        # TODO: Change following lines according to pyTorch notation.
        # Some important vars
        self.policy = policy = action_dim
        self.advants = advants = Variable()  # torch.zeros(None)
        self.oldaction_dist_mu = oldaction_dist_mu = Variable()  # torch.zeros(None, action_dim)
        self.oldaction_dist_logstd = oldaction_dist_logstd = Variable()  # torch.zeros(None, action_dim)

        self.noise = noise = action_dim  # [None, action_dim]
        self.beta = beta = Variable()  # [None, 1])

        # Create D and G
        print("Now we build trpo generator")
        self.generator = self.create_generator(state_dim, encode_dim, action_dim)  # TODO
        print("Now we build discriminator")
        self.discriminator = self.create_discriminator(state_dim, encode_dim, action_dim, noise, policy)  # TODO

        # Looks like important vars used during training like log probs and gradients
        self.demo_idx = 0

        # TODO: Change following lines according to pyTorch notation.
        action_dist_mu = self.generator.action_mean.bias
        action_dist_logstd = self.generator.action_log_std

        eps = 1e-8
        self.action_dist_mu = action_dist_mu
        self.action_dist_logstd = action_dist_logstd
        N = tf.shape(state_dim)[0]
        log_p_n = gauss_log_prob(action_dist_mu, action_dist_logstd, action_dim)
        log_oldp_n = gauss_log_prob(oldaction_dist_mu, oldaction_dist_logstd, action_dim)

        ratio_n = tf.exp(log_p_n - log_oldp_n)
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advants)
        var_list = self.generator.trainable_weights  # Save weights of G

        kl = gauss_KL(oldaction_dist_mu, oldaction_dist_logstd,
                      action_dist_mu, action_dist_logstd) / Nf
        ent = gauss_ent(action_dist_mu, action_dist_logstd) / Nf

        self.losses = [surr, kl, ent]
        self.pg = flatgrad(surr, var_list)
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
        # Create NN baseline based on expert demos for predicting actions. Will be used for training the Generator.
        self.baseline = NNBaseline(state_dim, encode_dim,
                                   self.config.lr_baseline, self.config.b_iter,
                                   self.config.batch_size, self.dir_path)
        # self.sess.run(tf.global_variables_initializer()) # TODO: What is this?
        # TODO End

        # Create a grid object to store the values
        self.GridMDP = gridworld.GridMDP([[0, 0, 0, 0, 0, None, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, None, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, None, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [None, None, None, 0, 0, 0, 0, 0, None, None, None],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, None, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, None, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, None, 0, 0, 0, 0, 0]], terminals=[(1.0, 1.0)])

    def create_generator(self, state, encodes, action_dim):
        """
        Create the generator network
        @param state: state dimension
        @param encodes: dimension for task var.
        @param action_dim: dimension of action
        @return: The newly built model
        """
        # TODO: Debug
        class G(nn.Module):
            def __init__(self, s, e, a):
                """
                :param s: State dim
                :param e: Encode dim
                :param a: Action dim
                """
                super(G, self).__init__()
                self.inp = nn.Linear(s + e, 128)
                self.h1 = nn.Linear(128, 128)
                self.h2 = nn.Linear(128, 128)

                self.action_mean = nn.Linear(128, a)
                self.action_mean.weight.data.mul_(0.1)
                self.action_mean.bias.data.mul_(0.0)

                self.action_log_std = nn.Parameter(torch.zeros(1, a), requires_grad=True)

                self.saved_actions = []
                self.rewards = []
                self.final_value = 0

            def forward(self, x):
                """
                Forward computation
                :param x: Input val
                :return: Action mean, log_std and std
                """
                x = lru(self.inp(x))
                x = lru(self.h1(x))
                x = lru(self.h2(x))

                action_mean = self.action_mean(x)
                action_log_std = self.action_log_std.expand_as(action_mean)
                action_std = torch.exp(action_log_std)

                return action_mean, action_log_std, action_std

        return G(state, encodes, action_dim)

    def create_discriminator(self, state, encode, action, noise,  policy):
        """
        Creates the discriminator net
        @param state: Dimension of state space
        @param action: Dimension of action space
        @param noise: Has dimension of action space
        @param encode: Dimension of task variable
        @param policy: A tf placeholder. Will contain the policy (action vector)
        @return: Newly built tf model
        """
        # TODO: Debug
        class D(nn.Module):
            def __init__(self, s, e, a, n, p):
                """
                :param s: State dim
                :param e: Encode dim
                :param a: Action dim
                :param n: Noise dim
                :param p: Policy dim
                """
                super(D, self).__init__()
                self.inp = nn.Linear(s + a + n + e, 128)
                self.h1 = nn.Linear(128, 128)
                self.h2 = nn.Linear(128, 128)
                self.h3 = nn.Linear(128, 1)
                self.p = p

                self.action_mean = nn.Linear(1, a)
                self.action_mean.weight.data.mul_(0.1)
                self.action_mean.bias.data.mul_(0.0)

                self.outputs = nn.Parameter(torch.zeros(1, a), requires_grad=True)
                self.action_log_std = nn.Parameter(torch.zeros(1, a), requires_grad=True)

                self.saved_actions = []
                self.rewards = []
                self.final_value = 0

            def forward(self, xin):
                """
                Forward computation. See S-GAIL paper for details.
                :param xin: Input val (action-state-encode + policy)
                :return: Action mean, log_std and std
                """
                p_start = len(xin) - 1 - self.p
                x = xin[:p_start]
                x = lru(self.inp(x))
                x = lru(self.h1(x))
                x = lru(self.h2(x))
                x = lru(self.h3(x))
                x = Lambda(lambda y: np.exp(y))(x)
                max_pin = np.max(xin[p_start:], axis=1, keepdims=True)
                out = Lambda(lambda p_reward: p_reward[0] / p_reward[1], output_shape=(1,))(np.concatenate((x, max_pin), 0))

                action_mean = self.action_mean(out)
                action_log_std = self.action_log_std.expand_as(action_mean)
                action_std = torch.exp(action_log_std)

                return action_mean, action_log_std, action_std

        return D(state, encode, action, noise, policy)

    #
    # Action select
    #
    def act(self, state, encodes, logstds, *args):
        """
        Sample actions and construct policy array. Used during rollout (See utils_pyTorch.py)
        @param state: Current state
        @param encodes: Task var specific
        @param logstds: Hardcoded to 0 in utils line 188
        @param args:
        @return: Sampled action and policy containing zeros + action
        """
        action_dist_mu = \
            self.sess.run(
                self.action_dist_mu,
                {self.state_dim: state, self.encode_dim: encodes}
            )

        act = copy.copy(action_dist_mu)
        policy = copy.copy(action_dist_mu)
        policy = policy * 0

        flag = 0
        while (flag == 0):
            random_action = random.random()
            if random_action < 1:
                flag = 1

        action_dir = 100
        if random_action < act[0][0]:
            action_dir = 0
        elif random_action < act[0][0] + act[0][1]:
            action_dir = 1
        elif random_action < act[0][0] + act[0][1] + act[0][2]:
            action_dir = 2
        else:
            action_dir = 3

        for actor in range(len(act[0])):
            if actor == action_dir:
                act[0, actor] += 1.0
            else:
                act[0, actor] -= 1.0

        act[:, 0] = np.clip(act[:, 0], 0, 1)
        act[:, 1] = np.clip(act[:, 1], 0, 1)
        act[:, 2] = np.clip(act[:, 2], 0, 1)
        act[:, 3] = np.clip(act[:, 3], 0, 1)

        policy[0, action_dir] = action_dist_mu[0][action_dir]

        return act, policy

    def get_policy(self, state, argmax_action):

        policy = np.zeros(self.action_dim, dtype=np.float32)

        for i in range(self.encode_dim):
            encodes = np.zeros((1, self.encode_dim), dtype=np.float32)
            encodes[0, i] = 1
            softmax_policy = \
                self.sess.run(
                    self.action_dist_mu,
                    {self.state_dim: state, self.encode_dim: encodes}
                )
            policy[argmax_action] += softmax_policy[0][argmax_action] / self.encode_dim
        return policy

    def get_policy_encode(self, state, argmax_action, encodes):
        """
        Receives a state, index of best action and current policy index and returns action based on policy
        :param state:
        :param argmax_action:
        :param encodes: State variable
        :return: Sampled action
        """

        policy = np.zeros(self.action_dim, dtype=np.float32)

        # TODO: Session again
        # Sample action for each state based on current policy
        softmax_policy = \
            self.sess.run(
                self.action_dist_mu,
                {self.state_dim: state, self.encode_dim: encodes}
            )

        policy[argmax_action] = softmax_policy[0][argmax_action]

        return policy

    def learn(self, state_expert, action_expert, new_dir_path):
        """
        Learning procedure
        @param state_expert: States sampled from demos
        @param action_expert: Actions sampled from demos
        @param new_dir_path: Where the model is going to get stored
        @return: Counter for iterations after finish
        """
        # 
        # Information Writing
        #
        file_path = new_dir_path + "/Readme.txt"
        f_read = open(file_path, "a")

        f_read.write("paths_per_collect:\n" + str(self.config.paths_per_collect) + "\n")
        f_read.write("max_step_limit:\n" + str(self.config.max_step_limit) + "\n")
        f_read.write("min_step_limit:\n" + str(self.config.min_step_limit) + "\n")
        f_read.write("n_iter:\n" + str(self.config.n_iter) + "\n")
        f_read.write("gamma:\n" + str(self.config.gamma) + "\n")
        f_read.write("lam:\n" + str(self.config.lam) + "\n")
        f_read.write("max_kl:\n" + str(self.config.max_kl) + "\n")
        f_read.write("cg_damping:\n" + str(self.config.cg_damping) + "\n")
        f_read.write("lr_discriminator:\n" + str(self.config.lr_discriminator) + "\n")
        f_read.write("lr_baseline:\n" + str(self.config.lr_baseline) + "\n")
        f_read.write("b_iter:\n" + str(self.config.b_iter) + "\n")
        f_read.write("lr_discriminator:\n" + str(self.config.lr_discriminator) + "\n")
        f_read.write("buffer_size:\n" + str(self.config.buffer_size) + "\n")
        f_read.write("sample_size:\n" + str(self.config.sample_size) + "\n")
        f_read.write("batch_size:\n" + str(self.config.batch_size) + "\n")
        f_read.write("inner_loop:\n" + str(self.config.inner_loop) + "\n")
        f_read.write("seed:\n" + str(self.config.seed) + "\n")
        f_read.write("schedule:\n" + str(self.config.schedule) + "\n")
        f_read.write("beta_for_entropy:\n" + str(self.config.beta) + "\n")
        f_read.write("w:\n" + str(self.config.w) + "\n")

        f_read.close()

        np.random.seed(self.seed)
        config = self.config
        start_time = time.time()
        numeptotal = 0
        start_epoch = 0
        progress = 0
        Omax = self.config.sample_size  ## Sample size

        state_d = state_expert
        actions_d = action_expert

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

        #
        # Learning part
        #
        for i in range(start_epoch, start_epoch + config.n_iter):  # Learning start
            progress += 1

            #
            # Load encode data
            #
            demo_dir = "Expert/"
            encodes_d = np.load(demo_dir + "encode_5.npy")

            encode_labels = [(0, 1) for num_expert in range(30)]
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
                config.max_step_limit,
                Omax_sample,
                count_goalagent,
                i,
            )
            latent_labels = [r for r in range(self.encode_dim)]
            sub_labels = []
            for r in range(int(Omax_sample / self.encode_dim)):
                sub_labels.extend(latent_labels)

            count_goalagent_stack.append(count_goalagent)

            #
            # Pool trajecoty
            #
            index_for_labels = 0

            for path in rollouts:
                path["labels"] = [sub_labels[index_for_labels]]
                index_for_labels += 1

            for path in rollouts:
                self.buffer.add(path)

            paths = self.buffer.get_sample(config.sample_size)

            #
            # Sorting trajecotries
            #
            logstds_n = np.concatenate([path["logstds"] for path in paths])
            state_n = np.concatenate([path["state"] for path in paths])
            encodes_n = np.concatenate([path["encodes"] for path in paths])
            actions_n = np.concatenate([path["actions"] for path in paths])
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

            loss_sum = 0

            idx = np.arange(numnototal)
            np.random.shuffle(idx)
            train_val_ratio = 0.7

            #
            # Save the trajectory for visualize
            #
            path_idx = 0
            start_w = 0
            for path in paths:

                noise = np.zeros((path["length"], self.action_dim), dtype=np.float16)
                file_path_world = new_dir_path + "/Generator_trajectories"

                if not os.path.exists(file_path_world):
                    os.mkdir(file_path_world)
                file_path_world = new_dir_path + "/Generator_trajectories/iter_%d_world_%d" % (i, path_idx)

                np.save(file_path_world, path["state_2dim"])
                np.save(file_path_world + "_path_length.npy", path["length"])
                np.save(file_path_world + "_actions.npy", actions_n[start_w:start_w + path["length"]])
                np.save(file_path_world + "_policies.npy", policies_n[start_w:start_w + path["length"]])
                np.save(file_path_world + "_labels.npy", labels_n[path_idx])

                start_w = start_w + path["length"]
                if start_w + path["length"] > numnototal:
                    start_w = (start_w + path["length"]) % numnototal

                path_idx += 1

            #
            # Inner loop Discriminator & Generator
            #
            if i == start_epoch:
                new_labels = []
                new_labels = encode_labels
                if not os.path.exists(new_dir_path + "/new_labels"):
                    os.mkdir(new_dir_path + "/new_labels")
                np.save(new_dir_path + "/new_labels/iter_%d_labels.npy" % (i), np.array(new_labels))

            for iiter in range(self.config.inner_loop):
                if iiter % 10 == 0:
                    print("Inner: ", iiter)
                if iiter % 50 == 0:
                    print(new_dir_path)

                count_goalagent_new = 0

                #
                # Create trajectory
                #
                new_labels = encode_labels

                rollouts_new, count_goalagent_new = rollout_contin(
                    self,
                    self.state_dim,
                    self.encode_dim,
                    self.action_dim,
                    config.max_step_limit,
                    Omax,
                    count_goalagent_new,
                    i,
                    encode_list=new_labels
                )
                print("GOAL: ", count_goalagent_new)

                if iiter == self.config.inner_loop - 1:
                    count_goalagent_new_stack.append(count_goalagent_new)

                index_for_labels = 0
                for path in rollouts_new:
                    path["labels"] = [new_labels[index_for_labels]]
                    index_for_labels += 1

                #
                # Pool trajectory (for buffer_new)
                #
                for path in rollouts_new:
                    self.buffer_new.add(path)

                paths_new = self.buffer_new.get_sample(config.sample_size)

                #
                # For TRPO (Single path)
                #
                # print ("Calculating actions ...")
                # TODO: Deal with tf session (What does run do?)
                for path in paths_new:
                    path["mus"] = self.sess.run(
                        self.action_dist_mu,
                        {self.state_dim: path["state"],
                         self.encode_dim: path["encodes"]}
                    )

                #
                # Sorting trajecotries
                #
                mus_new_n = np.concatenate([path["mus"] for path in paths_new])
                logstds_new_n = np.concatenate([path["logstds"] for path in paths_new])
                state_new_n = np.concatenate([path["state"] for path in paths_new])
                encodes_new_n = np.concatenate([path["encodes"] for path in paths_new])
                actions_new_n = np.concatenate([path["actions"] for path in paths_new])
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
                    policy_d.append(self.get_policy_encode([state_d[t]], np.argmax(actions_d[t]), [encodes_d[t]]))

                policy_d = np.array(policy_d)

                batch_size_agent = len(state_new_n)
                batch_size_expert = len(state_d)

                # noise_expert = np.random.normal(0, 0.2, (batch_size_expert, self.action_dim))
                # noise_agent = np.random.normal(0, 0.2, (batch_size_agent, self.action_dim))
                noise_expert = np.zeros((batch_size_expert, self.action_dim), dtype=np.float16)
                noise_agent = np.zeros((batch_size_agent, self.action_dim), dtype=np.float16)

                #
                # Update Discriminator (using Keras)
                #
                # TODO: pyTorch instead of tf/keras
                d_loss_real = self.discriminator.train_on_batch(
                    [state_d[start_d:start_d + batch_size_expert], actions_d[start_d:start_d + batch_size_expert],
                     noise_expert[:batch_size_expert], encodes_d[start_d:start_d + batch_size_expert],
                     policy_d[start_d:start_d + batch_size_expert]],
                    np.ones(batch_size_expert))
                d_loss_fake = self.discriminator.train_on_batch(
                    [state_new_n[start_n:start_n + batch_size_agent], actions_new_n[start_n:start_n + batch_size_agent],
                     noise_agent[:batch_size_agent], encodes_new_n[start_n:start_n + batch_size_agent],
                     policies_new_n[start_n:start_n + batch_size_agent]],
                    np.zeros(batch_size_agent))
                loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # print self.discriminator.summary()
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    l.set_weights(weights)

                if iiter == self.config.inner_loop - 1:
                    losses.append(loss)

                #
                # Update Generator 
                #
                for path in paths_new:

                    path["baselines"] = self.baseline.predict(path)
                    output_d = self.discriminator.predict(
                        [path["state"], path["actions"], noise_agent, path["encodes"], path["policies"]])

                    anti = 1.0 - copy.copy(output_d)

                    for an in range(path["length"]):
                        if anti[an] == 0.0:
                            anti[an] = 1e-10

                    #
                    # Calculate beta (with scheduling)
                    #
                    if self.config.schedule == 1:
                        beta = self.config.beta - self.config.w * i
                    #
                    # Calculate beta (without scheduling)
                    #
                    else:
                        beta = self.config.beta

                    output_d_for_reward = (np.log(output_d + 1e-10) - np.log(anti)).flatten() \
                                          + beta * np.log(np.max(path["policies"], axis=1) + 1e-10).flatten()

                    path["rewards"] = output_d_for_reward.flatten()

                    path_baselines = np.append(path["baselines"], 0 if
                    path["flag"] == 1 else
                    path["baselines"][-1])

                    deltas = path["rewards"] + config.gamma * path_baselines[1:] - \
                             path_baselines[:-1]

                    # path["returns"] = discount(path["rewards"], config.gamma)
                    # path["advants"] = path["returns"] - path["baselines"]
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

                    #
                    # Calcurate Generator's reward(average each epochs)
                    #
                    generator_loss.append(episoderewards.mean())

                #
                # Computing baseline function for next iter.
                #
                b_loss = self.baseline.fit(paths_new, batch_size_agent)
                if iiter == self.config.inner_loop - 1:
                    baseline_loss.append(b_loss)

                #
                # Update Generator's parametor using TRPO
                # TODO: pyTorch implementation
                #
                feed = {self.state_dim: state_new_n,
                        self.encode_dim: encodes_new_n,
                        self.action_dim: actions_new_n,
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
            for k, v in stats.items():
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
            np.save(file_results + "reward_Generator.npy", generator_loss_A)
            np.save(file_results + "loss_Discriminator.npy", losses_A)

            #
            # Learning rate of Baseline
            #
            baseline_loss_A = np.array(baseline_loss)
            np.save(file_results + "baseline_loss.npy", baseline_loss_A)

            #
            # Number of goal agent
            #
            goal_agent_A = np.array(count_goalagent_stack)
            np.save(file_results + "number_of_goal_agent.npy", goal_agent_A)

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
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax.plot(t, losses)
        ax.set_title("Discriminator_loss")
        ax2.plot(t, generator_loss)
        ax2.set_title("Generator_reward")
        plt.xlabel("Iteration Nr.")
        plt.ylabel("Loss Value")
        plt.savefig(new_dir_path + '/loss.png')
        np.save(file_results + "reward_Generator.npy", generator_loss)
        np.save(file_results + "loss_Discriminator.npy", losses)

        #
        # Learning rate of Baseline
        #
        baseline_loss = np.array(baseline_loss)
        t = np.arange(progress)
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t, baseline_loss)
        ax.set_title("baseline_loss")
        plt.xlabel("Iteration Nr.")
        plt.ylabel("Loss Value")
        plt.savefig(new_dir_path + '/baseline_loss.png')
        np.save(file_results + "baseline_loss.npy", baseline_loss)

        #
        # Number of goal agent
        #
        goal_agent = np.array(count_goalagent_stack)
        t = np.arange(progress)
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t, goal_agent)
        ax.set_title("Goal_agent")
        plt.xlabel("Iteration Nr.")
        plt.ylabel("Number of Reached Goals")
        plt.savefig(new_dir_path + '/goal_agent.png')

        return progress
