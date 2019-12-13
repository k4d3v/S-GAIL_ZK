#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random
import time
import tensorflow as tf
import scipy.signal
from keras.applications.resnet50 import preprocess_input

from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Activation, Flatten, Input, merge, Lambda, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import keras.backend as K

from collections import deque
import copy

from datetime import datetime
import gym
import pybulletgym

# env = gym.make('Reacher-v1')
#env = gym.make('Reacher-v2')
#env = gym.make("MinitaurBulletEnv-v0")
env = gym.make("ReacherPyBulletEnv-v0")
env.reset()

dtype = tf.float32


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x - min) / (max - min)
    return result


def norm(x, a_min, a_max):
    result = np.clip((x - a_min) / (a_max - a_min), 0.0, 1.0)
    return result


def initialize(path, seed):
    random.seed(seed)
    np.random.seed(seed)
    print("mujoco seed: ", seed)
    file_path = path + "/Initialize_seed.txt"
    f = open(file_path, "a")
    f.write("Grid world Seed:\n" + str(seed) + "\n")
    f.close()

    return seed


def seed_initialize(path, seed):
    print("utils seed", seed)

    initialize(path, seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    file_path = path + "/Initialize_seed.txt"
    f = open(file_path, "a")
    f.write("Seed:\n" + str(seed) + "\n")
    f.close()

    return seed


def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def gauss_prob_val(mu, logstd, x):
    std = np.exp(logstd)
    var = np.square(std)
    gp = np.exp(-np.square(x - mu) / (2 * var)) / ((2 * np.pi) ** .5 * std)
    return np.prod(gp, axis=1)


def gauss_prob(mu, logstd, x):
    std = tf.exp(logstd)
    var = tf.square(std)
    gp = tf.exp(-tf.square(x - mu) / (2 * var)) / ((2 * np.pi) ** .5 * std)
    return tf.reduce_prod(gp, [1])


def gauss_log_prob(mu, logstd, x):
    var = tf.exp(2 * logstd)
    gp = -tf.square(x - mu) / (2 * var) - .5 * tf.log(tf.constant(2 * np.pi)) - logstd
    return tf.reduce_sum(gp, [1])


def gauss_selfKL_firstfixed(mu, logstd):
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd
    return gauss_KL(mu1, logstd1, mu2, logstd2)


def gauss_KL(mu1, logstd1, mu2, logstd2):
    var1 = tf.exp(2 * logstd1)
    var2 = tf.exp(2 * logstd2)
    kl = tf.reduce_sum(logstd2 - logstd1 + (var1 + tf.square(mu1 - mu2)) / (2 * var2) - 0.5)
    return kl


def gauss_ent(mu, logstd):
    h = tf.reduce_sum(logstd + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32))
    return h


def gauss_sample(mu, logstd):
    return mu + tf.exp(logstd) * tf.random_normal(tf.shape(logstd))


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(0, [tf.reshape(grad, [numel(v)])
                         for (v, grad) in zip(var_list, grads)])


def get_feat(imgs, feat_extractor):
    x = preprocess_input(imgs.astype(np.float32))
    x = feat_extractor.predict(x)
    return x


def look(x):
    return x


def delete(x, s_min, s_max):
    y = np.delete(x, [4, 5, 8, 9, 10])
    result = np.clip((y - s_min) / (s_max - s_min), 0.0, 1.0)
    return result


def norm_act(x, a_min, a_max):
    result = np.clip((x - a_min) / (a_max - a_min), 0.0, 1.0)
    return result


def rollout_contin(agent, state_dim, encode_dim, actions_dim, s_max, s_min, a_max, a_min,
                   max_step_limit, min_step_limit, paths_per_collect, count_goalagent, epoch, encode_list=None, iter_num=None):
    """
    Performs a rollout based on the TRPO agent's policy and returns the trajs and goals
    :param agent: TRPO agent object (See model_sgail)
    :param state_dim:
    :param encode_dim:
    :param actions_dim:
    :param s_max:
    :param s_min:
    :param a_max:
    :param a_min:
    :param max_step_limit:
    :param min_step_limit:
    :param paths_per_collect:
    :param count_goalagent:
    :param epoch:
    :param encode_list:
    :param iter_num:
    :return: Sampled trajs and number of reached goals
    """
    paths = []
    timesteps_sofar = 0
    encode_axis = 0
    encode_axisold = 0
    conjugate_gradient = 0

    for p in range(paths_per_collect):
        states, encodes, actions, logstds = \
            [], [], [], []
        policies = []
        state_1times = []
        acts = []
        goal_flag = 0
        length = 0

        state = np.ones((1, state_dim), dtype=np.float32)

        encode = np.zeros((1, encode_dim), dtype=np.float32)
        if encode_list == None:
            encode[0, encode_axis] = 1
            encode_axisold = encode_axis
            encode_axis = (encode_axis + 1) % encode_dim
        else:
            encode[0, encode_list[p]] = 1

        state = np.ones((1, state_dim), dtype=np.float32)
        obs = env.reset()
        state = delete(copy.copy(obs), s_min, s_max)

        for i in range(max_step_limit):

            states.append(state)
            length += 1

            logstd = np.array([[-3.0, -3.0]], dtype=np.float32)
            state_1times.append([state])
            encodes.append(encode)
            logstds.append(logstd)
            act, action, policy = agent.act([state], encode, logstd)
            acts.append(norm_act(action, a_min, a_max))
            actions.append(action)
            policies.append(policy)

            if i + 1 == max_step_limit or goal_flag == 1:

                #
                # GOAL or timeup
                #
                if goal_flag == 1:
                    flag = 1
                else:
                    flag = 0

                path = dict2(state=np.concatenate(state_1times),
                             encodes=np.concatenate(encodes),
                             actions=np.concatenate(actions),
                             act=np.concatenate(acts),
                             policies=np.concatenate(policies),
                             length=length,
                             logstds=np.concatenate(logstds),
                             flag=flag
                             )
                paths.append(path)
                break

            #
            # Transition state
            #
            obs, r, done, _ = env.step(action[0])
            state = delete(copy.copy(obs), s_min, s_max)

            if np.argmax(encode) == 0:
                if i + 1 >= min_step_limit and np.sum(abs(obs[-3:])) <= 0.018 and abs(action[0][0]) < 5e-4 \
                        and abs(action[0][0]) < 5e-4:
                    goal_flag = 1
                    count_goalagent += 1

            elif np.argmax(encode) == 1:
                if i + 1 >= min_step_limit and np.sum(abs(obs[-3:])) >= 0.40 and abs(action[0][0]) < 5e-4 \
                        and abs(action[0][0]) < 5e-4:
                    goal_flag = 1
                    count_goalagent += 1

    return paths, count_goalagent


# TODO: Maybe delete, as unused
class LinearBaseline(object):
    coeffs = None

    def _features(self, path):
        o = path["states"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, np.ones((l, 1))], axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        n_col = featmat.shape[1]
        lamb = 2.0
        self.coeffs = np.linalg.lstsq(
            featmat.T.dot(featmat) + lamb * np.identity(n_col),
            featmat.T.dot(returns))[0]

    def predict(self, path):
        return np.zeros(len(path["rewards"])) if self.coeffs is None else \
            self._features(path).dot(self.coeffs)


def pathlength(path):
    return len(path["actions"])


def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


# TODO: Maybe delete, as unused
class TimeDependentBaseline(object):
    def __init__(self):
        self.baseline = None

    def fit(self, paths):
        rets = [path["returns"] for path in paths]
        maxlen = max(len(ret) for ret in rets)
        retsum = np.zeros(maxlen)
        retcount = np.zeros(maxlen)
        for ret in rets:
            retsum[:len(ret)] += ret
            retcount[:len(ret)] += 1
        retmean = retsum / retcount
        self.baseline = retmean
        pred = np.concatenate([self.predict(path) for path in paths])
        return {"EV": explained_variance(pred, np.concatenate(rets))}

    def predict(self, path):
        if self.baseline is None:
            return np.zeros(pathlength(path))
        else:
            lenpath = pathlength(path)
            lenbase = len(self.baseline)
            if lenpath > lenbase:
                return np.concatenate([self.baseline, self.baseline[-1] +
                                       np.zeros(lenpath - lenbase)])
            else:
                return self.baseline[:lenpath]


class NNBaseline(object):
    def __init__(self, sess, state, encodes, state_dim, encode_dim, lr_baseline,
                 b_iter, batch_size, dir_path):
        """
        A NN baseline based on expert demos for training generator
        :param sess: tf var
        :param state: tf structure with same dimension as state space
        :param encodes:
        :param state_dim:
        :param encode_dim:
        :param lr_baseline: Linear baseline
        :param b_iter:
        :param batch_size:
        :param dir_path: Where new models should be stored
        """
        print("Now we build baseline")
        self.model = self.create_net(state, encodes, state_dim, encode_dim, lr_baseline)
        self.sess = sess
        self.b_iter = b_iter
        self.batch_size = batch_size
        self.first_time = True
        self.mixfrac = 0.1
        file_path = dir_path + "/Readme.txt"
        f_read = open(file_path, "a")
        f_read.write("Baseline alpha:\n" + str(self.mixfrac) + "\n")
        f_read.close()

    def create_net(self, state, encodes, state_dim, encode_dim, lr_baseline):
        """
        Creates the baseline model
        :param state:
        :param encodes:
        :param state_dim:
        :param encode_dim:
        :param lr_baseline:
        :return:
        """
        # TODO: pyTorch impl; Remove hardcoded dims
        K.set_learning_phase(1)

        states = Input(tensor=state)
        x = Dense(128)(states)
        x = LeakyReLU()(x)
        encodes = Input(tensor=encodes)
        e = Dense(128)(encodes)
        e = LeakyReLU()(e)
        h = merge([x, e], mode='sum')
        h = Dense(32)(h)
        h = LeakyReLU()(h)
        p = Dense(1)(h)

        model = Model(input=[state, encodes], output=p)
        adam = Adam(lr=lr_baseline)
        model.compile(loss='mse', optimizer=adam)
        return model

    def fit(self, paths, batch_size):
        """
        Fit the baseline using expert trajs
        :param paths: Expert trajs
        :param batch_size:
        :return: Loss fun value
        """
        state = np.concatenate([path["state"] for path in paths])
        encodes = np.concatenate([path["encodes"] for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])

        if self.first_time:
            self.first_time = False
            b_iter = 100
        else:
            returns_old = np.concatenate([self.predict(path) for path in paths])
            returns = returns * self.mixfrac + returns_old * (1 - self.mixfrac)
            b_iter = self.b_iter

        num_data = state.shape[0]
        idx = np.arange(num_data)
        np.random.shuffle(idx)
        train_val_ratio = 0.7
        num_train = int(num_data * train_val_ratio)
        state_train = state[idx][:num_train]
        encodes_train = encodes[idx][:num_train]
        returns_train = returns[idx][:num_train]

        state_val = state[idx][num_train:]
        encodes_val = encodes[idx][num_train:]
        returns_val = returns[idx][num_train:]

        # TODO: pyTorch
        start = 0
        for i in range(b_iter):
            loss = self.model.train_on_batch(
                [state_train[start:start + batch_size],
                 encodes_train[start:start + batch_size]],
                returns_train[start:start + batch_size]
            )
            start += batch_size
            if start >= num_train:
                start = (start + batch_size) % num_train
            val_loss = np.average(np.square(self.model.predict(
                [state_val, encodes_val]).flatten() - returns_val))
            # print ("Baseline loss:", loss, "val:", val_loss)
            if i == b_iter - 1:
                b_loss = val_loss
        return b_loss

    def predict(self, path):
        """
        Predict actions based on states
        :param path: States traj
        :return: Policy
        """
        # TODO: pyTorch
        if self.first_time:
            return np.zeros(pathlength(path))
        else:
            ret = self.model.predict(
                [path["state"], path["encodes"]])
        return np.reshape(ret, (ret.shape[0],))


class GetFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat(0, [tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return self.op.eval(session=self.session)


class SetFromFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        assigns = []
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(v, tf.reshape(theta[start:start + size], shape))
            )
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_paths = 0
        self.buffer = deque()

    def get_sample(self, sample_size):
        if self.num_paths < sample_size:
            return random.sample(self.buffer, self.num_paths)
        else:
            return random.sample(self.buffer, sample_size)

    def size(self):
        return self.buffer_size

    def add(self, path):
        if self.num_paths < self.buffer_size:
            self.buffer.append(path)
            self.num_paths += 1
        else:
            self.buffer.popleft()
            self.buffer.append(path)

    def count(self):
        return self.num_paths

    def erase(self):
        self.buffer = deque()
        self.num_paths = 0
