#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import operator
import random

def initialize(path, seed):
    random.seed(seed)
    np.random.seed(seed)
    print("grid seed: ", seed)
    file_path = path + "/Initialize_seed.txt"
    f = open(file_path, "a")
    f.write("Grid world Seed:\n" + str(seed) + "\n")
    f.close()

class MDP:

    def __init__(self, init, actlist, terminals, gamma=.9):
        self.init = init
        self.actlist = actlist
        self.terminals = terminals
        if not (0 <= gamma < 1):
            raise ValueError("An MDP must have 0 <= gamma < 1")
        self.gamma = gamma
        self.states = set()
        self.reward = {}

    def T(self, state, action):
        raise NotImplementedError

    def actions(self, state):
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

class GridMDP(MDP):

    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
        grid.reverse()  # because we want row 0 on bottom, not on top                                                                                                  
        MDP.__init__(self, init, actlist=[],
                     terminals=terminals, gamma=gamma)
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x] is not None:
                    self.states.add((x, y))

    #
    # State Transition Rule
    #
    def T(self, state, action):
        if action is None:
            return state
        else:
            return self.go(state, action)

    def go(self, state, direction):
    
        direction = np.array(direction)
        action_dir = np.argmax(direction)

        if action_dir == 0:
            #print ">"
            direct = np.array([[1, 0]], dtype=np.float32)
        elif action_dir == 1:
            #print "^"
            direct = np.array([[0, 1]], dtype=np.float32)
        elif action_dir == 2:
            #print "<"
            direct = np.array([[-1,0]], dtype=np.float32)
        elif action_dir == 3:
            #print "v"
            direct = np.array([[0,-1]], dtype=np.float32)

        state_new = tuple(map(operator.add, state[0], direct[0]))

        if state_new in self.states :
            if state_new is not None:
                s = np.array([state_new])
                return s
        else:
            return state

if __name__ == '__main__':
    print("OK")
