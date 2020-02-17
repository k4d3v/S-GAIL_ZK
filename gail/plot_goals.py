import argparse
import gym
import pybulletgym
import os
import sys
import pickle
import time
import copy
import numpy as np 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import count
from utils import *

from utils.plot_rewards import plot_reached
from estimate_target import targets


def run_and_plot(args):
    lower_dim = args.lower_dim < env.observation_space.shape[0]

    # Iterate over all models
    reached_rel = []
    iters_list = range(10, 1001, 10)
    for iters in iters_list:
        print("Current model trained for "+str(iters)+" iterations")

        """Load a model"""
        model_path = os.path.join(assets_dir(), 'learned_models/{}_gail_{}_{}.p'.format(args.env_name, "comp" if lower_dim else "full", str(iters)))
        policy_net, _, _, running_state = pickle.load(open(model_path, "rb"))
        is_encode = env.observation_space.shape[0] < policy_net.affine_layers[0].in_features

        """For this model, run and count goals"""
        num_steps = 0
        goals, reached = 0,0

        # Perform 20 episodes
        i_episode = 0
        while i_episode<50:
            state = env.reset()
            state = running_state(state)

            # Determine target and current demo class
            target_pose = env.env.robot.target.pose().xyz()[:2]
            t00, t01, t10, t11 = targets(target_pose)
            if not t00: 
            #if not (t00 or t01 or t10 or t11): 
                #print("Goal not accepted")
                continue  # Skip following lines if cordinates alternate
            
            goals+=1
            
            # SGAIL agent
            if is_encode:
                encode = [0,0,0,1] if t00 else [0,0,1,0] if t01 else [0,1,0,0] if t10 else [1,0,0,0]

            if args.lower_dim == 6 and args.env_name == "ReacherPyBulletEnv-v0":
                state = np.delete(copy.copy(state), [4, 5, 8]) 
            elif args.lower_dim == 6 and args.env_name == "Reacher-v2":
                state = np.delete(copy.copy(state), [4, 5, 8, 9, 10])
            reward_episode = 0
            
            for t in range(10000):
                if is_encode:
                    state = np.hstack((state, encode))
                state_var = tensor(state).unsqueeze(0).to(dtype)
                # choose mean action
                action = policy_net(state_var)[0][0].detach().numpy()
                # choose stochastic action
                # action = policy_net.select_action(state_var)[0].cpu().numpy()
                action = action.astype(np.float64)
                next_state, reward, done, _ = env.step(action)
                next_state = running_state(next_state)

                if args.lower_dim == 6 and args.env_name == "ReacherPyBulletEnv-v0":
                    next_state = np.delete(copy.copy(next_state), [4, 5, 8])
                elif args.lower_dim == 6 and args.env_name == "Reacher-v2":
                    next_state = np.delete(copy.copy(next_state), [4, 5, 8, 9, 10])
                
                reward_episode += reward
                num_steps += 1

                if args.render:
                    time.sleep(0.1 / 60.)  # For human-friendly visualization
                    env.render(mode="human")
                if done:
                    break

                state = next_state

            print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

            finger_pose = env.env.robot.fingertip.pose().xyz()[:2]
            print(np.linalg.norm(target_pose - finger_pose))
            # Look if goal was reached
            if np.linalg.norm(target_pose - finger_pose) < 0.05:
                reached+=1
            
            i_episode+=1
            
        print("Num goals: ", goals)
        print("Num reached: ", reached)
        reached_rel.append(reached/goals)

    # Plot results
    plot_reached(iters_list, reached_rel, args.env_name, "comp" if lower_dim else "full")

parser = argparse.ArgumentParser(description='Rollout learner')
parser.add_argument('--env-name', default="ReacherPyBulletEnv-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--lower_dim', type=int, default=6, metavar='N',
                    help='Lower dimension. Is smaller than dim of state, if on (default: 10000)')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
env = gym.make(args.env_name)
env.seed(args.seed)
if args.render: env.render(mode="human")
torch.manual_seed(args.seed)

run_and_plot(args)
