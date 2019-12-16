import argparse
import gym
import pybulletgym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import count
from utils import *


parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="ReacherPyBulletEnv-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', default="/home/developer/S-GAIL_ZK/assets/learned_models/ReacherPyBulletEnv-v0_trpo.p", metavar='G',
                    help='name of the expert model') # TODO: Relative path
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=10000, metavar='N',
                    help='maximal number of main iterations (default: 50000)')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
env = gym.make(args.env_name)
env.seed(args.seed)
if args.render: env.render(mode="human")
torch.manual_seed(args.seed)
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]

policy_net, _, running_state = pickle.load(open(args.model_path, "rb"))
expert_traj = []
expert_traj01, expert_traj10 = [], []


def main_loop():

    num_steps = 0

    for i_episode in count():
        state = env.reset()

        # Determine target and current demo class
        target_pose = env.env.robot.target.pose().xyz()
        t01 = target_pose[0]<0 and target_pose[1]<0
        t10 = target_pose[0]>0 and target_pose[1]>0
        if not t01 and not t10: 
            continue  # Skip following lines if cordinates alternate
        
        state = running_state(state)
        reward_episode = 0

        for t in range(10000):
            state_var = tensor(state).unsqueeze(0).to(dtype)
            # choose mean action
            action = policy_net(state_var)[0][0].detach().numpy()
            # choose stochastic action
            # action = policy_net.select_action(state_var)[0].cpu().numpy()
            action = int(action) if is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)
            reward_episode += reward
            num_steps += 1

            if t01:
                expert_traj01.append(np.hstack([state, action]))
            else:
                expert_traj10.append(np.hstack([state, action]))

            if args.render:
                time.sleep(0.1 / 60.)  # For human-friendly visualization
                env.render(mode="human")
            if done or num_steps >= args.max_expert_state_num:
                break

            state = next_state

        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

        if num_steps >= args.max_expert_state_num:
            break


main_loop()

expert_traj01 = np.stack(expert_traj01)
expert_traj10 = np.stack(expert_traj10)
expert_traj = np.concatenate((expert_traj01, expert_traj10), axis=0)
# Encode class
encode = np.concatenate((np.array([(0,1)]*expert_traj01.shape[0]),
                        np.array([(1,0)]*expert_traj10.shape[0])))
# Save trajs and encodes
pickle.dump((expert_traj, running_state), open(os.path.join(assets_dir(), 'expert_traj/{}_expert_traj.p'.format(args.env_name)), 'wb'))
pickle.dump((encode, running_state), open(os.path.join(assets_dir(), 'expert_traj/{}_encode.p'.format(args.env_name)), 'wb'))
