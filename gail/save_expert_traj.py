import argparse
import gym
import pybulletgym
import os
import sys
import pickle
import time
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import count
from utils import *


parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="ReacherPyBulletEnv-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', default="/home/developer/S-GAIL_ZK/assets/learned_models/ReacherPyBulletEnv-v0_trpo_full.p", metavar='G',
                    help='name of the expert model') # TODO: Relative path
parser.add_argument('--lower_dim', type=int, default=10000, metavar='N',
                    help='Lower dimension. Is smaller than dim of state, if on (default: 10000)')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=10000, metavar='N',
                    help='maximal number of main iterations (default: 100000)')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
env = gym.make(args.env_name)
env.seed(args.seed)
if args.render: env.render(mode="human")
torch.manual_seed(args.seed)
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]

# Load policy
try:
    policy_net, _, running_state = pickle.load(open(args.model_path, "rb"))
except ValueError:  # Maybe more stuff was pickled (e.g. Discriminator for gail)
    policy_net, _, _, running_state = pickle.load(open(args.model_path, "rb"))
running_state.fix = True

expert_traj = []
expert_traj0001, expert_traj0010, expert_traj0100, expert_traj1000 = [], [], [], []


def main_loop():

    num_steps = 0

    for i_episode in count():
        state = env.reset()

        # Determine target and current demo class
        target_pose = env.env.robot.target.pose().xyz()[:2]
        pos = 0.15
        dist = 0.005
        t00 = np.linalg.norm(target_pose-[-pos,-pos])<dist
        t01 = np.linalg.norm(target_pose-[-pos,pos])<dist
        t10 = np.linalg.norm(target_pose-[pos,-pos])<dist
        t11 = np.linalg.norm(target_pose-[pos,pos])<dist
        if not (t00 or t01 or t10 or t11): 
            #print("Goal not accepted")
            continue  # Skip following lines if cordinates alternate
        
        if args.lower_dim == 6 and args.env_name == "ReacherPyBulletEnv-v0":
            state = np.delete(copy.copy(state), [4, 5, 8])
        elif args.lower_dim == 6 and args.env_name == "Reacher-v2":
            state = np.delete(copy.copy(state), [4, 5, 8, 9, 10])
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
            if args.lower_dim == 6 and args.env_name == "ReacherPyBulletEnv-v0":
                next_state = np.delete(copy.copy(next_state), [4, 5, 8])
            elif args.lower_dim == 6 and args.env_name == "Reacher-v2":
                next_state = np.delete(copy.copy(next_state), [4, 5, 8, 9, 10])
            next_state = running_state(next_state)
            reward_episode += reward
            num_steps += 1

            if t00:  # Blue left
                expert_traj0001.append(np.hstack([state, action]))
            elif t01:  # White up
                expert_traj0010.append(np.hstack([state, action]))
            elif t10:  # White down
                expert_traj0100.append(np.hstack([state, action]))
            elif t11:  # Blue right
                expert_traj1000.append(np.hstack([state, action]))

            if args.render:
                time.sleep(1 / 60.)  # For human-friendly visualization
                env.render(mode="human")
            if done or num_steps >= args.max_expert_state_num:
                break
            
            state = next_state

        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

        if num_steps >= args.max_expert_state_num:
            break


main_loop()

print(len(expert_traj0001), len(expert_traj0010), len(expert_traj0100), len(expert_traj1000))
expert_traj0001 = np.stack(expert_traj0001)
expert_traj0010 = np.stack(expert_traj0010)
expert_traj0100 = np.stack(expert_traj0100)
expert_traj1000 = np.stack(expert_traj1000)
expert_traj = np.concatenate(
    (expert_traj0001, expert_traj0010, expert_traj0100, expert_traj1000), axis=0)
# Encode class
encode = np.concatenate((np.array([(0,0,0,1)]*expert_traj0001.shape[0]),
                        np.array([(0,0,1,0)]*expert_traj0010.shape[0]),
                        np.array([(0,1,0,0)]*expert_traj0100.shape[0]),
                        np.array([(1,0,0,0)]*expert_traj1000.shape[0])), axis=0)
# Save trajs and encodes
pickle.dump((expert_traj, running_state), open(os.path.join(assets_dir(), 'expert_traj/{}_expert_traj.p'.format(args.env_name)), 'wb'))
pickle.dump((encode), open(os.path.join(assets_dir(), 'expert_traj/{}_encode.p'.format(args.env_name)), 'wb'))
