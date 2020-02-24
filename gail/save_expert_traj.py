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
from estimate_target import targets


parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="ReacherPyBulletEnv-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', default="/home/developer/S-GAIL_ZK/assets/learned_models/ReacherPyBulletEnv-v0_trpo_full.p", metavar='G',
                    help='name of the expert model') # TODO: Relative path
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=25000, metavar='N',
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

trajs00, trajs01, trajs10, trajs11 = [],[],[],[]
trajs00_xy, trajs01_xy, trajs10_xy, trajs11_xy = [],[],[],[]

def main_loop():

    num_steps = 0

    for i_episode in count():
        atraj00, atraj01, atraj10, atraj11 = [],[],[],[]
        atraj00_xy, atraj01_xy, atraj10_xy, atraj11_xy = [],[],[],[]
        state = env.reset()

        # Determine target and current demo class
        target_pose = env.env.robot.target.pose().xyz()[:2]
        t00, t01, t10, t11 = targets(target_pose, dist=0.005)
        if not (t00 or t01 or t10 or t11): 
            #print("Goal not accepted")
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

            tip = env.env.robot.fingertip.pose().xyz()[:2]
            sa = np.hstack([state, action])

            if t00:  # Blue left
                expert_traj0001.append(sa)
                atraj00.append(sa)
                atraj00_xy.append(tip)
            elif t01:  # White up
                expert_traj0010.append(sa)
                atraj01.append(sa)
                atraj01_xy.append(tip)
            elif t10:  # White down
                expert_traj0100.append(sa)
                atraj10.append(sa)
                atraj10_xy.append(tip)
            elif t11:  # Blue right
                expert_traj1000.append(sa)
                atraj11.append(sa)
                atraj11_xy.append(tip)

            if args.render:
                time.sleep(0.01 / 60.)  # For human-friendly visualization
                env.render(mode="human")
            if done or num_steps >= args.max_expert_state_num:
                if len(atraj00)>0: 
                    trajs00.append(np.array(atraj00)) 
                    trajs00_xy.append(np.array(atraj00_xy)) 
                if len(atraj01)>0: 
                    trajs01.append(np.array(atraj01)) 
                    trajs01_xy.append(np.array(atraj01_xy)) 
                if len(atraj10)>0: 
                    trajs10.append(np.array(atraj10)) 
                    trajs10_xy.append(np.array(atraj10_xy)) 
                if len(atraj11)>0: 
                    trajs11.append(np.array(atraj11))
                    trajs11_xy.append(np.array(atraj11_xy))
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
#pickle.dump((expert_traj, running_state), open(os.path.join(assets_dir(), 'expert_traj/{}_expert_traj.p'.format(args.env_name)), 'wb'))
#pickle.dump((encode), open(os.path.join(assets_dir(), 'expert_traj/{}_encode.p'.format(args.env_name)), 'wb'))

print(len(trajs00), len(trajs01), len(trajs10), len(trajs11))
trajs = trajs00+trajs01+trajs10+trajs11
trajs_xy = trajs00_xy+trajs01_xy+trajs10_xy+trajs11_xy
pickle.dump(trajs, open(os.path.join(assets_dir(), 'expert_traj/test_trajs.p'.format(args.env_name)), 'wb'))
pickle.dump(trajs_xy, open(os.path.join(assets_dir(), 'expert_traj/test_trajs_xy.p'.format(args.env_name)), 'wb'))