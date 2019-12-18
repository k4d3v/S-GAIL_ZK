import pickle
import numpy as np

from utils import *


def get_exp(env, args):
    # load trajectory from expert
    # TODO: Is running state needed?
    is_disc_action = len(env.action_space.shape) == 0
    action_dim = 1 if is_disc_action else env.action_space.shape[0]
    try:
        expert_traj, running_state = pickle.load(open(args.expert_traj_path+"expert_traj.p", "rb"))
        # running_reward = ZFilter((1,), demean=False, clip=10)
        encodes_d = pickle.load(open(args.expert_traj_path+"encode.p", "rb"))
        
        if args.lower_dim <env.observation_space.shape[0]:
            state_dim = args.lower_dim
            running_state = ZFilter((state_dim,), clip=5)
            if args.env_name == "ReacherPyBulletEnv-v0":
                expert_traj = np.delete(expert_traj, [4,5,8], axis=1)
            elif args.env_name == "Reacher-v2":
                expert_traj = np.delete(expert_traj, [4,5,8,9,10], axis=1)
        else:
            state_dim = env.observation_space.shape[0]  

    # Expert trajs are not needed
    except AttributeError:
        expert_traj, running_state, encodes_d = None, None, None
        state_dim = args.lower_dim if args.lower_dim <env.observation_space.shape[0] else env.observation_space.shape[0]  

    state_max, state_min, action_max, action_min = None, None, env.action_space.high, env.action_space.low

    return (state_dim, action_dim, is_disc_action,
            expert_traj, running_state, encodes_d,
            state_max, state_min, action_max, action_min)