import pickle

from utils import *


def get_exp(env, args, demo_dir):
    # State and action dim + filter
    # TODO: Why different?
    if args.env_name == "Reacher-v2":
        # Labels
        encodes_d = np.load(demo_dir + "encode_mujoco.npy")  # Class two has index 6392

        # States
        state_expert = np.load(demo_dir + "state_mujoco.npy")[:6392]  # State is 6D, but normally 11D
        action_expert = np.load(demo_dir + "action_mujoco.npy")[:6392]  # Actions are 2 dim

        # Normalize & Get Min-Max
        state_expert_norm = min_max(state_expert, axis=0)
        action_expert_norm = min_max(action_expert, axis=0)

        state_dim = state_expert_norm.shape[1]
        is_disc_action = len(env.action_space.shape) == 0
        action_dim = env.action_space.shape[0]
        running_state = ZFilter((state_dim,), clip=5)

        # Combine state and actions into one array
        expert_traj = np.concatenate((state_expert_norm, action_expert_norm), axis=1)

        state_max = np.max(state_expert, axis=0)
        state_min = np.min(state_expert, axis=0)
        action_max = np.max(action_expert, axis=0)  # = [0.2112, 0.3219]
        action_min = np.min(action_expert, axis=0)  # = [-0.1343, -0.0819]

    # load trajectory from expert
    # TODO: Is running state needed?
    else:
        state_dim = env.observation_space.shape[0]
        is_disc_action = len(env.action_space.shape) == 0
        action_dim = 1 if is_disc_action else env.action_space.shape[0]
        expert_traj, running_state = pickle.load(open(args.expert_traj_path+"expert_traj.p", "rb"))
        expert_traj = expert_traj[:450] # State: 9d, action: 2d
        # First 2100 (s,a) pairs are class 1; 1750,3450,2700
        # 11250,11400,14600,12750
        # running_reward = ZFilter((1,), demean=False, clip=10)
        encodes_d = pickle.load(open(args.expert_traj_path+"encode.p", "rb"))

        state_max, state_min, action_max, action_min = None, None, None, None

    return (state_dim, action_dim, is_disc_action,
            expert_traj, running_state, encodes_d,
            state_max, state_min, action_max, action_min)