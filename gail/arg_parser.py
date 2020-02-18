import argparse


def prep_parser():
    """
    Add parser args
    """
    parser = argparse.ArgumentParser(description='PyTorch GAIL example')
    parser.add_argument('--env-name', default="ReacherPyBulletEnv-v0", metavar='G',
                        help='name of the environment to run, default: Hopper-v2')
    parser.add_argument('--expert-traj-path', default="/home/developer/S-GAIL_ZK/assets/expert_traj/ReacherPyBulletEnv-v0_", metavar='G',
                        help='path of the expert trajectories (Reacher: /home/developer/S-GAIL_ZK/assets/expert_traj/ReacherPyBulletEnv-v0_)')
    parser.add_argument('--lower_dim', type=int, default=6, metavar='N',
                    help='Lower dimension. If value is smaller than dim of state, sgail will use states with dim=value (default: 10000)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='render the environment')
    parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                        help='log std for the policy (default: -0.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                        help='gae (default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                        help='gae (default: 3e-4)')
    parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                        help='clipping epsilon for PPO')
    parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                        help='number of threads for agent (default: 4)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--min-batch-size', type=int, default=6144, metavar='N',
                        help='minimal batch size per PPO update (default: 2048)')
    parser.add_argument('--max-iter-num', type=int, default=2000, metavar='N',
                        help='maximal number of main iterations (default: 5000)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 2)')
    parser.add_argument('--save-model-interval', type=int, default=50, metavar='N',
                        help="interval between saving model (default: 0, means don't save)")
    parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
    # Args for SGAIL: Beta and beta change, encode_dim
    parser.add_argument('--encode_dim', type=int, default=4, metavar='N',
                    help='Number of demo classes.')
    parser.add_argument("--beta", type=float, default=.9,
                        help="Entropy correction (default in S-GAIL paper: 0.9)")
    parser.add_argument("--w", type=float, default=1e-4,
                        help="Scheduling param for beta (default: 1e-3)")
    return parser.parse_args()
