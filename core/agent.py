import multiprocessing

import copy

from utils.replay_memory import Memory
from utils.math import delete, norm_act
from utils.torch import *
import math
import time

from gail.estimate_target import *


def collect_samples(pid, queue, env, policy, custom_reward,
                    mean_action, render, running_state, min_batch_size, 
                    beta, lower_dim, encode_list, targets_est=False):
    """
    Rollout for each thread
    @return: memory, log
    """
    env_name = env.spec.id
    torch.randn(pid)
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0

    is_encode = not (encode_list is None)
    encode_dim = np.max(encode_list)+1 if is_encode else 1
    goals, reached = 0, 0
    
    while num_steps < min_batch_size*encode_dim:
        state = env.reset()
        state = running_state(state)

        if targets_est:
            # Determine target and current demo class
            target_pose = env.env.robot.target.pose().xyz()[:2]
            t00, t01, t10, t11 = targets(target_pose)

            if is_encode:
                encode = np.zeros(encode_dim, dtype=np.float32)
                encode[encode_list[num_episodes]] = 1

                if not ((t00 and np.array_equal(encode,[0,0,0,1]))  # Blue left
                    or (t01 and np.array_equal(encode,[0,0,1,0]))  # White up
                    or (t10 and np.array_equal(encode,[0,1,0,0]))  # White down
                    or (t11 and np.array_equal(encode,[1,0,0,0]))):  # Blue right
                    continue  # Skip following lines if cordinates alternate

            elif not t00: 
                continue  # Skip following lines if cordinates alternate
            
        goals+=1

        if lower_dim:
            if env_name == "ReacherPyBulletEnv-v0":
                state = np.delete(copy.copy(state), [4, 5, 8])
            elif env_name == "Reacher-v2":
                state = np.delete(copy.copy(state), [4, 5, 8, 9, 10])
        
        reward_episode = 0

        for t in range(10000):
            astate = np.hstack((state, encode)) if is_encode else state
            state_var = tensor(astate).unsqueeze(0)
            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                else:
                    action = policy.select_action(state_var)[0].numpy()
            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            #action = norm_act(aaction, a_min, a_max) if a_min is not None else aaction
            next_state, reward, done, _ = env.step(action)
            reward_episode += reward

            next_state = running_state(next_state)

            if lower_dim:
                if env_name == "ReacherPyBulletEnv-v0":
                    next_state = np.delete(copy.copy(next_state), [4, 5, 8])
                elif env_name == "Reacher-v2":
                    next_state = np.delete(copy.copy(next_state), [4, 5, 8, 9, 10])
            
            if custom_reward is not None:
                if is_encode: 
                    se = torch.from_numpy(np.hstack((state, encode)).reshape(1,-1))
                    pol = policy.get_policy(se, action)
                    reward = custom_reward(state, action, encode, pol, beta) 
                else:
                    reward = custom_reward(state, action, beta=beta)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1

            if is_encode:
                memory.push(state, action, mask, next_state, reward, encode, pol)
            else:
                memory.push(state, action, mask, next_state, reward, None, None)

            if render:
                env.render()
            if done:
                break

            state = next_state

        if targets_est:
            # Look if goal was reached
            reached+=goal_reached(target_pose, env.env.robot.fingertip.pose().xyz()[:2])
            
        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward
        log['goals'] = goals
        log['reached_goals'] = reached

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env, policy, device, custom_reward=None, targets=False,
                 mean_action=False, render=False, running_state=None, lower_dim=False, num_threads=1):
        """
        @param num_threads: More than one means parallel execution when collecting samples
        """
        self.targets = targets
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.lower_dim = lower_dim
        self.num_threads = num_threads

    def collect_samples(self, min_batch_size, beta=None, encode_list=None):
        """
        Parallelized version of rollout. Each worker calls outer fun
        """
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env, self.policy, self.custom_reward, self.mean_action,
                           False, self.running_state, thread_batch_size)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env, self.policy, self.custom_reward, self.mean_action,
                                      self.render, self.running_state, thread_batch_size, beta, self.lower_dim, encode_list, targets_est=self.targets)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log
