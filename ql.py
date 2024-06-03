import random

import numpy as np
import csv
import argparse
from envs.donut import Donut
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product
max_ep_len = 12

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')


def counterfactual_Q_update(fair_env, Q, state, action, prev_reward, next_state, actual_memory, args, num_updates=2):
    all_possible = []
    for i in range(len(actual_memory)):
        tmp = []
        ed = min(max_ep_len, actual_memory[i] + num_updates)
        for j in range(actual_memory[i], ed):
            tmp.append(j)
        all_possible.append(tmp)
    possible_memories = list(product(*all_possible))


    for i in range(len(possible_memories)):
        curr = list(possible_memories[i])
        reward = 0
        if curr[action] == max_ep_len:
            continue
        next_memories = curr.copy()
        next_memories[action] += 1
        for j in range(len(curr)):
            reward += np.log(float(next_memories[j] + 1))
        if prev_reward == 0:
            reward = 0
        memory = fair_env.encode(curr)
        next_memory = fair_env.encode(next_memories)
        max_action = argmax_greedy(Q[next_state, next_memory])
        Q[state, memory, action] = Q[state, memory, action] + args.alpha * (
                    reward + args.gamma * Q[next_state, next_memory, max_action] - Q[state, memory, action])


def run_Q_learning(episodes: int, alpha: float, epsilon: float, gamma: float, dim_factor: float, args):

    Q = np.zeros([fair_env.observation_space.n, fair_env.memory_space.n, fair_env.action_space.n], dtype=float)
    visited = np.full([fair_env.observation_space.n, fair_env.memory_space.n], epsilon, dtype=float)  # for epsilon

    if args.preload:
        Q_file_name = './dataasets/Q/' + args.state_mode + '-' + str(max_ep_len) + '-' + str(args.prev_epsiodes) + '.npy'
        visited_file_name = './dataasets/visited/' + args.state_mode + '-' + str(max_ep_len) + '-' + str(args.prev_epsiodes) + '.npy'

        Q = np.load(Q_file_name)
        visited = np.load(visited_file_name)

    total_rewards = []
    total_donuts = []
    i = 0

    for i in range(1, episodes + 1):
        if i % 1000 == 0:
            print("Episode", i)
            evaluate(Q, printable=False)

        state, memory = fair_env.reset()

        done = False
        avg = []
        cum_reward = 0
        while not done:
            epsilon = visited[state, memory]
            avg.append(epsilon)
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, fair_env.people)
                if visited[state, memory] > 0.2:
                    visited[state, memory] *= dim_factor
            else:
                action = argmax_greedy(Q[state, memory])

            actual_memory = fair_env.memory.copy()
            next_state, next_memory, reward, done, _ = fair_env.step(action)
            cum_reward += reward
            max_action = argmax_greedy(Q[next_state, next_memory])
            Q[state, memory, action] =  Q[state, memory,action] + alpha * (reward + gamma * Q[next_state, next_memory, max_action] - Q[state,memory, action])

            if args.counterfactual:
                counterfactual_Q_update(fair_env, Q, state, action, reward, next_state, actual_memory, args)

            state = next_state
            memory = next_memory

        cum_eval_reward, donuts_allocated = evaluate(Q, printable=False)
        total_rewards.append(cum_eval_reward)
        total_donuts.append(donuts_allocated)


    evaluate(Q)
    if args.preload:
        Q_file_name = './datasets/Q/'+ current_time + "-cf" + str(args.counterfactual) + "-" + args.state_mode +'-' +str(max_ep_len) + '-' + str(args.epsiodes + args.prev_episodes) +'.npy'
        visited_file_name = './datasets/visited/' + current_time + "-cf" + str(args.counterfactual) + "-" + args.state_mode + '-' + str(
        max_ep_len) + '-' + str(args.epsiodes + args.prev_episodes) + '.npy'

    else:
        Q_file_name = './datasets/Q/' + current_time + "-cf" + str(args.counterfactual) + "-" + args.state_mode + '-' + str(max_ep_len) + '-' + str(args.episodes) + '.npy'
        visited_file_name = './datasets/visited/' + current_time + "-cf" + str(args.counterfactual) + "-" + args.state_mode + '-' + str(
        max_ep_len) + '-' + str(args.episodes) + '.npy'

    np.save(Q_file_name, Q)
    np.save(visited_file_name, visited)
    return total_rewards, total_donuts

def evaluate(Q, printable=False):
    done = False
    state, memory = fair_env.reset()
    cum_reward = 0
    donuts_allocated = 0
    while not done:
        action = argmax_greedy(Q[state, memory])
        next, next_memory, reward, done, _ = fair_env.step(action)
        cum_reward += reward
        if reward != 0:
            donuts_allocated += 1
        state = next
        memory = next_memory
    return cum_reward, donuts_allocated

def argmax_greedy(gamma_Q):
    actions = []
    max_q = np.max(gamma_Q)
    for i in range(0, fair_env.action_space.n):
        if abs(gamma_Q[i] - max_q) < 1e-5:
            actions.append(i)
    action = np.random.randint(0, int(len(actions)))
    return actions[action]


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""NSW Q-learning on Taxi""")
    prs.add_argument("-ep", dest="episodes", type=int, default=50000, required=False, help="episodes.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")

    prs.add_argument("-e", dest="epsilon", type=float, default=1.0, required=False, help="Exploration rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Discount rate\n")

    prs.add_argument("-d", dest="dim_factor", type=float, default=0.95, required=False,
                     help="Diminish factor for epsilon\n")
    prs.add_argument("-t", dest="tolerance", type=float, default=1e-5, required=False,
                     help="Loss threshold for Q-values between each episode\n")
    prs.add_argument("-sm", dest="state_mode", type=str, default='compact', required=False,
                     help="State represntation mode\n")

    prs.add_argument("-pl", dest="preload", type=bool, default=False, required=False,
                     help="Load Q from file\n")

    prs.add_argument("-prep", dest="prev_episodes", type=int, default=100000, required=False,
                     help="previous trained epiosdes\n")

    prs.add_argument("-cf", dest="counterfactual", type=bool, default=False, required=False,
                     help="Counterfactual Update\n")
    args = prs.parse_args()

    num_exp = 10
    num_people = 3
    pathprefix = "./datasets/donut-deterministic-q/"+ args.state_mode
    str_people = str(num_people)
    rewards_dataset_paths = [pathprefix + "-people" + str_people+ "-cf" + str(args.counterfactual) + "-" + current_time + f"_{i}.csv" for i in range(num_exp)]

    reward_data_all = []
    donuts_data_all = []
    x = 0

    seed = 2024
    random.seed(seed)
    fair_env = Donut(people=num_people, episode_length=max_ep_len, seed=seed,state_mode=args.state_mode, p=[0.8, 0.8, 0.8])
    reward_t, donuts_t = run_Q_learning(episodes=args.episodes, alpha=args.alpha, epsilon=args.epsilon, gamma=args.gamma, dim_factor=args.dim_factor, args=args)

    with open(rewards_dataset_paths[0], 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["preload", "prev_episodes", "episodes", "people", "maxeplen"])
        csv_writer.writerow([str(args.preload),str(args.prev_episodes), str(args.episodes),str(num_people), str(max_ep_len)])
        csv_writer.writerow(reward_t)
        csv_writer.writerow([""])
        csv_writer.writerow(donuts_t)

    reward_tmp = []
    donuts_tmp = []
    interv = 1000
    std_reward = []
    std_donut = []
    for j in range(0, len(reward_t), interv):
        end = j + interv
        end = min(end, len(reward_t))
        mn = np.mean(reward_t[j:end], axis=0)
        mn_d = np.mean(donuts_t[j:end], axis=0)
        std_r = np.std(reward_t[j:end], axis=0)
        std_d = np.std(donuts_t[j:end], axis=0)
        reward_tmp.append(mn)
        donuts_tmp.append(mn_d)
        std_reward.append(std_r)
        std_donut.append(std_d)
    reward_data_all.append(reward_tmp)
    donuts_data_all.append(donuts_tmp)

    std_reward = np.array(std_reward)
    std_donut= np.array(std_donut)
    x = [i for i in range(len(donuts_data_all[0]))]
    fig, ax = plt.subplots(1, 2)
    reward_means = np.mean(reward_data_all, axis=0)
    ax[0].plot(x, reward_means, label="Reward")
    ci = 1.96 * std_reward / np.sqrt(interv)
    ax[0].fill_between(x, (reward_means - ci), (reward_means + ci), alpha=.3)


    donut_means = np.mean(donuts_data_all, axis=0)
   # std_donut = np.std(donuts_data_all, axis=0)
    ax[1].plot(x, donut_means, label="Reward")
    ci = 1.96 * std_donut / np.sqrt(interv)

    ax[1].fill_between(x, (donut_means - ci), (donut_means + ci), alpha=.3)

    plt.savefig("./donut/reward" + args.state_mode + "-cf" + str(args.counterfactual) + "-" + current_time + ".png")
    plt.show()




