import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from envs.lending import Lending
import argparse
from datetime import datetime
import csv

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')


class Net(nn.Module):

    def __init__(self, states, actions):
        super(Net, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(states, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, actions))

    def forward(self, x):
        action_prob = self.model(x)
        return action_prob


class DQN():

    def __init__(self, num_states, num_actions, memory_capacity, learning_rate, args):
        super(DQN, self).__init__()


        self.eval_net, self.target_net = Net(num_states, num_actions), Net(num_states, num_actions)

        def init_weights(m):
            if hasattr(m, 'weight'):
                nn.init.orthogonal_(m.weight.data)
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias.data, 0)

        self.eval_net.apply(init_weights)
        self.target_net.load_state_dict(
            self.eval_net.state_dict())

        self.num_states = num_states
        self.num_actions = num_actions
        self.memory_capacity = memory_capacity
        self.args = args

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_capacity, num_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()


    def choose_action(self, state, greedy=False):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # get a 1D array
        if greedy:
            with torch.no_grad():
                action_value = self.target_net.forward(state)
                action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]

        elif np.random.uniform() >= self.args.epsilon:  # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]

        else:  # random policy
            action = np.random.randint(0, self.num_actions)
            action = action
        return action

    def store_transition(self, state, memory, action, reward, next_state, next_memory):
        transition = np.hstack((state, memory, [action, reward], next_state, next_memory))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        if self.learn_step_counter % self.args.q_network_iterations == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_capacity, self.args.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.num_states])
        batch_action = torch.LongTensor(batch_memory[:, self.num_states:self.num_states + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.num_states + 1:self.num_states + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.num_states:])
        # q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)

        with torch.no_grad():
            q_next = self.target_net(batch_next_state).detach()
            max_q_next = q_next.max(1)[0].view(self.args.batch_size, 1)

        q_target = batch_reward + self.args.gamma * max_q_next
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def counterfactual_update(self, fair_env, state, action, prev_reward, next_state, actual_memory, max_ep_len, state_mode, num_updates=5):

        for i in range(len(actual_memory)):
            for k in range(1, num_updates+1):
                possible_memory = actual_memory.copy()
                possible_memory[i] += k
                if possible_memory[i] >= max_ep_len + 1:
                    break

                subg = 0
                if action > 1:
                    subg = 1
                if possible_memory[subg] == max_ep_len:
                    continue

                next_memories = possible_memory.copy()
                next_memories[subg] += 1
                reward = -1.0 * abs(possible_memory[1] - possible_memory[0])
                if prev_reward < -1 * max_ep_len:
                    reward = prev_reward

                if state_mode == 'reset':
                    mn = min(possible_memory)
                    for j in range(2):
                        possible_memory[j] = possible_memory[j] - mn
                    mn = min(next_memories)
                    for j in range(2):
                        next_memories[j] = next_memories[j] - mn

                memory = fair_env.binary_state(possible_memory, max_ep_len + 1)
                next_memory = fair_env.binary_state(next_memories, max_ep_len + 1)
                self.store_transition(state, memory, action, reward, next_state, next_memory)



def run(num_people, max_ep_len, memory_capacity, args, seed):

    env = Lending(people=num_people, episode_length=max_ep_len, seed=seed, state_mode=args.state_mode,
                p=[0.9, 0.9, 0.9, 0.9, 0.9])

    num_actions = env.action_space.n
    state, memory = env.reset()
    print(state, memory)
    num_states = len(state) + len(memory)

    dqn = DQN(num_states, num_actions, memory_capacity, args.lr, args)

    episodes = args.episodes
    print("Collecting Experience....")
    reward_list = []

    for i in range(episodes):
        state, memory = env.reset()
        ep_reward = 0
        while True:
            state_input = state.copy()
            state_input.extend(memory)
            action = dqn.choose_action(state_input)
            actual_memory = env.loans.copy()
            next_state, next_memory, reward, done, info = env.step(action)
            # print("train", env.donuts.copy(), reward, action, dqn.args.epsilon)
            dqn.store_transition(state, memory, action, reward, next_state, next_memory)
            if args.counterfactual:
                dqn.counterfactual_update(env, state, action, reward, next_state, actual_memory, max_ep_len, args.state_mode)
            ep_reward += reward

            if dqn.memory_counter >= memory_capacity:
                dqn.learn()
                if done and i % 1000 == 0:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
            state = next_state
            memory = next_memory


        if dqn.args.epsilon > 0.2:
            dqn.args.epsilon = dqn.args.epsilon * 0.999
        ep_reward = 0

        state, memory = env.reset()
        state_input = state.copy()
        state_input.extend(memory)
        ep_reward = 0
        while True:
            action = dqn.choose_action(state_input, True)

            next_state, next_memory, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                break
            state = next_state
            memory = next_memory
            state_input = state.copy()
            state_input.extend(memory)
        if i % 10 == 0:
            print(i, "-------------------------------")
            print("done", ep_reward, env.loans, env.success)
        reward_list.append(ep_reward)

    return reward_list

def main():
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Fair Lending""")
    prs.add_argument("-ep", dest="episodes", type=int, default=50000, required=False, help="episodes.\n")
    prs.add_argument("-lr", dest="lr", type=float, default=0.0001, required=False, help="learning rate.\n")

    prs.add_argument("-e", dest="epsilon", type=float, default=1.0, required=False, help="Exploration rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.95, required=False, help="Discount rate\n")

    prs.add_argument("-sm", dest="state_mode", type=str, default='deep', required=False,
                     help="State represntation mode\n")

    prs.add_argument("-cf", dest="counterfactual", type=bool, default=False, required=False,
                     help="Counterfactual Update\n")
    prs.add_argument("-bs", dest="batch_size", type=int, default=64, required=False,
                     help="Batch Size\n")
    prs.add_argument("-qiter", dest="q_network_iterations", type=int, default=100, required=False,
                     help="Q network iterations\n")

    prs.add_argument("-nexp", dest="num_exps", type=int, default=1, required=False,
                     help="Number of Experiments\n")
    args = prs.parse_args()

    num_people = 4
    seed = 2024
    max_ep_len = 40
    memory_capacity = 1000

    if args.counterfactual:
        args.batch_size = args.batch_size * 8
        memory_capacity *= 8

    num_exps = args.num_exps
    reward_list = []
    for i in range(num_exps):
        random.seed(seed)
        np.random.seed(seed + i + 1)
        reward_t = run(num_people, max_ep_len, memory_capacity, args, seed + i)
        reward_list.append(reward_t)
    save_plot_avg(reward_list, args, num_exps, num_people, max_ep_len)


def save_plot_avg(reward_list_all, args, num_exps, num_people, max_ep_len):

    pathprefix = "./datasets/lending/"+ args.state_mode
    rewards_dataset_paths = pathprefix + "-people" + str(num_people) + "-cf" + str(args.counterfactual) + "-" + current_time + ".csv"

    with open(rewards_dataset_paths, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([ "episodes", "people", "maxeplen", "learning rate", "batch size"])
        csv_writer.writerow(
            [str(args.episodes), str(num_people), str(max_ep_len), str(args.lr), str(args.batch_size)])

        for i in range(num_exps):
            csv_writer.writerow(reward_list_all[i])
            csv_writer.writerow([""])
            csv_writer.writerow([" "])

    reward_list_all = np.array(reward_list_all)

    interv = 10
    reward_list = []
    for k in range(len(reward_list_all)):
        reward_list_t = []
        for j in range(0, len(reward_list_all[k]), interv):
            end = j + interv
            end = min(end, len(reward_list_all[k]))
            mn = np.mean(reward_list_all[k][j:end], axis=0)
            reward_list_t.append(mn)
        reward_list.append(reward_list_t)
    reward_list = np.array(reward_list)

    mean_rewards = np.mean(reward_list, axis=0)
    std_rewards = np.std(reward_list, axis=0)

    x = [i * 10 for i in range(len(mean_rewards))]
    fig, ax = plt.subplots(1, 1)

    ax.plot(x, mean_rewards, label="Reward")
    ci = 1.96 * std_rewards / np.sqrt(num_exps)
    ax.fill_between(x, (mean_rewards - ci), (mean_rewards + ci), alpha=.3)


    ax.set_ylabel("Sum of NSW")

    title = args.state_mode
    if args.counterfactual:
        title += " with Counterfactuals"
    plt.suptitle(title, fontsize=16)
    plt.savefig("./datasets/lending/DQN" + args.state_mode + "-cf" + str(args.counterfactual) + "-" + current_time + ".png")
    plt.show()



if __name__ == '__main__':
    main()