import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from envs.donut import Donut
import argparse
from datetime import datetime
import csv

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')


class Net(nn.Module):
    def __init__(self, states, actions, batch_size = 256, hidden_size = 16):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(states, 32)
        self.layer2 =  nn.Linear(32, 16)
        self.rnn = nn.GRU(16, hidden_size, batch_first=True)
        self.out_layer = nn.Linear(16, actions)
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def forward(self, x, prev_hidden=None):
        if prev_hidden == None:
            prev_hidden = torch.zeros([1, self.hidden_size])

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x, hidden = self.rnn(x, prev_hidden)
        x = self.out_layer(x)
        return x, hidden


class DQN():
    """docstring for DQN"""

    def __init__(self, num_states, num_actions, memory_capacity, learning_rate, args):
        super(DQN, self).__init__()

        self.eval_prev_hidden = None
        self.target_prev_hidden = None

        self.eval_net, self.target_net = Net(num_states, num_actions), Net(num_states, num_actions)

        def init_weights_rnn(m):
            classname = m.__class__.__name__
            if classname.find("Linear") != -1:
                m.weight.data.normal_(0, 1)
                m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.eval_net.apply(init_weights_rnn)

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


    def choose_action(self, state, hidden, greedy=False):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if greedy:
            with torch.no_grad():
                action_value, hidden = self.target_net.forward(state, hidden)

                action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
            return action, hidden

        action_value, hidden = self.eval_net.forward(state, hidden)
        if np.random.uniform() >= self.args.epsilon:  # greedy policy
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]

        else:  # random policy
            action = np.random.randint(0, self.num_actions)
            action = action
        return action, hidden

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
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
        q_eval, hidden = self.eval_net(batch_state)
        q_eval = q_eval.gather(1, batch_action)

        with torch.no_grad():
            q_next, tar_hidden = self.target_net(batch_next_state)
            q_next = q_next.detach()
            max_q_next = q_next.max(1)[0].view(self.args.batch_size, 1)

        q_target = batch_reward + self.args.gamma * max_q_next
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run(num_people, max_ep_len, memory_capacity, args, seed):

    env = Donut(people=num_people, episode_length=max_ep_len, seed=seed, state_mode=args.state_mode,
                p=[0.8, 0.8, 0.8, 0.8, 0.8])

    num_actions = env.action_space.n
    state = env.reset()
    print(state)
    num_states = len(state)

    dqn = DQN(num_states, num_actions, memory_capacity, args.lr, args)

    episodes = args.episodes
    print("Collecting Experience....")
    reward_list = []
    donuts_list = []

    for i in range(episodes):
        state = env.reset()
        hidden = torch.zeros([1, 16])
        ep_reward = 0
        ep_donuts = 0
        while True:
            state_input = state.copy()
            action, new_hidden = dqn.choose_action(state_input, hidden)
            next_state, reward, done, info = env.step(action)
            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward
            if reward != 0:
                ep_donuts += 1

            if dqn.memory_counter >= memory_capacity:
                dqn.learn()
                if done and i % 1000 == 0:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
            state = next_state
            hidden = new_hidden

        if dqn.args.epsilon > 0.2:
            dqn.args.epsilon = dqn.args.epsilon * 0.999
        ep_reward = 0
        ep_donuts = 0

        state = env.reset()
        state_input = state.copy()
        ep_reward = 0
        hidden = torch.zeros([1, 16])

        while True:
            action, new_hidden = dqn.choose_action(state_input, hidden, True)

            next_state, reward, done, info = env.step(action)

            ep_reward += reward
            if reward != 0:
                ep_donuts += 1
            if done:
                break
            state = next_state
            hidden = new_hidden
            state_input = state.copy()
        if i % 10 == 0:
            print(i, "-------------------------------")
            print("done", ep_reward, env.donuts.copy())
        reward_list.append(ep_reward)
        donuts_list.append(ep_donuts)
    return reward_list, donuts_list

def main():
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Fair Donut""")
    prs.add_argument("-ep", dest="episodes", type=int, default=50000, required=False, help="episodes.\n")
    prs.add_argument("-lr", dest="lr", type=float, default=0.002, required=False, help="learning rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=1.0, required=False, help="Exploration rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.95, required=False, help="Discount factor\n")
    prs.add_argument("-sm", dest="state_mode", type=str, default='deep', required=False,
                     help="State representation mode\n")
    prs.add_argument("-cf", dest="counterfactual", type=bool, default=False, required=False,
                     help="Counterfactual Update\n")
    prs.add_argument("-bs", dest="batch_size", type=int, default=256, required=False,
                     help="Batch Size\n")
    prs.add_argument("-qiter", dest="q_network_iterations", type=int, default=1000, required=False,
                     help="Q network iterations\n")
    prs.add_argument("-nexp", dest="num_exps", type=int, default=1, required=False,
                     help="Number of Experiments\n")
    args = prs.parse_args()

    num_people = 5
    seed = 2024
    max_ep_len = 100
    memory_capacity = 1000


    if args.counterfactual:
        args.batch_size = args.batch_size * 8
        memory_capacity *= 8

    num_exps = args.num_exps
    reward_list = []
    donut_list = []
    for i in range(num_exps):
        random.seed(seed + i + 1)
        np.random.seed(seed + i + 1)
        reward_t, donut_t = run(num_people, max_ep_len, memory_capacity, args, seed + i)
        reward_list.append(reward_t)
        donut_list.append(donut_t)
        save_plot_avg(reward_list, donut_list, args, i+1, num_exps, num_people, max_ep_len)


def save_plot_avg(reward_list_all, donuts_list_all, args, curr, num_exps, num_people, max_ep_len):

    pathprefix = "./datasets/donut-dqn/"+ args.state_mode
    rewards_dataset_paths = pathprefix + "-people" + str(num_people) + "-cf" + str(args.counterfactual) + "-" + current_time + ".csv"

    with open(rewards_dataset_paths, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([ "episodes", "people", "maxeplen", "learning rate", "batch size"])
        csv_writer.writerow(
            [str(args.episodes), str(num_people), str(max_ep_len), str(args.lr), str(args.batch_size)])

        for i in range(curr):
            csv_writer.writerow(reward_list_all[i])
            csv_writer.writerow([""])
            csv_writer.writerow(donuts_list_all[i])

    if curr < num_exps - 1:
        return
    reward_list_all = np.array(reward_list_all)
    donuts_list_all = np.array(donuts_list_all)

    interv = 10
    reward_list = []
    donuts_list = []
    for k in range(len(reward_list_all)):
        reward_list_t = []
        donuts_list_t = []
        for j in range(0, len(reward_list_all[k]), interv):
            end = j + interv
            end = min(end, len(reward_list_all[k]))
            mn = np.mean(reward_list_all[k][j:end], axis=0)
            mn_d = np.mean(donuts_list_all[k][j:end], axis=0)
            reward_list_t.append(mn)
            donuts_list_t.append((mn_d))
        reward_list.append(reward_list_t)
        donuts_list.append(donuts_list_t)
    reward_list = np.array(reward_list)
    donuts_list = np.array(donuts_list)

    mean_rewards = np.mean(reward_list, axis=0)
    mean_donuts = np.mean(donuts_list, axis=0)

    std_rewards = np.std(reward_list, axis=0)
    std_donuts = np.std(donuts_list, axis = 0)

    x = [i * 10 for i in range(len(mean_rewards))]
    fig, ax = plt.subplots(1, 2)

    ax[0].plot(x, mean_rewards, label="Reward")
    ci = 1.96 * std_rewards / np.sqrt(num_exps)
    ax[0].fill_between(x, (mean_rewards - ci), (mean_rewards + ci), alpha=.3)

    ax[1].plot(x, mean_donuts, label="Reward")
    ci = 1.96 * std_donuts / np.sqrt(num_exps)
    ax[1].fill_between(x, (mean_donuts - ci), (mean_donuts + ci), alpha=.3)

    ax[0].set_ylabel("Sum of NSW")
    ax[1].set_ylabel("Number of allocated donuts")

    title = args.state_mode
    if args.counterfactual:
        title += " with Counterfactuals"
    plt.suptitle(title, fontsize=16)
    plt.savefig("./donut/DQN-RNN-" + args.state_mode + "-cf" + str(args.counterfactual) + "-" + current_time + ".png")
    plt.show()


if __name__ == '__main__':
    main()