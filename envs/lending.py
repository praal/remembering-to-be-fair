import gym
import random
import numpy as np

class Lending(gym.Env):
    def __init__(self, people, episode_length, seed, state_mode = 'full',p = None):
        self.people = people
        self.seed = seed
        self.episode_length = episode_length
        self.action_space = gym.spaces.Discrete(self.people)
        self.state_mode = state_mode
        self.observation_space = gym.spaces.MultiBinary(np.power(2, self.people), seed=self.seed)

        self.default_credit = [4, 4, 7, 7]
        self.credit = self.default_credit.copy()

        self.loans = [0 for _ in range(2)]
        self.memory = [0 for _ in range(2)]
        self.success = self.episode_length

        self.curr_episode = 0
        self.default_obs = [1 for _ in range(self.people)]
        self.last_obs = self.default_obs

        if p is None:
            self.prob = [1.0 for _ in range(self.people)]
            self.stochastic = False
        else:
            self.prob = p
            self.stochastic = True
        self.reset(seed)

    def binary_state(self, s, length):
        zero_fill = int(np.ceil(np.log2(length)))
        ans = ""
        for i in s:
            ans += bin(i)[2:].zfill(zero_fill)

        int_ans = []
        for t in ans:
            int_ans.append(int(t))
        return int_ans

    def equal_reward(self, obs):
        ans = abs(obs[1] - obs[0])
        return -1.0 * ans

    def step(self, action):
        self.curr_episode += 1
        done = False
        obs = self.last_obs.copy()

        if self.curr_episode >= self.episode_length:
            done = True
        subg = 0
        if action > 1:
            subg = 1

        wrong_action = False
        if self.last_obs[action]:
            self.loans[subg] += 1
            self.memory[subg] += 1
            repayment = random.random()
            if repayment <= ((self.credit[action] + 2)/10):
                self.success += 1
                self.credit[action] = min(self.credit[action] + 1, 7)
            else:
                self.success -= 1
                self.credit[action] = max(self.credit[action] - 1, 0)
        else:
            wrong_action = True
        for i in range(self.people):
            p = random.random()
            if p <= self.prob[i]:
                obs[i] = 1
            else:
                obs[i] = 0

        self.last_obs = obs.copy()
        reward = self.equal_reward(self.loans.copy())

        obs = self.last_obs

        if self.state_mode == "reset":
            mn = min(self.memory)
            for i in range(2):
                self.memory[i] = self.memory[i] - mn

        elif self.state_mode == "equal":
            mn = min(self.memory)
            mx = max(self.memory)
            if mn == mx:
                for i in range(2):
                    self.memory[i] = self.memory[i] - mn

        out_memory = self.binary_state(self.memory.copy(), self.episode_length + 1)
        out_success = self.binary_state([self.success], (self.episode_length + 1) * 2)
        out_state = obs.copy()
        out_credit = self.binary_state(self.credit, 7)
        out_state.extend(out_success)
        out_state.extend(out_credit)

        if wrong_action:
            reward = -1 * self.episode_length
        if done and self.success < self.episode_length + int(self.episode_length / 10):
            reward = -10 * self.episode_length

        if self.state_mode == "rnn":
            return out_state, reward, done, {}
        return out_state, out_memory, reward, done, {}


    def reset(self, seed=None):
        self.loans = [0 for _ in range(2)]
        self.memory = [0 for _ in range(2)]
        self.curr_episode = 0
        self.success = self.episode_length
        self.credit = self.default_credit.copy()

        if seed is not None:
            self.seed = seed
            random.seed(self.seed)

        self.last_obs = self.default_obs
        if self.stochastic:
            for i in range(self.people):
                p = random.random()
                if p <= self.prob[i]:
                    self.last_obs[i] = 1
                else:
                    self.last_obs[i] = 0
        obs = self.last_obs.copy()
        out_memory = self.binary_state(self.memory.copy(), self.episode_length + 1)
        out_success = self.binary_state([self.success], (self.episode_length + 1) * 2)
        out_state = obs.copy()
        out_credit = self.binary_state(self.credit, 7)
        out_state.extend(out_success)
        out_state.extend(out_credit)

        if self.state_mode == "rnn":
            return out_state
        return out_state, out_memory



