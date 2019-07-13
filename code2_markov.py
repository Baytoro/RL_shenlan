#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

class Env(object):
    def __init__(self):
        self.S = ['s1', 's2', 's3', 's4', 's5']  # 状态集合
        self.P = dict(
            s1=dict(phone=[1, 0, 0, 0, 0], quit=[0, 1, 0, 0, 0]),
            s2=dict(phone=[1, 0, 0, 0, 0], study=[0, 0, 1, 0, 0]),
            s3=dict(study=[0, 0, 0, 1, 0], sleep=[0, 0, 0, 0, 1]),
            s4=dict(review=[0, 0, 0, 0, 1], noreview=[0, 0.2, 0.4, 0.4, 0])
        )
        self.R = dict(
            s1=dict(phone=-1, quit=0),
            s2=dict(phone=-1, study=-2),
            s3=dict(study=-2, sleep=0),
            s4=dict(review=10, noreview=-5)
        )

    def step(self, s, a):  # 状态转移函数和奖励函数
        s_n = np.random.choice(self.S, p=self.P[s][a])
        r = self.R[s][a]
        terminal = s_n == 's5'
        return s_n, r, terminal


class Agent(object):
    def __init__(self):
        self.A = ['quit', 'phone', 'study', 'sleep', 'review', 'noreview']
        self.available_actions = {
            's1': ['phone', 'quit'],
            's2': ['phone', 'study'],
            's3': ['study', 'sleep'],
            's4': ['review', 'noreview']
        }
        self.policy = self.random_policy

    def random_policy(self, s):
        a = None
        # 实现自己的代码
        if s in self.available_actions:
            available_actions = self.available_actions[s]
            a = np.random.choice(available_actions)
        return a




if __name__ == "__main__":
    # 仿真随机策略
    # 寻找最优策略
    # 我这里给出一次仿真的示例, 假设初始状态是s2
    env = Env()
    agent = Agent()
    gamma = 0.5
    max_time_step = 1000
    s = 's2'
    curr_gamma = 1
    g = 0  # 这次实验的回报值, 多次实验后平均，即得到v(s2)的估计
    for i in range(max_time_step):
        a = agent.random_policy(s)
        s, r, term = env.step(a)
        curr_gamma *= gamma
        g += curr_gamma * r
        if term:
            break
