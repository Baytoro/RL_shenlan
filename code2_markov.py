#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Env(object):
    def __init__(self):
        self.S = ['s1', 's2', 's3', 's4', 's5']  # 状态集合

    def step(self, s, a):  # 状态转移函数和奖励函数
        s_n = None
        r = None
        terminal = False   # 是否进入终止状态
        # 实现自己的代码
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

    def random_policy(self, s):
        a = None
        # 实现自己的代码
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
