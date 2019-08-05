#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time


WORLD_SIZE = 5
N_ACTIONS = 4  # [North, South, West, East]
gamma = 0.9


class Env(object):
    def __init__(self):
        self.world_size = 5
        self.A_pos = [0, 1]
        self.A_n_pos = [4, 1]
        self.B_pos = [0, 3]
        self.B_n_pos = [2, 3]
        # 构建P(s'|s, a)和R(r|s, a)
        # 这是确定性的动态转移矩阵
        # 这里用前两维表示状态，第三维表示动作
        # 本节使用了动态规划的方法，因此P和R对Agent也是已知的
        # 动作上，0:N, 1:S, 2:W, 3:E
        self.P = np.empty((self.world_size, self.world_size, 4), dtype=np.object)
        self.R = np.zeros((self.world_size, self.world_size, 4))
        for i in range(self.world_size):
            for j in range(self.world_size):
                for a in range(4):
                    s = [i, j]
                    if a == 0:  # North
                        if i == 0:
                            s_n = s
                            r = -1
                        else:
                            s_n = [i - 1, j]
                            r = 0
                    elif a == 1:  # South
                        if i == self.world_size - 1:
                            s_n = s
                            r = -1
                        else:
                            s_n = [i + 1, j]
                            r = 0
                    elif a == 2:  # West
                        if j == 0:
                            s_n = s
                            r = -1
                        else:
                            s_n = [i, j - 1]
                            r = 0
                    else:  # East
                        if j == WORLD_SIZE - 1:
                            s_n = s
                            r = -1
                        else:
                            s_n = [i, j + 1]
                            r = 0

                    if s == self.A_pos:
                        s_n = self.A_n_pos
                        r = 10
                    elif s == self.B_pos:
                        s_n = self.B_n_pos
                        r = 5
                    self.P[i, j, a] = s_n
                    self.R[i, j, a] = r

    def step(self, s, a):
        s_n = self.P[s[0], s[1], a]
        r = self.R[s[0], s[1], a]
        return s_n, r

def value_evaluate(policy, env, max_step=1000, tol=1e-6):
    V = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.float32)  # 初始化
    update_step = 0
    for _ in range(max_step):
        new_V = V.copy()
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):  # 对每一个状态，更新其值函数
                qs = np.zeros((N_ACTIONS,), dtype=np.float32)   # 存储每个动作的Q值
                for a in range(N_ACTIONS):
                    n_s = env.P[i, j, a]
                    r = env.R[i, j, a]
                    n_V = V[n_s[0], n_s[1]]
                    qs[a] = r + gamma * n_V
                new_V[i, j] = np.sum(qs * policy[i, j])
        if np.sum(np.abs(V - new_V)) < tol:
            break
        V = new_V
    return V


def policy_improvement(env, V):
    policy = np.zeros((WORLD_SIZE, WORLD_SIZE, N_ACTIONS), dtype=np.float32)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            qs = np.zeros((N_ACTIONS,), dtype=np.float32)
            for a in range(N_ACTIONS):
                n_s = env.P[i, j, a]
                r = env.R[i, j, a]
                qs[a] = r + gamma * V[n_s[0], n_s[1]]
            p = (np.abs(qs - np.max(qs)) < 1e-6)   # 这里没有使用==，主要是避免精度损失
            p = np.array(p, dtype=np.float32) / np.sum(p)  # 默认情况下bool型array会当成int型处理，所以在归一化之前，需要转化浮点型
            policy[i, j] = p
    return policy

def policy_iteration(max_iter=1000, max_step=1000, tol=1e-6):
    env = Env()
    # 初始化策略
    policy = np.full((WORLD_SIZE, WORLD_SIZE, N_ACTIONS), 1. / N_ACTIONS, dtype=np.float32)
    mean_values = []
    run_times = []
    last_V = None
    st = time.time()
    for _ in range(max_iter):
        V = value_evaluate(policy, env, max_step, tol)
        policy = policy_improvement(env, V)
        mean_values.append(np.mean(V))
        run_times.append(time.time() - st)
        if last_V is not None and np.sum(np.abs(V - last_V)) < tol:
            break
        last_V = V
    return V, mean_values, policy, run_times


def value_iteration(max_iter=1000, max_step=1000, tol=1e-6):
    env = Env()
    V = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.float32)
    mean_values = []
    run_times = []
    st = time.time()
    for _ in range(max_iter):
        new_V = V.copy()
        update_steps = 0
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                qs = np.zeros((N_ACTIONS,), dtype=np.float32)
                for a in range(N_ACTIONS):
                    n_s = env.P[i, j, a]
                    r = env.R[i, j, a]
                    qs[a] = r + gamma * V[n_s[0], n_s[1]]
                    update_steps += 1
                new_V[i, j] = np.max(qs)
        mean_values.append(np.mean(V))
        run_times.append(time.time() - st)
        if np.sum(np.abs(V - new_V)) < tol:
            break
        V = new_V
    return V, mean_values, run_times

def inplace_value_iteration(max_iter=1000, max_step=1000, tol=1e-6):
    env = Env()
    V = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.float32)
    mean_values = []
    run_times = []
    st = time.time()
    for _ in range(max_iter):
        update_steps = 0
        last_V = V.copy()   # 本质上inplace更新是不需要存两个V的，不过这里为了和之前的方法对比，也使用了相同的停止条件
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                qs = np.zeros((N_ACTIONS,), dtype=np.float32)
                for a in range(N_ACTIONS):
                    n_s = env.P[i, j, a]
                    r = env.R[i, j, a]
                    qs[a] = r + gamma * V[n_s[0], n_s[1]]
                    update_steps += 1
                V[i, j] = np.max(qs)   # 这里直接对V进行更新
        run_times.append(time.time() - st)
        mean_values.append(np.mean(V))
        if np.sum(np.abs(V - last_V)) < tol:
            break
    return V, mean_values, run_times