#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

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
                        pass  # 实现自己的逻辑
                    else:     # East
                        pass   # 实现自己的逻辑

                    if s == self.A_pos:
                        s_n = self.A_n_pos
                        r = 10
                    elif s == self.B_pos:
                        s_n = self.B_n_pos
                        r = 5

                    self.P[i, j, a] = s_n
                    self.R[i, j, a] = r

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

