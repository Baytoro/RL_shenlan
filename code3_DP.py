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
