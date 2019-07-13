#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random

BOARD_LEN = 3

class TicTacToeEnv(object):
    def __init__(self):
        self.data = np.zeros((BOARD_LEN, BOARD_LEN))  # data 表示棋盘当前状态，1和-1分别表示x和o，0表示空位
        self.winner = None  # 1/0/-1表示玩家一胜/平局/玩家二胜，None表示未分出胜负
        self.terminal = False  # true表示游戏结束
        self.current_player = 1  # 当前正在下棋的人是玩家1还是-1

    def reset(self):
        # 游戏重新开始，返回状态
        self.data = np.zeros((BOARD_LEN, BOARD_LEN))
        self.winner = None
        self.terminal = False
        state = self.getState()
        return state

    def getState(self):
        # 注意到很多时候，存储数据不等同与状态，状态的定义可以有很多种，比如将棋的位置作一些哈希编码等
        # 这里直接返回data数据作为状态
        return self.data

    def getReward(self):
        """Return (reward_1, reward_2)
        """
        if self.terminal:
            if self.winner == 1:
                return 1, -1
            elif self.winner == -1:
                return -1, 1
        return 0, 0

    def getCurrentPlayer(self):
        return self.current_player

    def getWinner(self):
        return self.winner

    def switchPlayer(self):
        if self.current_player == 1:
            self.current_player = -1
        else:
            self.current_player = 1

    def checkState(self):
        # 每次有人下棋，都要检查游戏是否结束
        # 从而更新self.terminal和self.winner
        # ----------------------------------
        # 实现自己的代码
        # ----------------------------------
        results = []
        # check row
        results.extend(np.sum(self.data, 1).tolist())
        # check column
        results.extend(np.sum(self.data, 0).tolist())
        # check diag
        results.append(np.sum(np.diag(self.data)))
        results.append(np.sum(np.diag(np.fliplr(self.data))))
        for v in results:
            if int(v) == BOARD_LEN:
                self.winner = 1
                self.terminal = True
                return
            elif int(v) == -BOARD_LEN:
                self.winner = -1
                self.terminal = True
                return

        # check whether it is a tie
        if np.sum(np.abs(self.data)) == BOARD_LEN * BOARD_LEN:
            self.winner = 0
            self.terminal = True

    def step(self, action):
        """action: is a tuple or list [x, y]
        Return:
            state, reward, terminal
        """
        # ----------------------------------
        # 实现自己的代码
        # ----------------------------------
        value = 1 if self.current_player == 1 else -1
        self.data[action[0], action[1]] = value
        self.checkState()
        next_state = self.getState()
        reward = self.getReward()
        self.switchPlayer()
        return next_state, reward, self.terminal


class RandAgent(object):
    def policy(self, state):
        """
        Return: action
        """
        # ----------------------------------
        # 实现自己的代码
        # ----------------------------------
        available_actions = np.where(state == 0)
        available_actions = zip(*available_actions)

        action = random.sample(list(available_actions), 1)[0]
        return action


# def main():
#     env = TicTacToeEnv()
#     agent1 = RandAgent()
#     agent2 = RandAgent()
#     state = env.reset()
#
#     # 这里给出了一次运行的代码参考
#     # 你可以按照自己的想法实现
#     # 多次实验，计算在双方随机策略下，先手胜/平/负的概率
#     while 1:
#         current_player = env.getCurrentPlayer()
#         if current_player == 1:
#             action = agent1.policy(state)
#         else:
#             action = agent2.policy(state)
#         next_state, reward, terminal = env.step(action)
#         print(next_state)
#         if terminal:
#             winner = 'Player1' if env.getWinner() == 1 else 'Player2'
#             print('Winner: {}'.format(winner))
#             break
#         state = next_state


if __name__ == "__main__":
    # main()
    print()