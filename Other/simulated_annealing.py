from random import random
import math
import sys
import numpy as np


class SimAnneal:
    def __init__(self, target_text="min"):
        self.target_text = target_text
        if self.target_text == "min":
            self.init_res = sys.maxsize
        else:
            self.init_res = -sys.maxsize

    def new_var(self, old_list, t):
        new_list = [i + (random() * 2 - 1) for i in old_list]
        return new_list

    def judge(self, func, new, old, t):
        dE = func(new) - func(old) if self.target_text == "max" else func(old) - func(new)
        if dE >= 0:
            x, ans = new, func(new)
        else:
            if math.exp(dE / t) > random():
                x, ans = new, func(new)
            else:
                x, ans = old, func(old)
        return x, ans


def map_range(one_d_range):
    return (one_d_range[1] - one_d_range[0]) * random() + one_d_range[0]


class OptimalSolution:
    def __init__(self, temperature_0=100, tem_delta=0.98, tem_final=1e-8,
                 markov_chain=2000, result=0):
        self.temperature_0 = temperature_0
        self.tem_delta = tem_delta
        self.tem_final = tem_final
        self.markov_chain = markov_chain
        self.result = result
        self.val_nd = [0]

    def solution(self, sa_new_var, sa_judge, judge_text, value_range, func):
        ti = self.temperature_0
        n_dim = len(value_range)
        f = max if judge_text == "max" else min
        nf = np.amax if judge_text == "max" else np.amin
        loops = 0

        while ti > self.tem_final:
            res_temp = []
            pre_v = [[map_range(value_range[j]) for _ in range(self.markov_chain)] for j in range(n_dim)]
            new_v = [sa_new_var(pre_v[j], t=ti) for j in range(n_dim)]

            for i in range(self.markov_chain):
                bool_v = True
                for j in range(n_dim):
                    bool_v &= (value_range[j][0] <= new_v[j][i] <= value_range[j][1])
                if bool_v is True:
                    res_temp.append(sa_judge(func=func, new=[new_v[k][i] for k in range(n_dim)],
                                             old=[pre_v[k][i] for k in range(n_dim)], t=ti))
                else:
                    continue
                loops += 1

            sol_temp = np.array(res_temp)
            extreme_temp = nf(sol_temp[:, 1])
            re = np.where(sol_temp == extreme_temp)

            result_temp = f(self.result, extreme_temp)
            self.val_nd = self.val_nd if result_temp == self.result else sol_temp[re[0][0], 0]
            self.result = result_temp
            ti *= self.tem_delta
            output = (self.val_nd, self.result)
            print(output)

        print("Total loops = %d" % loops)


def func0(w):
    x, = w
    fx = x * x - 4 * x + 1
    return fx


def func1(w):
    x, = w
    fx = x + 10 * math.sin(5 * x) + 7 * math.cos(4 * x)
    return fx


def func2(w):
    x, y = w
    fxy = y * np.sin(2 * np.pi * x) + x * np.cos(2 * np.pi * y)
    return fxy


def func3(w):
    x, a, b = w
    if x < a:
        return 1
    elif a <= x <= b:
        if a == b:
            print("输入值不符合")
            raise ValueError
        return (b - x) / (b - a)
    else:
        return 0


def func4(w):
    x, a, b = w
    if a <= x <= b:
        if a == b:
            print("输入值不符合")
            raise ValueError
        return (b - x) / (b - a)
    else:
        return -sys.maxsize


def func5(w):
    x, a, b, c, d = w
    if a < b < c < d:
        if a < x < b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return 1
        elif c < x < d:
            return (d - x) / (d - c)
        else:
            return 2
    else:
        return sys.maxsize


if __name__ == "__main__":
    sim_anneal = SimAnneal(target_text="min")
    calculate = OptimalSolution(markov_chain=3000, result=sim_anneal.init_res)

    #calculate.solution(sa_new_var=sim_anneal.new_var, sa_judge=sim_anneal.judge,
    #                   judge_text=sim_anneal.target_text, value_range=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)], func=func5)

    # calculate.solution(sa_new_var=sim_anneal.new_var, sa_judge=sim_anneal.judge,
    #                    judge_text=sim_anneal.target_text, value_range=[(0, 1), (0, 1), (0, 1)], func=func4)

    calculate.solution(sa_new_var=sim_anneal.new_var, sa_judge=sim_anneal.judge,
                       judge_text=sim_anneal.target_text, value_range=[(-1000, 1000)], func=func0)
