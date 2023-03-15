import numpy as np

"""###EpsilonGreedyAgent"""

import operator
from amalearn.agent import AgentBase

class EpsilonGreedyAgent(AgentBase):
    def __init__(self, id, environment, lr=None, eps=None):
        super(EpsilonGreedyAgent, self).__init__(id, environment)
        self.act_num = self.environment.available_actions()
        self.m0 = 0      # mean
        self.actions_estimation = dict(zip(range(self.act_num), [self.m0 for i in range(self.act_num)]))
        self.length = 0
        self.trial = {}
        self.lr = lr
        self.eps = eps
        print(self.actions_estimation)
    
    def reset(self):
        self.actions_estimation = dict(zip(range(self.act_num), [self.m0 for i in range(self.act_num)]))
        self.length = 0
        self.trial = {}
        print(self.actions_estimation)

    def u(self, r):
        return -800*((np.absolute(r))**(0.3))

    def update_estimation(self, action, r):
        u = self.u(r)
        if self.lr is None:
          self.actions_estimation[action] = \
            ((self.actions_estimation[action]*self.length)+u)/float(self.length+1)
        else:
          self.actions_estimation[action] = ((1-self.lr)*self.actions_estimation[action])+(u*self.lr)

        self.trial[self.length] = (action, u)
        self.length += 1

    def select_action(self):
        p = np.random.random()
        eps = self.eps
        if self.eps is None:
          eps = (1/float(self.length+1))-((1/float(self.length+1))/float(self.act_num))
        best_q_index = max(self.actions_estimation.items(), key=operator.itemgetter(1))[0]
        candid = [i for i in range(self.act_num) if i != best_q_index]
        if p < eps: 
          j = np.random.choice(len(candid))
          best_q_index = candid[j]
        return best_q_index

    def get_stat(self):
        return self.trial

    def get_best_action(self):
        best_q_index = max(self.actions_estimation.items(), key=operator.itemgetter(1))[0]
        return (best_q_index, self.actions_estimation[best_q_index])

    def take_action(self) -> (object, float, bool, object):
        action = self.select_action()
        obs, r, d, i = self.environment.step(action)
        self.update_estimation(action, r)
        print(obs, r, self.u(r), d, i)
        self.environment.render()
        return obs, r, self.u(r), d, i



"""### draw regret"""

def draw_list(data, xlabel, ylabel, data_label):
  plt.plot(range(len(data)), data, label=data_label)
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.grid()
  plt.legend()

def draw_lists_compare(data1, data2, xlabel, ylabel, data1_label, data2_label):
  plt.plot(range(len(data1)), data1, label=data1_label)
  plt.plot(range(len(data2)), data2, label=data2_label)
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.grid()
  plt.legend()

import matplotlib.pyplot as plt

def regret_per_run(data, best_action_mean, h):
    sum = 0
    for i in data.keys():
      if i > (h-1):
        break
      sum += data[i][1]
    return h*best_action_mean-sum

def regret(data, max_mean):
  regret_list = []
  for run in data:
    run_regret = []
    for j in range(len(run.keys())):
      run_regret.append(regret_per_run(run, max_mean, j+1))
    regret_list.append(run_regret)

  return regret_list

"""###draw optimal action %"""

def optimal_action_ratio(data, opta):
  res = []
  trial_num = len(data[0].keys())

  for j in range(trial_num):
    temp = []
    for i in data:
      temp.append(i[j][0])
    res.append((temp.count(opta))/len(temp))
  
  return res

"""###UCB agent"""

import operator
from amalearn.agent import AgentBase

class UCBAgent(AgentBase):
    def __init__(self, id, c, environment):
        super(UCBAgent, self).__init__(id, environment)
        self.act_num = self.environment.available_actions()
        self.m0 = 0      # mean
        self.actions_estimation = dict(zip(range(self.act_num), [self.m0 for i in range(self.act_num)]))
        self.act_count = dict(zip(range(self.act_num), [0 for i in range(self.act_num)]))
        self.length = 0
        self.trial = {}
        self.c = c
        print(self.actions_estimation)
    
    def reset(self):
        self.actions_estimation = dict(zip(range(self.act_num), [self.m0 for i in range(self.act_num)]))
        self.act_count = dict(zip(range(self.act_num), [0 for i in range(self.act_num)]))
        self.length = 0
        self.trial = {}
        print(self.actions_estimation)

    def u(self, r):
        return -800*((np.absolute(r))**(0.3))

    def update_estimation(self, action, r):
        u = self.u(r)
        self.actions_estimation[action] = ((self.actions_estimation[action]*self.length)+u)/float(self.length+1)

        self.trial[self.length] = (action, u)
        self.act_count[action] += 1
        self.length += 1

    def select_action(self):
        cond = {}
        for i in range(self.act_num):
          if self.act_count[i] == 0:
            cond[i] = float('inf')
            continue
          cond[i] = self.actions_estimation[i] + self.c * (np.sqrt((np.log(self.length))/self.act_count[i]))
        return max(cond.items(), key=operator.itemgetter(1))[0]

    def get_stat(self):
        return self.trial

    def get_best_action(self):
        best_q_index = max(self.actions_estimation.items(), key=operator.itemgetter(1))[0]
        return (best_q_index, self.actions_estimation[best_q_index])

    def take_action(self) -> (object, float, bool, object):
        action = self.select_action()
        obs, r, d, i = self.environment.step(action)
        self.update_estimation(action, r)
        print(obs, r, self.u(r), d, i)
        self.environment.render()
        return obs, r, self.u(r), d, i
