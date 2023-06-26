# import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# if GPU is to be used
device = th.device("cuda" if th.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

from ..agent import Agent
from typing import *
from gym import Env

class DQNAgent(Agent):
    
    def __init__(self, name, env: Env, 
                 lr: float = 1e-4,
                 action_step: float = 1, 
                 reference = False,
                 epsilon_start: float = 0.9,
                 epsilon_end: float = 0.05,
                 gamma: float = 0.99,
                 batch_size: int = 128):
        super().__init__(name, 'DQN', env)
        
        self.reference = reference 
        
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the ``AdamW`` optimizer
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = epsilon_start
        self.EPS_END = epsilon_end
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = lr

        self.n_agents = self._env.n_turbines
        # Get number of actions from gym action space        
        self.n_actions = int(((2 / action_step) + 1) * self.n_agents)
        self.action_step = action_step
        
        
        # Get the number of state observations
        n_observations = self.observation_shape

        self.policy_net = DQN(n_observations, self.n_actions).to(device)
        self.target_net = DQN(n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def find_action(self, observation, in_eval=False):
        
        # global steps_done
        sample = random.random()
        # eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * steps_done / self.EPS_DECAY)
        # self.eps_threshold
        # steps_done += 1
        
        # if True:
        if sample > self.eps_threshold or in_eval:
            with th.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = th.as_tensor(observation, device=device, dtype=th.float32)
                
                q =  self.policy_net(obs)
                # print(q.shape)
                # return q
                # action = th.tensor([q[i].argmax().item() for i in range(self.n_agents)], device=device)
                # return action
        else:
            # return th.tensor([[self._env.action_space.sample()]], device=device, dtype=th.long)
            # return th.randint(self.n_actions, (self.n_agents,),device=device)
            q = th.randn(self.n_actions).unsqueeze(dim=0)
            # print(ret.shape)
            # return ret
            
        action = [q.view(self.n_agents, 3)[i].argmax().item() for i in range(self.n_agents)]
        action[1] += 3
        action[2] += 6
        
        return th.tensor(action, device=device).unsqueeze(dim=0)
        
        
    def map_action(self, q:th.Tensor):
        # action = [q.view(3, 3)[i].argmax().item() for i in range(self.n_agents)]
        action = list(q.squeeze().cpu().numpy())
        action[1] -= 3
        action[2] -= 6
        
        return list(map(lambda x: x * self.action_step - 1 , action))
        
        
    # def optimize_model(self):
    def learn(self, i_episode, log_every, last_step):
        
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = th.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=th.bool)
        non_final_next_states = th.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = th.cat(batch.state)
        action_batch = th.cat(batch.action)
        reward_batch = th.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = th.zeros(self.BATCH_SIZE, self.n_agents, device=device)
        with th.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).view(self.BATCH_SIZE, self.n_agents, 3).max(-1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch.repeat_interleave(self.n_agents, -1).view(self.BATCH_SIZE, self.n_agents)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)).float()
        loss = criterion(state_action_values.float(), expected_state_action_values.float()).float()
        # print(loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        th.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        if i_episode % log_every == 0 and last_step:
            name = 'loss_dqn' if self.reference else 'loss' 
            
            print(f'{name},{i_episode},{loss}')
            
            if self.file:
                with open(self.file, 'a') as f:
                    f.write(f'{name},{i_episode},{loss}\n')
        
    
    def run(self,
            max_step: int = 500, #500
            n_episode: int = 400, #800
            # i_episode: int = 0,
            n_epoch: int = 25,
            epsilon: float = 0.9,
            render: bool = False,
            log: bool = True,
            log_every: int = 10,
            log_directory: Optional[str] = None,
            evaluate: bool = True,
            eval_steps: Optional[int] = 500,
            eval_every: Optional[int] = 10,):
        
        self.file = log_directory

        for i_episode in range(1, n_episode+1):
            
            self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * i_episode / self.EPS_DECAY)
            
            # Initialize the environment and get it's state
            state = self._env.reset()
            
            state = th.tensor(state, dtype=th.float32, device=device).unsqueeze(0)
            
            score = []
            for t in range(max_step):
                action = self.find_action(state)
                observation, reward, _, _ = self._env.step(self.map_action(action))
                
                score.append(sum(reward))
                
                reward = th.tensor([sum(reward)], device=device)


                next_state = th.tensor(observation, dtype=th.float32, device=device).unsqueeze(0)

                # # Store the transition in memory
                # #action back to index
                # action[1] = action[1] + 3
                # action[2] = action[2] + 6     
                # # 0, 1, 2, 3, 4, 5, 6, 7, 8
        
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.learn(i_episode, log_every, t == max_step - 1)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)
                
            # every episode print episodic power output
            if log and i_episode % log_every == 0:
                avg_score = sum(score) / max_step
                name = 'train_dqn' if self.reference else 'train' 

                print(f'{name},{i_episode},{avg_score}')
                
                if self.file:
                    with open(self.file, 'a') as f:
                        f.write(f'{name},{i_episode},{avg_score}\n')
                
            # evaluation
            if evaluate and i_episode % eval_every == 0:
                self._eval(self._env, eval_steps, i_episode)
        
                    
    def get_log_dict(self):
        return {'loss/q_loss': self.loss.item()}


    def _eval(self, env: Env, eval_steps, i_episode):
        
        obs = env.reset()
        steps = 0
        score = []
        
        while steps < eval_steps:
                steps+=1 
                
                #find actions
                action = self.find_action(obs, True)
                
                # step through environment with actions
                # next_obs, reward, terminate, _ = self._env.step(action)
                next_obs, reward, _, _ = self._env.step(self.map_action(action.cpu()))
                
                obs = next_obs
                score.append(sum(reward)) # global (total score) (cumulative sum)
        
        avg_score = sum(score) / eval_steps
        name = 'DQN' if self.reference else 'eval' 

        print(f'{name},{i_episode},{avg_score}')
        
        if self.file:
            with open(self.file, 'a') as f:
                f.write(f'{name},{i_episode},{avg_score}\n')