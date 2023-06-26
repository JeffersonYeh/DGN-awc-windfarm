import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

# import agent as Agent
# from ..windfarm.agent.agent import Agent
from ..agent import Agent
from .graph_wind_farm_env import GraphWindFarmEnv
import random



USE_CUDA = torch.cuda.is_available()
print(f'CUDA AVAILABLE: {USE_CUDA}')

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class Encoder(nn.Module):
	def __init__(self, din=32, hidden_dim=128):
		super(Encoder, self).__init__()
		self.fc = nn.Linear(din, hidden_dim)

	def forward(self, x):
		embedding = F.relu(self.fc(x))
		return embedding

class AttModel(nn.Module):
	def __init__(self, n_node, din, hidden_dim, dout):
		super(AttModel, self).__init__()
		self.fcv = nn.Linear(din, hidden_dim)
		self.fck = nn.Linear(din, hidden_dim)
		self.fcq = nn.Linear(din, hidden_dim)
		self.fcout = nn.Linear(hidden_dim, dout)

	def forward(self, x, mask):
		v = F.relu(self.fcv(x))
		q = F.relu(self.fcq(x))
		k = F.relu(self.fck(x)).permute(0,2,1)
		att = F.softmax(torch.mul(torch.bmm(q,k), mask) - 9e15*(1 - mask),dim=2)

		out = torch.bmm(att,v)
		#out = torch.bmm(mask,v) #commnet
		#out = torch.add(out,v)
		#out = F.relu(self.fcout(out))
		return out

class Q_Net(nn.Module):
	def __init__(self, hidden_dim, dout):
		super(Q_Net, self).__init__()
		self.fc = nn.Linear(hidden_dim, dout)

	def forward(self, x):
		q = self.fc(x)
		return q

class DGN(nn.Module):
	def __init__(self,n_agent,num_inputs,hidden_dim,num_actions):
		super(DGN, self).__init__()
		
		self.encoder = Encoder(num_inputs,hidden_dim)
		self.att_1 = AttModel(n_agent,hidden_dim,hidden_dim,hidden_dim)
		self.att_2 = AttModel(n_agent,hidden_dim,hidden_dim,hidden_dim)
		self.q_net = Q_Net(hidden_dim,num_actions)
		
	def forward(self, x, mask):
        # x dim:
        # mask dim: 
		h1 = self.encoder(x)
		h2 = self.att_1(h1, mask)
		h3 = self.att_2(h2, mask)
		q = self.q_net(h3)
		return q 


from typing import *
from .buffer import ReplayBuffer
from gym import Env

class DGNAgent(Agent):
    
    def __init__(self, name, env: Env,
                #  n_agents: int,
                #  observation_space: int,
                #  n_actions: int, 
                 hidden_dim: int = 64,
                 buffer_size: int = 1000000,
                 batch_size: int = 64,
                 action_step: float = 1,
                 
                #  max_step: int = 250, #500
                 GAMMA: float = 0.99,
                 lr: float = 1e-4,
                 epsilon_start: float = 0.9,
                 epsilon_end: float = 0.05,
                #  n_episode: int = 400, #800
                #  i_episode: int = 0,
                #  n_epoch: int = 25,
                #  epsilon: float = 0.9,
                #  score: int,
                #  comm_flag: int, # communication flag, default = 1
                #  threshold: float,
                #  tau: float,
                #  cost_all: int,
                #  cost_comm: int
                
                 graph_representation: Literal['undirected', 'directed'] = 'undirected',
                 optimal_distance: float = 10,
                 stream: bool = True
                 ):
        
        # max_step = 500
        # GAMMA = 0.99
        # n_episode = 800
        # i_episode = 0
        # capacity = 65000 
        # n_epoch = 25
        # epsilon = 0.9
        # score = 0
        # comm_flag = 1
        # threshold = -0.1
        # tau = 0.98
        # cost_all = 0
        # cost_comm = 0
        
        """
        
        Parameters:
        -----------
        name : str
            name of agent
            
        env : Env
            environment
            
        n_agent : int
            number of agents
            
        observation_space : int
            size of observation space per agent
            
        n_action : int
            action per agent
            
        hidden_dim : int
            size network hidden dimension
        
        buffer_size : int
            ReplayBuffer size
            
        batch_size : int
            training batch size
         
        """
        super().__init__(name, 'DGN', GraphWindFarmEnv.from_WindFarmEnv(env, graph_representation, optimal_distance, stream))
        
        self.n_agents = self._env.n_turbines
        self.observation_shape = int(self.observation_shape / self.n_agents)
        
        self.n_actions = int((2 / action_step) + 1)
        self.action_step = action_step
        # self.n_actions = 21 # map [-1, 1] to fixed actions of interval 0.1
        # assuming action representation is [-1, 1], create some action method --> look into q learning methods
        
        # self.n_agents = n_agents
        # self.epsilon = epsilon
        # self.n_actions = n_actions
        
        # self.i_episode = i_episode
        # self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.GAMMA = GAMMA
        self.EPS_START = epsilon_start
        self.EPS_END = epsilon_end
        # self.max_step = max_step
        
        # self.n_episode = n_episode
        
        #TODO: align observation space and n_action with self.observation_shape, self.action_shape
        self.buffer = ReplayBuffer(buffer_size)
        self.critic = DGN(self.n_agents, self.observation_shape, hidden_dim, self.n_actions).cuda() 
        self.critic_target = DGN(self.n_agents, self.observation_shape, hidden_dim, self.n_actions).cuda()
        
        self.optimizer = optim.Adam(self.critic.parameters(), lr = lr)
        # self.loss = None
        
        # used for training
        self.O = np.ones((batch_size, self.n_agents, self.observation_shape)) #observation
        self.Next_O = np.ones((batch_size, self.n_agents, self.observation_shape)) #next observation
        self.Matrix = np.ones((batch_size, self.n_agents, self.n_agents)) 
        self.Next_Matrix = np.ones((batch_size, self.n_agents, self.n_agents))
        
    
    
    def find_action(self, observation, adjacency, in_eval=False):
        
        # Q values
        q = self.critic(torch.Tensor(np.array([observation])).cuda(), torch.Tensor(adjacency).cuda())[0] 
        
        action = []
        for i in range(self.n_agents):
            if np.random.rand() < self.epsilon and not in_eval:  # random action
                a = np.random.randint(self.n_actions)
            else: # max q value action
                a = q[i].argmax().item() 
            
            action.append(a)
        
        return action
    

    def learn(self, i_episode, n_epoch, log_every):
        
        # is episodes lower than 100, do not do any learning yet
        if i_episode < 100:
            return

        # learning on n_epochs
        for e in range(n_epoch):
            
            # get POMDCH in batches
            batch = self.buffer.getBatch(self.batch_size)
    
            for j in range(self.batch_size):
                sample = batch[j]
                self.O[j] = sample[0]
                self.Next_O[j] = sample[3]
                self.Matrix[j] = sample[4]
                self.Next_Matrix[j] = sample[5]

            q_values = self.critic(torch.Tensor(self.O).cuda(), torch.Tensor(self.Matrix).cuda())
            target_q_values = self.critic_target(torch.Tensor(self.Next_O).cuda(), torch.Tensor(self.Next_Matrix).cuda()).max(dim = 2)[0]
            target_q_values = np.array(target_q_values.cpu().data)
            expected_q = np.array(q_values.cpu().data)
            
            for j in range(self.batch_size):
                sample = batch[j]
                for i in range(self.n_agents):
                    # expected_q[j][i][sample[1][i]] = sample[2][i] + (1-sample[6])*GAMMA*target_q_values[j][i]
                    expected_q[j][i][sample[1][i]] = sample[2][i] + self.GAMMA*target_q_values[j][i]
            
            loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if i_episode % log_every == 0:
            print(f'loss,{i_episode},{loss}')
            
            if self.file:
                with open(self.file, 'a') as f:
                    f.write(f'loss,{i_episode},{loss}\n')
            
        
        if i_episode%5 == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())
        
        

    def get_log_dict(self):
        return {'loss/q_loss': self.loss.item()}
    
    
    def map_action(self, action):
        return list(map(lambda x: x * self.action_step - 1 , action))
        
    
    def run(self,
            max_step: int = 500, #500
            n_episode: int = 400, #800
            i_episode: int = 0,
            n_epoch: int = 25,
            # epsilon: float = 0.9,
            render: bool = False,
            log: bool = True,
            log_every: int = 10,
            log_directory: Optional[str] = None,
            evaluate: bool = True,
            eval_steps: Optional[int] = 500,
            eval_every: Optional[int] = 10,
            
            
            # total_steps: int = 10000,
            # render: bool = False,
            # rescale_rewards: bool = True,
            # reward_range: Optional[Tuple[float, float]] = None,
            # log: bool = True,
            # log_every: int = 1,
            # log_directory: Optional[str] = None,
            # eval_envs: Optional[List[Env]] = None,
            # eval_steps: Optional[int] = 1000,
            # eval_every: Optional[int] = 1000,
            # eval_once: bool = False,  # for non-learning agents,
            # eval_only: bool = False  # will repeat evaluations but skip training
            ):
        
        self.epsilon = self.EPS_START
        self.file = log_directory
        
        while i_episode < n_episode:

            # if i_episode is above 100, increase probability of making action according to q value by 0.0004
            # as training progresses, less random exploration, more action based on learnt q value
            # never let probability of random move drop below 10%
        
            if i_episode > 100: 
                self.epsilon -= 0.0004 
                if self.epsilon < self.EPS_END:
                    self.epsilon = self.EPS_END
            i_episode+=1
            
            
            obs, adj = self._env.reset()
            
            steps = 0
            score = []
            # exploring environment in max_steps
            while steps < max_step:
                steps+=1 
                
                #find actions
                action = self.find_action(obs, adj)
                
                # step through environment with actions
                # next_obs, reward, terminate, _ = self._env.step(action)
                env_action = self.map_action(action)
                next_obs, reward, terminate, _ , next_adj = self._env.step(env_action)
                        
                # next_obs shape -> (1, 100, 29)
                # next_adj shape -> (1, 100, 100)
                # reward -> (100)
                # terminated -> bool
                
                #add observation to buffer 
                self.buffer.add(np.array(obs), action, reward,np.array(next_obs),adj,next_adj)
                # self.learn(obs, action, reward, next_obs, adj, next_adj, steps)
                obs = next_obs
                adj = next_adj
                
                score.append(sum(reward)) # global (total score) (cumulative sum)
            
            
            avg_score = sum(score) / max_step
            # every episode print episodic power output
            if log and i_episode % log_every == 0:
                print(f'train,{i_episode},{avg_score}')
                
                if self.file:
                    with open(self.file, 'a') as f:
                        f.write(f'train,{i_episode},{avg_score}\n')
            
            self.learn(i_episode, n_epoch, log_every)
            
            if render:
                self._env.render()
            
            # evaluation
            if evaluate and i_episode % eval_every == 0:
                self._eval(self._env, eval_steps, i_episode)
    
    
    def _eval(self, env: Env, eval_steps, i_episode):
        
        obs, adj = env.reset()
        steps = 0
        score = []
        
        while steps < eval_steps:
                steps+=1 
                
                #find actions
                action = self.find_action(obs, adj, True)
                
                # step through environment with actions
                # next_obs, reward, terminate, _ = self._env.step(action)
                env_action = self.map_action(action)
                next_obs, reward, terminate, _ , next_adj = self._env.step(env_action)
                
                obs = next_obs
                adj = next_adj
                
                score.append(sum(reward)) # global (total score) (cumulative sum)
        
        
        avg_score = sum(score) / eval_steps
        print(f'eval,{i_episode},{avg_score}')
        
        if self.file:
            with open(self.file, 'a') as f:
                f.write(f'eval,{i_episode},{avg_score}\n')
        
        
        
        
    # def _log_value(self, tag, value, writer, global_step=0):
    #     ...

    # def _log_dict(self, dictionary: dict, writer, global_step=0):
    #     ...
    
    
    
    
    
    
    