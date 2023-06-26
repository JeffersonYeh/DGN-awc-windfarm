from wind_farm_gym.wind_farm_env import WindFarmEnv
from agent.DGN.dgn import DGNAgent
from agent.DGN import FlorisAgent
from agent.DGN import DQNAgent
from typing import Literal


### ENV Setting
layout : Literal['3 Turbines', '16 Grid', 'Amalia', '3x2 Turbines', '9 Grid'] = '3 Turbines'

if layout == '3 Turbines':
    turbine_layout = ([0, 750, 1500], [0, 0, 0])
elif layout == '16 Grid':
    turbine_layout = ([0, 750, 1500, 2250, 0, 750, 1500, 2250, 0, 750, 1500, 2250, 0, 750, 1500, 2250], [0, 0, 0, 0, 750, 750, 750, 750, 1500, 1500, 1500, 1500, 2250, 2250, 2250, 2250])
elif layout == '3x2 Turbines':
    turbine_layout = ([0, 750, 1500, 0, 750, 1500], [0, 0, 0, 750, 750, 750])
elif layout == '9 Grid':
    turbine_layout = ([0, 750, 1500, 0, 750, 1500, 0, 750, 1500], [0, 0, 0, 750, 750, 750, 1500, 1500, 1500])
else:
    raise Exception('Not Implemented')

action_representation : Literal['wind', 'absolute', 'yaw'] ='yaw'

graph_representation: Literal['undirected', 'directed'] = 'undirected'
optimal_distance: float = 10
upstream: bool = True


# run setting
steps = 100
episodes = 10000
lr = 1e-7
batch_size = 128
eps_start = 0.9
eps_end = 0.05


def make_file(type : Literal['dqn', 'dgn', 'cmp'], count):
    
    return f'D:\TU Delft\CSE\Y3\Research Project\Code\MARL-awc-windfarm\windfarm\\agent\DGN\data\\{type}_{count}.csv'



def run_log(_file):
    meta = {'layout' : layout,
            'action' : action_representation,
            'steps' : steps,
            'episodes' : episodes,
            'learning rate' : lr,
            'batch size' : batch_size,
            'epsilon start' : eps_start,
            'epsilon end' : eps_end,
            'graph representation' : graph_representation,
            'up stream' : upstream,
            'optimal distance' : optimal_distance}

    with open(_file, 'a') as f:
        f.write(f'{meta}\n')


def run_dgn(_lr, _batch_size, _file, _graph_representation, _stream):
        
    env = WindFarmEnv(turbine_layout=turbine_layout, action_representation=action_representation)

    agent = DGNAgent('dgn', 
                     env, 
                     lr=_lr, 
                     epsilon_start=eps_start, 
                     epsilon_end=eps_end,
                     batch_size=_batch_size,
                     graph_representation=_graph_representation,
                     stream=_stream) #DGN
    
    agent.run(max_step= steps,
            n_episode=episodes,
            eval_steps= steps,
            log_directory=_file,
            log= True,
            render=False,
            )
    agent.close()
    

def run_floris():
    env = WindFarmEnv(turbine_layout=turbine_layout, action_representation=action_representation)

    agent = FlorisAgent('floris', env) #floris
    agent.run(max_step= steps,)
    agent.close()
    

def run_dqn(_lr, _file, ref = False):
    env = WindFarmEnv(turbine_layout=turbine_layout, action_representation=action_representation)

    agent = DQNAgent('dqn', 
                     env, 
                     reference=ref,
                     epsilon_start=eps_start,
                     epsilon_end=eps_end,
                     lr = _lr,
                     batch_size=batch_size)
    
    agent.run(max_step=steps,
              n_episode=episodes,
              eval_steps=steps,
              log_directory=_file,
              )
    agent.close()
    

def run_baseline():
    
    env = WindFarmEnv(turbine_layout=turbine_layout, action_representation=action_representation)
    
    obs = env.reset()
    action = [0]*env.n_turbines
    obs, reward, _, _  = env.step(action)  # Perform the action
    
    print(sum(reward))
    
    


# RUN
count = 0
file = make_file('dgn', count)
run_log(file)

# run_dqn(lr, file)
run_dgn(lr, batch_size, file, graph_representation, upstream)

