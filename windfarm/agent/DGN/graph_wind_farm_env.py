
# import sys
# sys.path.append("...") # Adds higher directory to python modules path.

from typing import Dict, List, Literal, Optional, Tuple, Union
from floris.tools.floris_interface import FlorisInterface

# from wind_farm_gym.wind_process import WindProcess
# from wind_farm_gym.wind_farm_env import WindFarmEnv

from wind_farm_gym.wind_farm_env import WindFarmEnv
from wind_farm_gym.wind_process import WindProcess

# from ..windfarm.wind_farm_gym.wind_process import WindProcess
# from ..windfarm.wind_farm_gym.wind_farm_env import WindFarmEnv
import networkx as nx
from networkx.linalg.graphmatrix import adjacency_matrix

import math
import numpy as np

class GraphWindFarmEnv(WindFarmEnv):
    
    def __init__(
            self,
            seed: Optional[int] = None,
            floris: Optional[Union[FlorisInterface, str]] = None,
            turbine_layout: Optional[Union[Dict[str, List[float]], Tuple[List[float], List[float]]]] = None,
            mast_layout: Optional[Union[Dict[str, List[float]], Tuple[List[float], List[float]]]] = None,
            time_delta: float = 1.0,
            max_angular_velocity: float = 1.0,
            desired_yaw_boundaries: Tuple[float, float] = (-45.0, 45.0),
            wind_process: Optional[WindProcess] = None,
            observe_yaws: bool = True,
            farm_observations: Optional[Union[str, Tuple[str, ...]]] = None,
            mast_observations: Optional[Union[str, Tuple[str, ...]]] = ('wind_speed', 'wind_direction'),
            lidar_observations: Optional[Union[str, Tuple[str, ...]]] = ('wind_speed', 'wind_direction'),
            lidar_turbines: Optional[Union[str, int, Tuple[int, ...]]] = 'all',
            lidar_range: Optional[float] = 10.0,
            observation_boundaries: Optional[Dict[str, Tuple[float, float]]] = None,
            normalize_observations: bool = True,
            random_reset=False,
            action_representation: Literal['wind', 'absolute', 'yaw'] = 'wind',
            perturbed_observations: Optional[Union[str, int, Tuple[int, ...]]] = None,
            perturbation_scale: float = 0.05,
            graph_representation: Literal['undirected', 'directed'] = 'undirected',
            optimal_distance: float = 10,
            stream: bool = True
    ):
        super().__init__(seed, 
                         floris, 
                         turbine_layout, 
                         mast_layout, 
                         time_delta, 
                         max_angular_velocity, 
                         desired_yaw_boundaries, 
                         wind_process, 
                         observe_yaws, 
                         farm_observations, 
                         mast_observations, 
                         lidar_observations, 
                         lidar_turbines, 
                         lidar_range, 
                         observation_boundaries, 
                         normalize_observations, 
                         random_reset, 
                         action_representation, 
                         perturbed_observations, 
                         perturbation_scale)
        
        self.graph_representation = graph_representation
        self.optimal_distance = optimal_distance
        
        if self.graph_representation == 'directed':
            self.stream = stream
        
        self._create_graph()
        
        
        
        
    
    
    @classmethod
    def from_WindFarmEnv(cls, wind_farm_env: WindFarmEnv, 
                         graph_representation: Literal['undirected', 'directed'] = 'undirected',
                         optimal_distsance: float = 10,
                         stream: bool = True):
        
        # Create new b_obj
        graph_wind_farm_env_obj = cls()
        # Copy all values of A to B
        # It does not have any problem since they have common template
        for key, value in wind_farm_env.__dict__.items():
            graph_wind_farm_env_obj.__dict__[key] = value
            
        graph_wind_farm_env_obj.graph_representation = graph_representation
        graph_wind_farm_env_obj.optimal_distance = optimal_distsance
        if graph_wind_farm_env_obj.graph_representation == 'directed':
            graph_wind_farm_env_obj.stream = stream
            
        graph_wind_farm_env_obj._create_graph()
        
        return graph_wind_farm_env_obj
    
    
    def _get_adjacency_matrix(self):
        return adjacency_matrix(self.graph).toarray()[np.newaxis]
    
    def _from_adjacency_matrix(self, adj):
        self.graph = nx.from_numpy_array(adj, create_using=type(self.graph))
        
    def _create_graph(self):
        
        if self.graph_representation == 'undirected':
            self._create_undirected_graph()
        elif self.graph_representation == 'directed':
            self._create_directed_graph(self.stream)
        else:
            print(self.graph_representation)
            raise Exception("Graph representation type undefined")
        
    def _create_undirected_graph(self):
        self.graph = nx.Graph()
        
        #number of turbines
        self.graph.add_nodes_from(range(self.n_turbines))
        
        #create self loop
        self.graph.add_edges_from([(i, i) for i in range(self.n_turbines)])        
        
        #add edge if turbines affect each other
        # optimal streamwise spacing 10- 15 D (turbine diameter) [m]
        optimal_spacing = 10 * self._farm.turbines[0].rotor_diameter
        # print(optimal_spacing)
        
        # naive compute graph based on optimal streamwise spacing
        # Edge is within spacing
        for i, a in enumerate(zip(*self.turbine_layout)):
            for j, b in enumerate(zip(*self.turbine_layout)):
                if j < i: continue
                
                if not self.turbine_optimally_spaced(a, b, optimal_spacing):
                    self.graph.add_edge(i, j)
        
    def _create_directed_graph(self, to_upstream=True):
        self.graph = nx.DiGraph()
        
        #number of turbines
        self.graph.add_nodes_from(range(self.n_turbines))
        
        #create self loop
        self.graph.add_edges_from([(i, i) for i in range(self.n_turbines)])        
        
        #add edge if turbines affect each other
        # optimal streamwise spacing 10- 15 D (turbine diameter) [m]
        optimal_spacing = 10 * self._farm.turbines[0].rotor_diameter
        # print(optimal_spacing)
        
        # naive compute graph based on optimal streamwise spacing
        # Edge is within spacing
        for i, a in enumerate(zip(*self.turbine_layout)):
            for j, b in enumerate(zip(*self.turbine_layout)):
                if j < i: continue
                
                if not self.turbine_optimally_spaced(a, b, optimal_spacing):
                    if to_upstream:
                        if a[0] < b[0]:
                            self.graph.add_edge(j, i)
                        elif a[0] > b[0]:
                            self.graph.add_edge(i, j)

                    else: #downstream
                        if a[0] < b[0]:
                            self.graph.add_edge(i, j)
                        elif a[0] > b[0]:
                            self.graph.add_edge(j, i)
                        
    
    @staticmethod
    def turbine_optimally_spaced(a: tuple, b: tuple, threshold: float) -> bool:
        """
        tuple (x, y)
        
        Returns
        -------
        bool
            True is optimally spaced, False  otherwise
        """
        # euclidean distance
        dist = math.dist(a, b)
        
        return dist >= threshold
        
        
    def update_graph(self, state):
        """
        Update the graph based on state
        """
        ...
        
        
    def reset(self):
        state = super().reset()
        
        self.update_graph(state)
        
        return self.split_state(state), self._get_adjacency_matrix()
    
    
    def step(self, action):
        """
        Returns state, reward, done, info, adjacency matrix
        """
        state, reward, done, info = super().step(action)
        
        self.update_graph(state)
    
        return self.split_state(state), reward, done, info, self._get_adjacency_matrix()
    
    def split_state(self, state):
        state_elements = list(map(lambda x: x.get('name'), self.observed_variables))
            
        return np.stack([np.take(state, [j for j in range(len(state_elements)) if f'turbine_{i}_' in state_elements[j]]) for i in range(self.n_turbines)])
