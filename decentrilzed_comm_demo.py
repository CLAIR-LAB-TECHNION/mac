from mac.coordination.centralized_coordinator import CentralizedCoordinator
from mac.coordination.decentralized_coordinator import DecentralizedCoordinator
from envs.taxi_wrapper import TaxiWrapper
from mac.agent import Agent
from mac.coordination.decentralized_coordinator_communication import DecentralizedCoordinatorComm

class TaxiRandomAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, step_data):
            return self.action_space.sample()

    def get_observation(self, step_data):
        obs, _, _, _, _ = step_data
        return obs
    
class TaxiRandomCentralAgent(Agent):
    def __init__(self, action_spaces):
        self.action_spaces = action_spaces

    def get_action(self, step_data):
        return {
            agent_id: self.action_spaces[agent_id].sample()
            for agent_id in self.action_spaces
        }

    def get_observation(self, step_data):
        obs, _, _, _, _ = step_data
        return obs
    

from multi_taxi.multi_taxi import multi_taxi_v0
env = multi_taxi_v0.parallel_env(num_taxis=5, num_passengers=5, render_mode='human')


env_wrapper = TaxiWrapper(env)
# central_agent = TaxiRandomCentralAgent({agent: env.action_space(agent) for agent in env.possible_agents})
# coordinator = CentralizedCoordinator(env_wrapper, central_agent)
env.possible_agents
communication_spec = {
    'taxi_0': {'targets': ['taxi_1', 'taxi_2', 'taxi_3', 'taxi_4'], 'shape': [1]*4},
    'taxi_1': {'targets': ['taxi_0', 'taxi_2', 'taxi_3', 'taxi_4'], 'shape': [1]*4},
    'taxi_2': {'targets': ['taxi_1', 'taxi_0', 'taxi_3', 'taxi_4'], 'shape': [1]*4},
    'taxi_3': {'targets': ['taxi_1', 'taxi_2', 'taxi_0', 'taxi_4'], 'shape': [1]*4},
    'taxi_4': {'targets': ['taxi_1', 'taxi_2', 'taxi_3', 'taxi_0'], 'shape': [1]*4},
}
agents_observation_spaces, agents_action_spaces = DecentralizedCoordinatorComm.create_agents_action_obs_space(env, communication_spec)

agents = {agent_id: TaxiRandomAgent(agents_action_spaces[agent_id]) for agent_id in env.possible_agents}

coordinator = DecentralizedCoordinatorComm(env_wrapper, agents, communication_spec)


coordinator.run(100)