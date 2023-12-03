from .base_coordinator import BaseCoordinator
from mac.utils import get_elements_order
from gymnasium.spaces import Dict
from gymnasium.spaces import Box
import numpy as np

class CommunicationSpec:
    def __init__(self, ) -> None:
        pass

    def get_communications_shape(self, ) -> int:
        pass

    def get_communication_size(self, ) -> int:
        pass

class DecentralizedCoordinatorComm(BaseCoordinator):
    def __init__(self, env, agents, communication_spec, b_random_order=True):
        super().__init__(env, agents)
        self.communication_spec = communication_spec
        self.b_random_order = b_random_order
        # env.env because its a warpped env
        self.agents_observation_spaces, self.agents_action_spaces = DecentralizedCoordinatorComm.create_agents_action_obs_space(env.env, communication_spec)
        self.last_msgs_send = {agent_id: {target_id: np.zeros_like(shape) for target_id,shape in zip(spec['targets'], spec['shape'])} for agent_id, spec in communication_spec.items()}


    def create_agents_action_obs_space(env, communication_spec):
        """
        Build agents.

        Args:
            agents (dict): Dictionary of agents in the environment.
            communication_spec (dict): Dictionary of communication specifications.
        """

        env.possible_agents = env.possible_agents
        agents_action_spaces = {agent_id: {'env_action': env.action_space(agent_id)} for agent_id in env.possible_agents}
        agents_observation_spaces = {agent_id: {'env_obs': env.observation_spaces[agent_id]} for agent_id in env.possible_agents}
        
        for agent_id, spec in communication_spec.items():
            for sendner_id, s in communication_spec.items():
                for i, target in enumerate(s['targets']):
                    if agent_id == target:
                        agents_observation_spaces[agent_id] = Dict({f'from_{sendner_id}': Box(low=0, high=1, shape=(s['shape'][i],)), **agents_observation_spaces[agent_id]})
                
            agent_comm_action_space = {f'to_{target_id}': Box(low=0, high=1, shape=(spec['shape'][j],)) for j, target_id in enumerate(spec['targets'])}
            
            agents_action_spaces[agent_id] = Dict({**agent_comm_action_space, **agents_action_spaces[agent_id]})

        return agents_observation_spaces, agents_action_spaces


    def get_ids(self):
        """
        Get agent IDs.

        Returns:
            list: List of agent IDs.
        """
        return list(self.agents.keys())


    def _get_joint_action(self, step_data):
        """
        Compute the joint action.

        Args:
            step_data: object
                Data for the current step.

        Returns:
            dict: Joint action to be performed by each agent.
        """
        agents_ids = self.get_ids()
        agents_order = get_elements_order(self.b_random_order, agents_ids)

        # Returning a dictionary of actions to be performed by each agent
        actions = {}
        all_obs, all_rewards, all_terms, all_truncs, all_infos = step_data
        all_env_actions = {}
        for agent_id in agents_order:
            agent = self.agents[agent_id]
            obs, rewards, terms, truncs, infos = all_obs[agent_id], all_rewards[agent_id], all_terms[agent_id], all_truncs[agent_id], all_infos[agent_id]
            obs = {'env_obs': obs}
            for sender_id, spec in self.communication_spec.items():
                for i, target_id in enumerate(spec['targets']):
                    if target_id == agent_id:
                        obs[f'from_{sender_id}'] = self.last_msgs_send[sender_id][agent_id]
            agent_step_data = obs, rewards, terms, truncs, infos
            actions = agent.get_action(agent_step_data)
            env_action = actions['env_action']
            all_env_actions[agent_id] = env_action
            del actions['env_action']
            actions = {k.replace('to_', ''):v for k,v in actions.items()} # just to remove the to_ prefix
            self.last_msgs_send[agent_id] = actions
        return all_env_actions


    def run_step(self, step_data):
        """
        Perform a single iteration of the coordination process.

        Args:
            step_data (object): Data for the current step.

        Returns:
            list: Updated step_data and joint_action.
        """
        joint_action = self._get_joint_action(step_data)
        step_data = self.env_wrapper.step(joint_action)
        return step_data, joint_action

