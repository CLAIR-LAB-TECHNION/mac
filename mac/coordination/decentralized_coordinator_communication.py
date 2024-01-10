from .base_coordinator import BaseCoordinator
from mac.utils import get_elements_order
from gymnasium.spaces import Dict
from gymnasium.spaces import Box
import numpy as np
from mac.coordination.communication_medium import CommunicationMedium
from mac.coordination.agent_proxy import AgentProxy

class DecentralizedCoordinatorComm(BaseCoordinator):
    def __init__(self, env, agents, communication_medium: CommunicationMedium, b_random_order=True):
        super().__init__(env, agents)
        self.agent_proxies = {agent_id: AgentProxy(self.agents[agent_id]) for agent_id in self.agents}

        self.communication_medium = communication_medium
        self.b_random_order = b_random_order

    def get_agent_observation_space(self, agent_id):
        return self.agent_proxies[agent_id].get_agent_observation_space()

    def get_agent_action_space(self, agent_id):
        return self.agent_proxies[agent_id].get_agent_action_space()


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
        all_obs, all_rewards, all_terms, all_truncs, all_infos = step_data
        all_env_actions = {}
        for agent_id in agents_order:
            agent_proxy = self.agent_proxies[agent_id]
            obs, rewards, terms, truncs, infos = all_obs[agent_id], all_rewards[agent_id], all_terms[agent_id], all_truncs[agent_id], all_infos[agent_id]
            obs = {'env_obs': obs}
            ### communication object
            agent_input_msgs = self.communication_medium.get_agent_msgs(id=agent_id)
            agent_step_data = obs, rewards, terms, truncs, infos
            action, target_ids, msgs = agent_proxy.get_action(agent_step_data, agent_input_msgs)
            all_env_actions[agent_id] = action
            self.communication_medium.set_agent_msgs(agent_id, target_ids=target_ids, msgs=msgs)
        self.communication_medium.step() # nesccery for updating the past msgs
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

