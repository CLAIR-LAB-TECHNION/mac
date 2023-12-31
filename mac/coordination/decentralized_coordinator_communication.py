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


class CommunicationMedium:
    def __init__(self, type: str='tcp') -> None:
        self.incoming_msgs = {}
        self.past_incoming_msgs = {}

    def set_agent_msgs(self, src_id: str, target_ids: [list, np.array], msgs: [list, np.array]): # should we include target_id?
        """set the messages buffer for a given agent id and target id"""
        if target_id not in self.incoming_msgs:
            self.incoming_msgs[target_id] = {}
        for target_id, msg in zip(target_ids, msgs):
            self.incoming_msgs[target_id] = {src_id: msg}

    def get_agent_msgs(self, id):
        """gets the messages buffer for a given agent id and target id"""
        return self.past_incoming_msgs[id]
    
    def step(self, ):
        """step the communication medium"""
        self.past_incoming_msgs = self.incoming_msgs
        self.incoming_msgs = {}

class AgentProxy:
    def __init__(self, agent) -> None:
        self.agent = agent

    def get_agent_action_space(self, ):
        """
        Take into account the communication action space in addition to env action space
        """
        # depends on the communication protocol

    def get_agent_observation_space(self, ):
        """
        Take into account the communication space in addition to env observation space
        """
        # depends on the communication protocol

    def sensor_func(self, step_data, communiation_msgs):
        """
        operates uppon a step data and communication messages and returns a new step_data
        """
        # do some logic here
        return step_data, communiation_msgs

    def comm_protocol(self):
        """
        it should be a function or a class?
        """
        pass

    def get_action(self, step_data, communiation_msgs):
        """
        returns the action for the agent and the communication messages
        """
        some_res  = self.sensor_func(step_data, communiation_msgs) # apply sensor function over step data? or msg only?
        action = self.agent.get_action(step_data) # plug the res as an input to agent?
        target_ids, msgs = self.comm_protocol(action) # apply the communication protocol over the action, and extract the msgs for each target (can be broadcast also)
        env_action = action['env_action']
        return env_action, target_ids, msgs

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

