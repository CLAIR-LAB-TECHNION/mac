from .base_coordinator import BaseCoordinator
from mac.utils import get_elements_order


class DecentralizedCoordinator(BaseCoordinator):
    def __init__(self, env, agents, b_random_order=True):
        super().__init__(env, agents)
        self.b_random_order = b_random_order

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
        for agent_id in agents_order:
            agent = self.agents[agent_id]
            agent_step_data = agent.get_observation(self.env_wrapper.get_agent_step_data(step_data, agent_id))
            action = agent.get_action(agent_step_data)
            actions[agent_id] = action

        return self.env_wrapper.transform_action_dict_to_env_format(actions)

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

