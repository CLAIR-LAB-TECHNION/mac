from .base_coordinator import BaseCoordinator


class CentralizedCoordinator(BaseCoordinator):
    """
    Coordinator for centralized environments.

    Args:
        env: Environment
            The environment.
        agents: dict
            Dictionary of agents in the environment.
        central_agent: CentralAgent
            The central agent.

    Methods:
        get_initial_data():
            Get initial data for the coordination process.

    """

    def __init__(self, env, central_agent):
        super().__init__(env, central_agent)
        self.central_agent = central_agent

    def get_initial_data(self):
        """
        Get initial data for the coordination process.

        Returns:
            object: Initial data for the coordination process.
        """

        obs, infos = self.env_wrapper.reset()
        rewards = {agent_id: 0 for agent_id in obs}
        terms = {agent_id: False for agent_id in obs}
        truncs = {agent_id: False for agent_id in obs}

        return obs, rewards, terms, truncs, infos

    def run_step(self, step_data):
        joint_action = self.central_agent.get_action(step_data)
        step_data = self.env_wrapper.step(joint_action)
        return step_data, joint_action

    def get_ids(self):
        return self.env_wrapper.env.agents

