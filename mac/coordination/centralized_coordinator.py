from .base_coordinator import BaseCoordinator


class CentralizedCoordinator(BaseCoordinator):
    """
    Coordinator for centralized environments.

    Args:
        env (Environment): The environment.
        central_agent (CentralAgent): The central agent.

    Attributes:
        central_agent (CentralAgent): The central agent associated with the coordinator.

    Methods:
        get_initial_data():
            Get initial data for the coordination process.
        run_step(step_data):
            Run a step of the coordination process.
        get_ids():
            Get the agent IDs in the environment.

    """

    def __init__(self, env, central_agent):
        """
        Initialize the CentralizedCoordinator.

        Args:
            env (Environment): The environment.
            central_agent (CentralAgent): The central agent.
        """
        super().__init__(env, central_agent)
        self.central_agent = central_agent

    def get_initial_data(self):
        """
        Get initial data for the coordination process.

        Returns:
            tuple: A tuple containing:
                - dict: Initial observations for agents.
                - dict: Initial rewards for agents.
                - dict: Initial termination flags for agents.
                - dict: Initial truncation flags for agents.
                - dict: Additional information for agents.
        """
        obs, infos = self.env_wrapper.reset()
        rewards = {agent_id: 0 for agent_id in obs}
        terms = {agent_id: False for agent_id in obs}
        truncs = {agent_id: False for agent_id in obs}

        return obs, rewards, terms, truncs, infos

    def run_step(self, step_data):
        """
        Run a step of the coordination process.

        Args:
            step_data (object): Data required for the coordination step.

        Returns:
            tuple: A tuple containing:
                - object: Step data after the coordination step.
                - object: Joint action taken by the central agent.
        """
        joint_action = self.central_agent.get_action(step_data)
        step_data = self.env_wrapper.step(joint_action)
        return step_data, joint_action

    def get_ids(self):
        """
        Get the agent IDs in the environment.

        Returns:
            list: List of agent IDs.
        """
        return self.env_wrapper.env.agents
