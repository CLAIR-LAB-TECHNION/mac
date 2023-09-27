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
