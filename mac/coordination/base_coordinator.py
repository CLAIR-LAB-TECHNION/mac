from abc import ABC, abstractmethod

class BaseCoordinator(ABC):
    """
    Abstract base class for coordinators in a multi-agent environment.

    Args:
        env_wrapper (EnvironmentWrapper): Wrapper for the environment.
        agents (dict): Dictionary of agents in the environment.

    Attributes:
        env_wrapper (EnvironmentWrapper): Wrapper for the environment.
        agents (dict): Dictionary of agents in the environment.
        evaluation_log (None): Placeholder for evaluation logs.

    Methods:
        run(iteration_limit):
            Perform a run of the coordination process.
        run_step(step_data):
            Perform a single iteration of the coordination process.
        perform_joint_action(joint_action):
            Activate the joint action in the environment.
        get_initial_data():
            Get initial data for the coordination process.
        get_ids():
            Get agent IDs.
        is_done(step_data):
            Check if the coordination process is done.
    """

    def __init__(self, env_wrapper, agents):
        """
        Initialize the BaseCoordinator.

        Args:
            env_wrapper (EnvironmentWrapper): Wrapper for the environment.
            agents (dict): Dictionary of agents in the environment.
        """
        self.env_wrapper = env_wrapper
        self.agents = agents
        self.evaluation_log = None

    def run(self, iteration_limit):
        """
        Perform a run of the coordination process.

        Args:
            iteration_limit (int): Maximum number of iterations to run.
        """
        step_data = self.get_initial_data()

        iteration_counter = 0
        done = False
        while not done and iteration_counter < iteration_limit:

            iteration_counter += 1
            [step_data, joint_action] = self.run_step(step_data)
            done = self.is_done(step_data)

    @abstractmethod
    def run_step(self, step_data):
        """
        Perform a single iteration of the coordination process.

        Args:
            step_data (object): Data for the current step.

        Returns:
            list: Updated step_data and joint_action.
        """
        raise NotImplementedError

    def perform_joint_action(self, joint_action):
        """
        Activate the joint action in the environment.

        Args:
            joint_action (dict): Joint action to be performed by agents.

        Returns:
            object: Updated step data.
        """
        step_data = self.env_wrapper.step(joint_action)
        return step_data

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

    @abstractmethod
    def get_ids(self):
        """
        Get agent IDs.

        Returns:
            list: List of agent IDs.
        """
        raise NotImplementedError

    def is_done(self, step_data):
        """
        Check if the coordination process is done.

        Args:
            step_data (object): Data for the current step.

        Returns:
            bool: True if done, False otherwise.
        """
        return self.env_wrapper.is_done(step_data)
