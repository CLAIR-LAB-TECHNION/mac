from abc import ABC, abstractmethod


class BaseCoordinator(ABC):
    """
    Abstract base class for coordinators.

    Args:
        env_wrapper: EnvironmentWrapper
            Wrapper for the environment.
        agents: dict
            Dictionary of agents in the environment.

    Attributes:
        env_wrapper: EnvironmentWrapper
            Wrapper for the environment.
        agents: dict
            Dictionary of agents in the environment.
        evalulation_log: None
            Placeholder for evaluation logs.

    Methods:
        run(iteration_limit, b_log=False, b_train=False, b_evaluate=False, step_evaluation_func=None, agg_evaluation_func=None):
            Perform a run of the coordination process.
        run_step(step_data):
            Perform a single iteration of the coordination process.
        perform_joint_action(joint_action):
            Activate the joint action in the environment.
        evaluate_step(evaluation_func, step_data, joint_action):
            Evaluate the step's performance.
        evaluate_agg(evaluation_func, evaluation_data):
            Evaluate the aggregated performance.
        perform_training_step(step_data, joint_action):
            Perform training step for agents.
        get_joint_action(step_data):
            Get the joint action for the current step.
        log_step(step_data):
            Log information about the step.
        init_log_data():
            Initialize log data.
        get_ids():
            Get agent IDs.
        is_done(step_data):
            Check if the coordination process is done.
    """

    def __init__(self, env_wrapper, agents):
        self.env_wrapper = env_wrapper
        self.agents = agents
        self.evalulation_log = None

    def run(self, iteration_limit):
        """
        Perform a run of the coordination process.

        Args:
            iteration_limit: int
                Maximum number of iterations to run.
        Returns:
            dict or None: Aggregated evaluation data if b_evaluate is True, else None.
        """
        step_data = self.get_initial_data()

        iteration_counter = 0
        done = False
        while not done and iteration_counter < iteration_limit:

            iteration_counter += 1
            [step_data, joint_action] = self.run_step(step_data)
            done = self.is_done()

    @abstractmethod
    def run_step(self, step_data):
        """
        Perform a single iteration of the coordination process.

        Args:
            step_data: object
                Data for the current step.

        Returns:
            list: Updated step_data and joint_action.
        """
        raise NotImplementedError

    def perform_joint_action(self, joint_action):
        """
        Activate the joint action in the environment.

        Args:
            joint_action: dict
                Joint action to be performed by agents.

        Returns:
            object: Updated step data.
        """
        step_data = self.env_wrapper.step(joint_action)
        return step_data

    @abstractmethod
    def get_initial_data(self):
        """
        Get initial data for the coordination process.

        Returns:
            object: Initial data for the coordination process.
        """
        raise NotImplementedError

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
            step_data: object
                Data for the current step.

        Returns:
            bool: True if done, False otherwise.
        """
        return self.env_wrapper.is_done(step_data)






