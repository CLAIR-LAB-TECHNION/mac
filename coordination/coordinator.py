from abc import ABC, abstractmethod
from ..utils import get_elements_order


class Coordinator(ABC):
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

    def __init__(self, env_wrapper, agents: dict):
        self.env_wrapper = env_wrapper
        self.agents = agents
        self.evalulation_log = None

    def run(self, iteration_limit, b_log=False, b_train=False, b_evaluate=False, step_evaluation_func=None, agg_evaluation_func=None):
        """
        Perform a run of the coordination process.

        Args:
            iteration_limit: int
                Maximum number of iterations to run.
            b_log: bool, optional
                Flag to log data during the run.
            b_train: bool, optional
                Flag to enable agent training during the run.
            b_evaluate: bool, optional
                Flag to evaluate performance during the run.
            step_evaluation_func: callable, optional
                Function to evaluate step performance.
            agg_evaluation_func: callable, optional
                Function to evaluate aggregated performance.

        Returns:
            dict or None: Aggregated evaluation data if b_evaluate is True, else None.
        """
        step_data = self.get_initial_data()

        if b_log:
            self.log_step(step_data)
        if b_evaluate:
            evaluation_data = {}

        iteration_counter = 0
        done = False
        while not done and iteration_counter < iteration_limit:

            iteration_counter += 1
            if done:
                break

            [step_data, joint_action] = self.run_step(step_data)

            if b_log:
                self.log_step(step_data)

            if step_evaluation_func:
                evaluation_data[iteration_counter] = self.evaluate_step(step_evaluation_func, step_data, joint_action)

            if b_train:
                self.perform_training_step(step_data, joint_action)

            done = self.is_done(step_data)

        if b_evaluate:
            return self.evaluate_agg(agg_evaluation_func, evaluation_data)
        else:
            return None

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
        pass

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

    def evaluate_step(self, evaluation_func, step_data, joint_action):
        """
        Evaluate the step's performance.

        Args:
            evaluation_func: callable
                Function to evaluate step performance.
            step_data: object
                Data for the current step.
            joint_action: dict
                Joint action performed by agents.

        Returns:
            dict: Evaluation results.
        """
        return evaluation_func(self.env_wrapper, step_data, joint_action)

    def evaluate_agg(self, evaluation_func, evaluation_data):
        """
        Evaluate the aggregated performance.

        Args:
            evaluation_func: callable
                Function to evaluate aggregated performance.
            evaluation_data: dict
                Dictionary of evaluation data.

        Returns:
            dict: Aggregated evaluation results.
        """
        return evaluation_func(evaluation_data)

    @abstractmethod
    def perform_training_step(self, step_data, joint_action):
        """
        Perform training step for agents.

        Args:
            step_data: object
                Data for the current step.
            joint_action: dict
                Joint action performed by agents.
        """
        pass

    @abstractmethod
    def get_joint_action(self, step_data):
        """
        Get the joint action for the current step.

        Args:
            step_data: object
                Data for the current step.
        """
        pass

    @abstractmethod
    def get_initial_data(self):
        """
        Get initial data for the coordination process.

        Returns:
            object: Initial data for the coordination process.
        """
        pass

    @abstractmethod
    def log_step(self, step_data):
        """
        Log information about the step.

        Args:
            step_data: object
                Data for the current step.
        """
        pass

    @abstractmethod
    def init_log_data(self):
        """
        Initialize log data.
        """
        pass

    @abstractmethod
    def get_ids(self):
        """
        Get agent IDs.

        Returns:
            list: List of agent IDs.
        """
        pass

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


class DecentralizedCoordinator(Coordinator):
    """
    Coordinator for decentralized environments.

    Args:
        env: Environment
            The environment.
        agents: dict
            Dictionary of agents in the environment.
        b_random_order: bool, optional
            Flag to specify whether to randomize agent order.

    Methods:
        get_initial_data():
            Get initial data for the coordination process.
        get_joint_action(step_data):
            Get the joint action for the current step.
        perform_training_step(step_data, joint_action):
            Perform training step for agents.
    """

    def __init__(self, env, agents, b_random_order=True):
        super().__init__(env, agents)
        self.b_random_order = b_random_order

    def get_initial_data(self):
        """
        Get initial data for the coordination process.

        Returns:
            object: Initial data for the coordination process.
        """
        # You can implement this method based on your specific requirements.
        pass

    def get_joint_action(self, step_data):
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

    def perform_training_step(self, step_data, joint_action):
        """
        Perform training step for agents.

        Args:
            step_data: object
                Data for the current step.
            joint_action: dict
                Joint action performed by agents.
        """
        joint_action = self.env_wrapper.transform_action_env_format_to_dict(joint_action)
        for agent_id in self.get_ids():
            agent_action = joint_action[agent_id]
            agent_step_data = self.agents[agent_id].get_observation(self.env_wrapper.get_agent_step_data(step_data, agent_id))
            self.agents[agent_id].perform_training_step(agent_action, agent_step_data)


class CentralizedCoordinator(Coordinator):
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
        get_joint_action(step_data):
            Get the joint action for the current step.
        perform_training_step(step_data, joint_action):
            Perform training step for agents.
    """

    def __init__(self, env, agents, central_agent):
        super().__init__(env, agents)
        self.central_agent = central_agent

    def get_joint_action(self, step_data):
        """
        Compute the joint action.

        Args:
            step_data: object
                Data for the current step.

        Returns:
            dict: Joint action to be performed by each agent.
        """
        # Get the agent's observation
        step_data = self.central_agent.get_observation(step_data)

        # Compute the joint action
        joint_action = self.central_agent.get_action(step_data)

        # Transform the joint action to the environment format
        joint_action = self.env_wrapper.transform_action_dict_to_env_format(joint_action)

        return joint_action

    def perform_training_step(self, step_data, joint_action):
        """
        Perform training step for agents.

        Args:
            step_data: object
                Data for the current step.
            joint_action: dict
                Joint action performed by agents.
        """
        self.central_agent.perform_training_step(step_data, joint_action)


class DecentralizedWithComCoordinator(DecentralizedCoordinator):
    """
    Coordinator for decentralized environments with communication signals.

    Args:
        env: Environment
            The environment.
        agents: dict
            Dictionary of agents in the environment.
        b_random_order: bool, optional
            Flag to specify whether to randomize agent order.

    Methods:
        get_initial_data():
            Get initial data for the coordination process.
        get_joint_action(step_data):
            Get the joint action for the current step.
        get_shared_com_signal(step_data):
            Get the shared communication signal for agents.
    """

    def __init__(self, env, agents, b_random_order=True):
        super().__init__(env, agents, b_random_order)

    def get_joint_action(self, step_data):
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

        # Get the shared communication signal
        signal = self.get_shared_com_signal(step_data)

        # Returning a dictionary of actions to be performed by each agent
        actions = {}
        for agent_id in agents_order:
            agent = self.agents[agent_id]
            agent_step_data = agent.get_observation(self.env_wrapper.get_agent_step_data(step_data, agent_id))

            # Combine agent's observation with the communication signal
            agent_step_data = [agent_step_data, signal]
            action = agent.get_action(agent_step_data)
            actions[agent_id] = action

        return self.env_wrapper.transform_action_dict_to_env_format(actions)

    @abstractmethod
    def get_shared_com_signal(self, step_data):
        """
        Get the shared communication signal for agents.

        Args:
            step_data: object
                Data for the current step.

        Returns:
            object: Communication signal shared among agents.
        """
        pass
