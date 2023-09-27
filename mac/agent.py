from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Abstract base class for reinforcement learning agents.

    This class defines the interface for reinforcement learning agents. Subclasses
    must implement the `get_action` and `get_observation` methods to interact with
    the environment.

    Attributes:
        None

    Methods:
        get_action(step_data):
            Get an action to perform in the environment.

        get_observation(step_data):
            Get the observation from the environment.

        perform_training_step(action, step_data):
            Perform a training step (optional).

    """

    @abstractmethod
    def get_action(self, step_data):
        """
        Get an action to perform in the environment.

        Args:
            step_data (object): Data required for selecting an action.

        Returns:
            object: The selected action to be performed.
        """
        pass

    @abstractmethod
    def get_observation(self, step_data):
        """
        Get the observation from the environment.

        Args:
            step_data (object): Data required for obtaining an observation.

        Returns:
            object: The observation received from the environment.
        """
        pass

    def perform_training_step(self, action, step_data):
        """
        Perform a training step (optional).

        This method can be implemented by subclasses if training steps are
        required for the agent.

        Args:
            action (object): The action taken in the environment.
            step_data (object): Data required for training.

        Returns:
            None
        """
        pass
