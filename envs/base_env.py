import logging
import gymnasium as gym
from gym.spaces import Dict as GymDict, Discrete, Box


class BaseEnv(gym.Env):
    """
    Base class for custom environments compatible with RLlib.

    This class provides a common structure for creating RL environments. It extends the
    `gym.Env` class and includes methods that must be implemented by derived
    environments.

    Args:
        env: The underlying environment.

    Attributes:
        env (gym.Env): The underlying environment.
        logger (logging.Logger): A logger for environment-specific messages.
        action_space (gym.Space): The action space of the environment.
        observation_space (gym.Space): The observation space of the environment.
        agents (list): List of possible agents in the environment.
        num_agents (int): The number of agents in the environment.
    """

    def __init__(self, env):
        """
        Initialize the environment.

        Args:
            env (gym.Env): The underlying environment.
        """
        self.env = env
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing environment...")
        self._init_env()
        self.logger.info("Environment initialized.")
        self.action_space = Discrete(self.env.action_spaces.popitem()[1].n)
        self.observation_space = GymDict({"obs": Box(
            low=-100.0,
            high=100.0,
            shape=(self.env.observation_spaces.popitem()[1].shape[0],),
            dtype=self.env.observation_spaces.popitem()[1].dtype)})
        self.agents = self.env.possible_agents
        self.num_agents = len(self.agents)

    def _init_env(self):
        """
        Initialize the environment.

        This method should be implemented by derived classes to perform any
        environment-specific initialization.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the environment.

        Returns:
            object: Initial observation.
        """
        raise NotImplementedError

    def step(self, action):
        """
        Perform an action in the environment.

        Args:
            action (object): Action to perform.

        Returns:
            tuple: A tuple containing:
                - object: Observation
                - float: Reward
                - bool: Done flag
                - dict: Additional information
        """
        raise NotImplementedError

    def render(self, mode="human"):
        """
        Render the environment.

        Args:
            mode (str, optional): Rendering mode. Defaults to "human".

        Raises:
            NotImplementedError: This method should be implemented for rendering.

        Note:
            This method is not necessary for RL training but can be used for
            visualization.

        """
        raise NotImplementedError

    def close(self):
        """
        Close the environment.

        This method should release any resources used by the environment.

        Raises:
            NotImplementedError: This method should be implemented for cleanup.
        """
        raise NotImplementedError

    def get_env_info(self):
        """
        Get information about the environment.

        Returns:
            dict: A dictionary containing environment information.
                - "space_obs" (gym.Space): Observation space
                - "space_act" (gym.Space): Action space
                - "num_agents" (int): Number of agents
        """
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
        }
        return env_info

    def transform_action_dict_to_env_format(self, actions) -> dict:
        """
        Transform the action dictionary to the format used by the environment.

        Args:
            actions (dict): Dictionary containing actions for each agent.

        Returns:
            object: The transformed action.
        """
        raise NotImplementedError

