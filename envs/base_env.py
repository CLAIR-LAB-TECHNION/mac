import gymnasium as gym
from gym.spaces import Dict as GymDict, Discrete, Box
import logging


class BaseEnv(gym.Env):
    """
    Base class for all environments.
    """

    def __init__(self, env):
        """
        Initialize the environment.
        :param env: .
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
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the environment.
        :return: (object) Initial observation.
        """
        raise NotImplementedError

    def step(self, action):
        """
        Perform an action in the environment.
        :param action: (object) Action to perform.
        :return: (object, float, bool, dict) Observation, reward, done, info.
        """
        raise NotImplementedError

    def render(self, mode="human"):
        """
        Render the environment.
        :param mode: (str) Rendering mode.
        """
        raise NotImplementedError

    def close(self):
        """
        Close the environment.
        """
        raise NotImplementedError

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
        }
        return env_info

