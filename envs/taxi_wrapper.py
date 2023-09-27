from .base_env import BaseEnv
import time

SLEEP_TIME = 0.2


class TaxiWrapper(BaseEnv):
    """
    Wrapper for Gymnasium environments.
    """

    def _init_env(self):
        """
        Initialize the environment.
        """
        pass

    def reset(self):
        """
        Reset the environment.
        :return: (object) Initial observation.
        """
        out = self.env.reset(return_info=True)
        time.sleep(SLEEP_TIME)
        self.env.render()
        return out

    def step(self, action):
        """
        Perform an action in the environment.
        :param action: (object) Action to perform.
        :return: (object, float, bool, dict) Observation, reward, done, info.
        """
        out = self.env.step(action)
        time.sleep(SLEEP_TIME)
        self.env.render()
        return out

    def render(self, mode="human"):
        """
        Render the environment.
        :param mode: (str) Rendering mode.
        """
        self.env.render(mode)

    def close(self):
        """
        Close the environment.
        """
        self.env.close()

    def is_done(self, step_data):
        return self.env.env_done()

