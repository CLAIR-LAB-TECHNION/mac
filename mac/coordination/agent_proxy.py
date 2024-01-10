class AgentProxy:
    def __init__(self, agent) -> None:
        self.agent = agent

    def get_agent_action_space(self, ):
        """
        Take into account the communication action space in addition to env action space
        """
        # depends on the communication protocol

    def get_agent_observation_space(self, ):
        """
        Take into account the communication space in addition to env observation space
        """
        # depends on the communication protocol

    def sensor_func(self, step_data, communiation_msgs):
        """
        operates uppon a step data and communication messages and returns a new step_data
        """
        # do some logic here
        return step_data, communiation_msgs

    def comm_protocol(self):
        """
        it should be a function or a class?
        """
        pass

    def get_action(self, step_data, communiation_msgs):
        """
        returns the action for the agent and the communication messages
        """
        some_res  = self.sensor_func(step_data, communiation_msgs) # apply sensor function over step data? or msg only?
        action = self.agent.get_action(step_data) # plug the res as an input to agent?
        target_ids, msgs = self.comm_protocol(action) # apply the communication protocol over the action, and extract the msgs for each target (can be broadcast also)
        env_action = action['env_action']
        return env_action, target_ids, msgs