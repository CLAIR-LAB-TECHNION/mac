
import numpy as np
class CommunicationMedium:
    def __init__(self) -> None:
        self.incoming_msgs = {}
        self.past_incoming_msgs = {}

    def set_agent_msgs(self, src_id: str, target_ids: [list, np.array], msgs: [list, np.array]): # should we include target_id?
        """set the messages buffer for a given agent id and target id"""
        if target_id not in self.incoming_msgs:
            self.incoming_msgs[target_id] = {}
        for target_id, msg in zip(target_ids, msgs):
            self.incoming_msgs[target_id] = {src_id: msg}

    def get_agent_msgs(self, id):
        """gets the messages buffer for a given agent id and target id"""
        return self.past_incoming_msgs[id]
    
    def step(self, ):
        """step the communication medium"""
        self.past_incoming_msgs = self.incoming_msgs
        self.incoming_msgs = {}


class CommunicationMediumUDP(CommunicationMedium):
    def __init__(self, sucess_rate: float=1) -> None:
        super().__init__()
        self.sucess_rate = sucess_rate

    def set_agent_msgs(self, src_id: str, target_ids: [list, np.array], msgs: [list, np.array]): # should we include target_id?
        """set the messages buffer for a given agent id and target id"""
        if target_id not in self.incoming_msgs:
            self.incoming_msgs[target_id] = {}
        for target_id, msg in zip(target_ids, msgs):
            if np.random.rand() < self.sucess_rate:
                self.incoming_msgs[target_id] = {src_id: msg}