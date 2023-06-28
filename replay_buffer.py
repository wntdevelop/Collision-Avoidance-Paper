REPALY_MEMORY = 1000000
MIN_BATCH_SIZE = 1024

import random
import numpy as np
from collections import deque

# experience replay 
class ReplayBuffer:

    def __init__(self):
        self.epuck_experiences = deque(maxlen=REPALY_MEMORY)                                               

    # store experience
    def store_experience(self, state, next_state, reward, action, done):                       
        self.epuck_experiences.append((state, next_state, reward, action, done))

    # smapling experiences in storage
    def sample_batch(self):                                                                    
        batch_size = min(MIN_BATCH_SIZE, len(self.epuck_experiences))                                           
        sampled_epuck_batch = random.sample(self.epuck_experiences, batch_size)                   
        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        for epuck_experience in sampled_epuck_batch:
            state_batch.append(epuck_experience[0])
            next_state_batch.append(epuck_experience[1])
            reward_batch.append(epuck_experience[2])
            action_batch.append(epuck_experience[3])
            done_batch.append(epuck_experience[4])
        return np.array(state_batch), np.array(next_state_batch), np.array(
            action_batch), np.array(reward_batch), np.array(done_batch)                                 