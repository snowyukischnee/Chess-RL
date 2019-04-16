import numpy as np
from collections import deque
import random
from typing import Any

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))


class Memory(object):
    """
    Memory class for storing observations
    """
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience: Any) -> None:
        """Add a new experience to memory buffer
        :param
            experience(Any): the experience to be added
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int, continuous: bool = True) -> Any:
        """Sample
        :param
            batch_size(int): size of batch
        :param
            continuous(bool): true if the experiences is continuous in the buffer, false otherwise
        :return:
            Any: array of sampled experiences
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self) -> None:
        """Clear the memory buffer
        """
        self.buffer.clear()