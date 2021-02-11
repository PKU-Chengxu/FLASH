import numpy as np
from abc import ABC, abstractmethod

class Compressor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def GradientCompress(self, gradient_list):
        pass
        # return compressed_gradient_list, old_size, new_size


