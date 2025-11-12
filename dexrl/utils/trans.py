import numpy as np

class Normalizer:
    def __init__(self, min_value: np.ndarray, max_value: np.ndarray):
        self.min_value = min_value
        self.max_value = max_value

        self.scale = self.max_value - self.min_value

class Normalizer_01(Normalizer):
    def normalize(self, denormalized_value):
        return (denormalized_value - self.min_value) / (self.scale)
    
    def denormalize(self, normalized_value):
        return normalized_value * self.scale + self.min_value


class Normalizer_N11(Normalizer):
    def normalize(self, denormalized_value):
        return 2 * (denormalized_value - self.min_value) / (self.scale) - 1
    
    def denormalize(self, normalized_value):
        return (normalized_value + 1) * (self.scale) / 2 + self.min_value
