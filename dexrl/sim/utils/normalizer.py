import torch

class Normalizer_N1_1:
    def __init__(self, min_value: torch.Tensor, max_value: torch.Tensor):
        self.min_value = min_value
        self.max_value = max_value

        self.scale = self.max_value - self.min_value
    #     self.bias = (max_value + min_value) / 2

    def normalize(self, denormalized_value):
        return 2 * (denormalized_value - self.min_value) / (self.scale) - 1
    
    def denormalize(self, normalized_value):
        return (normalized_value + 1) * (self.scale) / 2 + self.min_value


class Normalizer_0_1(Normalizer_N1_1):
    def normalize(self, denormalized_value):
        return (denormalized_value - self.min_value) / (self.scale)
    
    def denormalize(self, normalized_value):
        return normalized_value * self.scale + self.min_value


class Normalizer_N1_1(Normalizer_N1_1):
    def normalize(self, denormalized_value):
        return 2 * (denormalized_value - self.min_value) / (self.scale) - 1
    
    def denormalize(self, normalized_value):
        return (normalized_value + 1) * (self.scale) / 2 + self.min_value