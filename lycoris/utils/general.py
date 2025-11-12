import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math

class AID(nn.Module):
    def __init__(self, p=0.9):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor):
        if self.training:
            pos_mask = (x >= 0) * torch.bernoulli(torch.ones_like(x) * self.p)
            neg_mask = (x < 0) * torch.bernoulli(torch.ones_like(x) * (1 - self.p))
            return x * (pos_mask + neg_mask)
        else:
            pos_part = (x >= 0) * x * self.p
            neg_part = (x < 0) * x * (1 - self.p)
            return pos_part + neg_part
        
    @staticmethod
    def get_point_on_curve(block_id, total_blocks=38, peak=0.9, shift=0.75):
        # Normalize the position to 0-1 range
        normalized_pos = block_id / total_blocks

        # Shift the sine curve to only use the first 3/4 of the cycle
        # This gives us: start at 0, peak in the middle, end around 0.7
        phase_shift = shift * math.pi
        sine_value = math.sin(normalized_pos * phase_shift)

        # Scale to our desired peak of 0.9
        result = peak * sine_value

        return result

def product(xs: list[int | float]):
    res = 1
    for x in xs:
        res *= x
    return res

