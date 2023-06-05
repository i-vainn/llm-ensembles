import torch
from transformers import StoppingCriteria

def permute_vocab(reference, target):
    """
        This is a naive method for matching two vocabs
        of different sizes. It permutes tokens of target vocab
        so that their possitions match reference vocab tokens.
    """
    assert len(set(reference).difference(target)) == 0, \
            "Reference vocab should be subset of target vocab."

    permutation = [0] * len(reference)
    for key, val in target.items():
        if key in reference:
            permutation[reference[key]] = val
    
    return permutation

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops_ids = []):
        super().__init__()
        self.stop_ids = [stop for stop in stops_ids]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stop_ids:
            if (stop == input_ids[0][-1]).item():
                return True

        return False
