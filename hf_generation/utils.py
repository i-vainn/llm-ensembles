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
