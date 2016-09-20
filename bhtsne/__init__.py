
import numpy as np
from bhtsne_wrapper import BHTSNE

class InvalidSeedPositionError(Exception):
    pass

def tsne(data, dimensions=2, perplexity=30.0, theta=0.5, max_iter=1000, stop_lying_iter = 250, mom_switch_iter = 250, momentum = 0.5, final_momentum = 0.8, eta = 200.0, rand_seed=-1, seed_positions=np.array([])):
    tsne = BHTSNE()
    skip_random_init = False
    if len(seed_positions) > 0:
        skip_random_init = True
        if seed_positions.shape[0] != data.shape[0]:
            raise InvalidSeedPositionError("Seed positions needs to be same number of rows as input matrix")
    Y = tsne.run(data, data.shape[0], data.shape[1], dimensions, perplexity, theta, max_iter, stop_lying_iter, mom_switch_iter, momentum, final_momentum, eta, rand_seed, seed_positions, skip_random_init)
    return Y
