import numpy as np
from viterbi import viterbi


if __name__ == '__main__':
    n_hid = 2
    n_obs = 3

    trans_hid = np.array( [ [0.5,0.5], [0.5,0.5] ] )
    trans_obs = np.array( [ [0.5,0.4,0.1], [0.4,0.1, 0.5] ])

    solver = viterbi(n_hid, n_obs, trans_hid, trans_obs)

    obs = np.array( [0,1,1,0,2,0,2,2,2,0,2,2,2,2,2,0,0,1,1,2] )

    mlp = solver.get_MLP(obs)

    print mlp
