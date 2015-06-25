import numpy as np

class viterbi(object):
    def __init__(self, states_hid, states_obs, transition_hid, transition_obs):
        self.states_hid = states_hid
        self.states_obs = states_obs
        assert transition_hid.shape == (states_hid, states_hid)
        assert transition_obs.shape == (states_hid, stats_obs)
        self.transition_hid = transition_hid
        self.transition_obs = transition_obs

    def get_MLP(observations):
        n_chain = observations.size
        assert np.unique(observations)[-1] < self.states_obs


