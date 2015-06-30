import numpy as np

class viterbi(object):
    def __init__(self, states_hid, states_obs, transition_hid, transition_obs):
        self.states_hid = states_hid
        self.states_obs = states_obs
        assert transition_hid.shape == (states_hid, states_hid)
        assert transition_obs.shape == (states_hid, states_obs)
        self.transition_hid = transition_hid
        self.transition_obs = transition_obs

    def get_MLP(self, observations):
        n_chain = observations.size
        assert np.unique(observations)[-1] < self.states_obs

        # declare the return value == most likely path
        mlp = np.zeros( n_chain )
        # declare messages:
        alpha = np.zeros( (n_chain, self.states_hid) )
        beta  = np.zeros( (n_chain, self.states_hid) )
        gamma = np.zeros( (n_chain, self.states_hid) )

        #init messages
        for i in range( n_chain ):
            obs = observations[i]
            gamma[i,:] = - np.log( self.transition_obs[:,obs] )
        beta[-1,:] = 0

        #backward sweep
        for i in np.arange( n_chain - 1 )[::-1]:
            beta[i,0] = np.min( gamma[i+1] - np.log( self.transition_hid[:,0] ) + beta[i+1] )
            beta[i,1] = np.min( gamma[i+1] - np.log( self.transition_hid[:,1] ) + beta[i+1] )

        mlp[0] = np.argmin( beta[0] + gamma[0] )

        #forward sweep
        for i in range(1, n_chain):
            alpha[i,:] = - np.log( self.transition_hid[ : , mlp[i-1] ] )
            mlp[i]     = np.argmin( alpha[i] + beta[i] + gamma[i] )

        return mlp

