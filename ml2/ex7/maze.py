import numpy as np

def get_transition_probability_matrix(M):
    P = np.zeros(M.shape)
    for i in range(M.shape[0]):
        P[i,:] = M[i,:] / np.sum(M[i,:])
        assert np.isclose( np.sum(P[i,:]), 1. )
    return P

def random_walker(M, start, target):
    pos = start
    reached_target = False
    steps = 0
    while not reached_target:
        steps += 1
        reachable = np.where( M[pos,:] != 0 )[0]
        num_reachable = reachable.size
        random_step = np.random.randint(0, num_reachable)
        pos = reachable[random_step]
        if pos == target:
            reached_target = True
    return steps

def random_traversal_time(M, start, target, iterations = 100000):
    assert start >= 0
    assert target >= 0
    assert start < M.shape[0]
    assert target < M.shape[0]
    times = np.zeros(iterations)
    for i in range(iterations):
        if i % (iterations/100.) == 0:
            print "Traversal time calculation via RW:", i / float(iterations) * 100., "%"
        times[i] = random_walker(M, start, target)
    return np.mean( times ), np.std( times )


if __name__ == '__main__':
    M = np.load("maze.npy")
    P = get_transition_probability_matrix(M)
    res = random_traversal_time(M, 0, 99)
    print res[0], "+-", res[1]
