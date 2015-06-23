import numpy as np
import pylab as plot
import scipy.sparse
import scipy.sparse.linalg

def get_transition_probability_matrix(M):
    P = np.zeros(M.shape)
    for i in range(M.shape[0]):
        if np.sum(M[i,:]) != 0.:
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


# find the shortest path using dijkstras algorithm
def find_shortest_path(M, start, target):

    distances  = np.inf * np.ones(M.shape[0])
    distances[start] = 0.
    prev  = -1*np.ones(M.shape[0], dtype = int)

    queue = [ x for x in range(M.shape[0]) ]

    while queue:
        min_dist = np.inf
        next_room = -1
        for room in queue:
            if distances[room] < min_dist:
                min_dist = distances[room]
                next_room = room

        if next_room == target:
            print "Escape!"
            break

        queue.remove(next_room)

        for adjacent_room in np.where(M[next_room] == 1)[0]:
            alt_dist = distances[next_room] + 1.
            if alt_dist < distances[adjacent_room]:
                distances[adjacent_room] = alt_dist
                prev[adjacent_room]      = next_room

    path = [target]
    room = prev[target]
    while room != start:
        room = prev[room]
        path.append(room)

    return path, distances[target]

# find the most probable path using dijkstras algorithm
def find_most_probable_path(R, start, target):

    probs  = np.zeros(R.shape[0])
    probs[start] = 1.
    prev  = -1*np.ones(M.shape[0], dtype = int)

    queue = [ x for x in range(M.shape[0]) ]

    while queue:
        max_prob = 0.
        next_room = -1
        for room in queue:
            if probs[room] > max_prob:
                max_prob = probs[room]
                next_room = room

        if next_room == target:
            print "Escape!"
            break

        queue.remove(next_room)

        for adjacent_room in np.where(R[next_room] != 0.)[0]:
            alt_prob = R[next_room,adjacent_room] * probs[next_room]
            if alt_prob > probs[adjacent_room]:
                probs[adjacent_room] = alt_prob
                prev[adjacent_room]  = next_room

    path = [target]
    room = prev[target]
    while room != start:
        room = prev[room]
        path.append(room)

    return path, probs[target]


def plot_maze(M):
    plot.imshow(M,interpolation = "Nearest")
    plot.show()


def get_expected_random_walk_time(R, start, target):
    n_rooms = R.shape[0]
    lin_sys_eq = scipy.sparse.lil_matrix((n_rooms * n_rooms, n_rooms * n_rooms))
    res = np.ones(n_rooms * n_rooms) * -1

    # build a large linear system of equations
    for i in range(n_rooms):
        for j in range(n_rooms):
            # we need to lniearize coordinates i, j to idx
            idx = i * n_rooms + j
            lin_sys_eq[idx, idx] = -1
            for adjacent_room in np.where(R[i] != 0.)[0]:
                idx2 = adjacent_room * n_rooms + j
                lin_sys_eq[idx, idx2] = R[i, adjacent_room]
    lin_sys_eq = lin_sys_eq.tocsc()

    # for some reason this does only work for end up to 98. If end = 99 is gives 3*10**17 as a result(?!)
    # if we run this with cropped M and R (M[:98, :98]) we end up with pretty much the exact time as the random walker...
    all_times = scipy.sparse.linalg.spsolve(lin_sys_eq, res)

    # our result should now be here:
    expected_time = all_times[start * n_rooms + target]

    return expected_time


if __name__ == '__main__':
    M = np.load("maze.npy")
    M -= np.diag( np.ones( M.shape[0] ) )
    R = get_transition_probability_matrix(M)

    #plot_maze(M)

    shortest_path, shortest_dist = find_shortest_path(M, 0, 99)
    print "Shortest Path:", shortest_path, "Distance:", shortest_dist

    max_path, max_prob = find_most_probable_path(R, 0, 99)
    print "Most Probable Path:", max_path, "Probability:", max_prob

    res_exp = get_expected_random_walk_time(R[:98,:98], 0, 97)
    print "Expected time for Random walker to cross the maze:", res_exp

    #res_rw = random_traversal_time(M, 0, 99)
    #print "Experimental time:", res_rw[0], "+-", res_rw[1]
