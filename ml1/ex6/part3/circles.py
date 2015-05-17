# RANSAC-Algorithmus (Random Sample Consensus)

# @see http://en.wikipedia.org/wiki/RANSAC

import numpy as np
from numpy.linalg import norm

def circleRANSAC(points, r, numberOfCircles):
    # initiate output array containing circles
    circles = [None]*numberOfCircles
    
    # determine RANSAC parameters
    p = 0.99                    #we want a certainty of 99%
    e = 0.25                    #we estimate the inliner percentage of the 
                                #larges circles to be roughly 25%
    trials = np.log(1 - p) / np.log(1 - e**3)   #optimal number of trials
    
    # iterate over number of circles
    for i in range(numberOfCircles):
        incircum = [False]*len(points)              #property map #1
        
        # iterate over optimal number of trials
        for l in range(int(trials)):
            incircum_new = [False]*len(points)              #property map #2
            a = b = c = None
            
            #while  not (a != b != c != a):                  #while not unequal
            while  (a == b) or (b == c) or (c == a):         #while pariwise equal
                a = np.random.randint(len(points))
                b = np.random.randint(len(points))
                c = np.random.randint(len(points))
            
            # find radius and circumcenter
            R = getCircumradius(points[a], points[b], points[c])
            U = getCircumcenter(points[a], points[b], points[c])
            
            # check for inliners
            for k in range(len(points)):
                if abs(norm(U - points[k]) - R ) < r:     #compare radii
                    incircum_new[k] = True
            
            #print(sum(incircum), sum(incircum_new))
            if sum(incircum) < sum(incircum_new):       #update incircle, in case
                incircum = np.copy(incircum_new)       #a better approx is found
                circles[i] = (U, R)
        
        # update points
        copy = []
        for j in range(len(points)):            #delete already allocated points,
            if not incircum[j]:                 #i.e. create new array without them
                copy.append(points[j])
        
        print('outliers / total for the ', i+1, '. circle: ',
              round( len(copy)/len(points), 4) )
        points = np.copy(copy)               #does not alter input points
                                              #function calls by const reference or value
        
    return circles


#@see http://en.wikipedia.org/wiki/Circumscribed_circle
def getCircumcenter(p1, p2, p3):
    #translation
    # a = p1 - p1 = 0
    b = p2 - p1
    c = p3 - p1
    
    #calculate circumcenter
    d = 2*np.cross(b,c)
    ux = (c[1]*b.dot(b) - b[1]*c.dot(c)) / d
    uy = (b[0]*c.dot(c) - c[0]*b.dot(b)) / d
    
    return np.array([ux, uy]) + p1


def getCircumradius(p1, p2, p3):
    #translation
    # a = p1 - p1 = 0
    b = p2 - p1
    c = p3 - p1
    
    #calculate radius
        
    return 0.5*norm(b)*norm(c)*norm(b - c) / norm(np.cross(b,c))


if __name__ == '__main__':
    import pylab as plt
    
    points = np.load('circles.npy')
    print('total number of points:', len(points))
    
    #analysing the png pictures we choose
    r = 0.03
    numberOfCircles = 3
    
    circles = circleRANSAC(points, r, numberOfCircles)
    print(circles)
    
    plt.figure()
    plt.axes().set_aspect('equal')
    #plt.axes([0, 0, 1, 1])
    plt.scatter(points[:,0], points[:,1], s=50, alpha=.5)
    
    plt.xlim(0, 1)
    plt.xticks(())
    plt.ylim(0, 1)
    plt.yticks(())
    
    if True:
        for circle in circles:
            c = plt.Circle(circle[0], circle[1] + r, fill=False, color='g')
            plt.gca().add_artist(c)                   # Kreis in den Plot einfügen
            c = plt.Circle(circle[0], circle[1], fill=False, color='r')
            plt.gca().add_artist(c)
            c = plt.Circle(circle[0], circle[1] - r, fill=False, color='g')
            plt.gca().add_artist(c)
            plt.axis()
    
    plt.savefig('circles.png', dpi=220, bbox_inches='tight')
