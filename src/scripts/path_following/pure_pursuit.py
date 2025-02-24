#!/usr/bin/eny python3

import numpy as np
import math
import matplotlib.pyplot as plt

import unittest

'''
Pure Pursuit controller for differential drive vehicle
'''
class PurePursuit:
    def __init__(self, path=[], spacing=0.1, lookahead=10, dAnchor=-1.0):
        self.path = path # list of waypoints
        self.spacing = spacing # distance between waypoints
        self.lookahead = lookahead # lookahead distance
        self.dAnchor = dAnchor # distance along vehicle axis from pose reference to anchor point

    def setLookahead(self, lookahead):
        self.lookahead = lookahead

    def getAnchor(self, pose):
        anchor = np.copy(pose)
        anchor[0] += self.dAnchor * np.cos(pose[2])
        anchor[1] += self.dAnchor * np.sin(pose[2])
        return anchor

    def getClosestIndex(self, pose, targetIndex=None, tolerance=2.0):
        if targetIndex is not None:
            tolIndices = int(tolerance / self.spacing)
            minIndex = max(0, targetIndex - tolIndices)
            maxIndex = min(self.path.shape[1] - 1, targetIndex + tolIndices)
        else:
            minIndex = 0
            maxIndex = self.path.shape[1] - 1

        distances = ((self.path[0] - pose[0]) ** 2 +
                (self.path[1] - pose[1]) ** 2) ** 0.5
        index = np.argmin(distances[minIndex:maxIndex + 1]) + minIndex
        
        return index

    def getLookaheadPoint(self, closestIndex):
        lookaheadWaypoints = self.lookahead / self.spacing
        index = closestIndex + int(lookaheadWaypoints)
        if index >= self.path.shape[1] - 1:
            lookaheadPoint = self.path[:, -1]
        else:
            lookaheadPoint = self.path[:, index].copy() # avoid getting view and
            # modifying original array
            # interpolate
            direction = self.path[:, index + 1] - self.path[:, index]
            lookaheadPoint = lookaheadPoint + (lookaheadWaypoints % 1) * \
                    direction
        return lookaheadPoint

    def run(self, pose, targetIndex=None):
        closestIndex = self.getClosestIndex(pose, targetIndex)
        lookaheadPoint = self.getLookaheadPoint(closestIndex)

        # Calculate desired yaw (ENU convention)
        '''
        yaw = math.atan2(lookaheadPoint[1] - pose[1], lookaheadPoint[0] -
                pose[0])
        return yaw, lookaheadPoint, closestIndex
        '''

        # Calculate required curvature
        anchor = self.getAnchor(pose)
        yaw = math.atan2(lookaheadPoint[1] - anchor[1], lookaheadPoint[0] -
                anchor[0])
        relative_bearing = yaw - anchor[2]
        if relative_bearing > math.pi:
            relative_bearing -= 2 * math.pi
        elif relative_bearing < -math.pi:
            relative_bearing += 2 * math.pi
        curvature = 2 * math.sin(relative_bearing) / self.lookahead

        return curvature, lookaheadPoint, closestIndex

class Test(unittest.TestCase):
    def testClosestIndex(self):
        path = np.array([[0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5]])
        pose = np.array([4, 3, 0])
        pp = PurePursuit(path)
        pp.run(pose)

    def testLookaheadPoint(self):
        #path = np.array([[0, 1, 2, 3, 4, 5],
        #    [0, 1, 2, 3, 4, 5]])
        path = np.array([[0, .1, .2, .3, .4, .5],
                         [0, .1, .1,  0, .2, .3],
                         [0,  0,  0,  0,  0,  0]])
        pp = PurePursuit(path, spacing=0.1, lookahead=0.17)
        lookahead_points = np.zeros(path.shape)

        for i in range(path.shape[1]):
            c, point, cI = pp.run(path[:, i])
            lookahead_points[:, i] = point

        fig, ax = plt.subplots()
        ax.plot(path[0], path[1], color='black')
        ax.scatter(lookahead_points[0], lookahead_points[1], color='green')
        
        plt.show()

    def testAnchor(self):
        path = []
        pose = np.zeros(3)
        pp = PurePursuit(path, dAnchor=1.0)
        fig, ax = plt.subplots()

        for yaw in np.linspace(0, 2 * np.pi, 50):
            print(yaw)
            pose[2] = yaw
            anchor = pp.getAnchor(pose)
            ax.scatter(anchor[0], anchor[1])

        ax.axis('equal')
        plt.show()
