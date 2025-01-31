#!/usr/bin/env python3

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import time

import environment as env
from gym_env import heading_error

class CostmapReward:
    def __init__(self, resolution=0.1):
        start = env.Pose2d(5.0, 5.0, 0.0)
        costmapResolution = resolution # meters per cell
        costmapDimension = int(4.0 / costmapResolution) # cells
        self.env = env.Environment2d(start, costmapWidth=costmapDimension,
                costmapHeight=costmapDimension,
                costmapResolution=costmapResolution)
        self.costmapDistance, self.costmapHeading = \
                self.env.getCostmapProperties()
        self.env.updateCostmap(start.center)
        self.costmap = self.env.getCostmap()

    def plotCellContribution(self):
        heading = 0.0
        result = np.zeros(self.costmap.shape)
        angle_threshold = math.pi / 2 # angle from heading for considering
        # obstacles
        range_threshold = (self.costmap.shape[0] - 1) / 2 # maximum range in
        # cells for considering obstacles

        for row in range(self.costmap.shape[0]):
            for col in range(self.costmap.shape[1]):
                rel_heading = heading_error(self.costmapHeading[row, col],
                        heading)
                distance = self.costmapDistance[row, col]
                if abs(rel_heading) <= angle_threshold and distance <= \
                        range_threshold:
                    result[row, col] = ((angle_threshold - abs(rel_heading))
                            / angle_threshold) ** 2 * ((range_threshold -
                                distance) / range_threshold) ** 2

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        x = range(result.shape[1])
        y = range(result.shape[0])
        X, Y = np.meshgrid(x, y)

        #ax.plot_wireframe(X, Y, result)
        ax.plot_surface(X, Y, result, cmap=cm.jet)
        ax.set_xlabel('x [cell]')
        ax.set_ylabel('y [cell]')
        plt.show()

        print(result.sum())

    def loopCalculation(self):
        heading = 0.0
        R = np.zeros(self.costmap.shape)
        angle_threshold = math.pi / 2 # angle from heading for considering
        # obstacles
        range_threshold = (self.costmap.shape[0] - 1) / 2 # maximum range in
        # cells for considering obstacles

        for row in range(self.costmap.shape[0]):
            for col in range(self.costmap.shape[1]):
                rel_heading = heading_error(self.costmapHeading[row, col],
                        heading)
                distance = self.costmapDistance[row, col]
                if abs(rel_heading) <= angle_threshold and distance <= \
                        range_threshold:
                    R[row, col] = ((angle_threshold - abs(rel_heading))
                            / angle_threshold) ** 2 * ((range_threshold -
                                distance) / range_threshold) ** 2
        
        return R, R.sum()
    
    def vectorizedCalculation(self):
        heading = 0.0
        angle_threshold = math.pi / 2
        range_threshold = (self.costmap.shape[0] - 1) / 2
        costmap = np.ones(self.costmap.shape, dtype=np.bool_)
        
        # Calcualte absolute relative heading matrix, H
        H = self.costmapHeading - heading
        H[H > math.pi] -= 2 * math.pi
        H[H < -math.pi] += 2 * math.pi
        H = abs(H)

        # Calculate reward matrix, R
        R = costmap * (1 - H / angle_threshold) ** 2 * (1 - self.costmapDistance
                / range_threshold) ** 2
        R[~((H < angle_threshold) & (self.costmapDistance < range_threshold))] \
                = 0 # zero irrevelant cells

        return R, R.sum()

def visualization():
    cr = CostmapReward()
    cr.plotCellContribution()

def compareCalcs():
    cr = CostmapReward(resolution=.01)
    t0 = time.time()
    R_n, r_n = cr.loopCalculation()
    t1 = time.time()
    R_v, r_v = cr.vectorizedCalculation()
    t2 = time.time()
    print('loop: %s' %(t1 - t0))
    print('vectorized: %s' %(t2 - t1))
    print('equal: %s' %np.allclose(R_n, R_v))

if __name__ == '__main__':
    #visualization()
    compareCalcs()
