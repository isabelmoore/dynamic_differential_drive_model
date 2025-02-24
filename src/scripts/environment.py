#!/usr/bin/env python3

import numpy as np
import math
import cv2
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import random
import unittest

import rospy
from nav_msgs.msg import OccupancyGrid

from abc import abstractmethod

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

class Pose2d:
    def __init__(self, *args):
        if len(args) == 3:
            self.center = Point(args[0], args[1])
            self.yaw = args[2]
        elif len(args) == 2:
            self.center = args[0]
            self.yaw = args[1]
        else:
            print('Invalid arguments: %d args' %len(args))

class Shape:
    '''
    center - Point in meters from lower-left corner of environment
    '''
    def __init__(self, center):
        self.center = center
        self.shift = 7 # 7 bit decimal precision
        self.artist = None

    def __del__(self):
        self.plotRemove()

    def updateCenter(self, center):
        self.center = center

    '''
    pose - Pose2d
    '''
    @abstractmethod
    def updatePose(self, pose):
        pass

    '''
    Fill image pixels using OpenCV
    '''
    @abstractmethod
    def draw(self, img, relCenter, resolution, value=100):
        pass

    '''
    Plot using Matplotlib
    '''
    @abstractmethod
    def plot(self, ax, relCenter, resolution, color):
        pass

    def plotRemove(self):
        if self.artist is not None:
            self.artist.remove()

class Circle(Shape):
    def __init__(self, center, radius):
        super().__init__(center)
        self.radius = radius
        self.artist = None

    def updatePose(self, pose):
        self.updateCenter(pose.center)

    def draw(self, img, relCenter, resolution, value=100):
        factor = 1 << self.shift
        center = (round(relCenter[0] / resolution * factor), round(relCenter[1]
            / resolution * factor))
        radius = round(self.radius / resolution * factor)
        cv2.circle(img, center, radius, value, thickness=-1, shift=self.shift)

    def plot(self, ax, relCenter, resolution, color='black'):
        center = (relCenter[0] / resolution, relCenter[1] / resolution)
        radius = self.radius / resolution

        cir = patches.Circle(center, radius, fill=False, linewidth=1,
                color=color)
        self.artist = ax.add_patch(cir)

class Rectangle(Shape):
    '''
    width - meters
    height - meters
    angle - degrees in ENU convention
    '''
    def __init__(self, center, width, height, angle):
        super().__init__(center)
        self.width = width
        self.height = height
        self.angle = angle

    def updateAngle(self, angle):
        self.angle = angle

    def updatePose(self, pose):
        self.updateCenter(pose.center)
        self.updateAngle(pose.yaw * 180.0 / np.pi)

    def draw(self, img, relCenter, resolution, value=100):
        center = (relCenter[0] / resolution, relCenter[1] / resolution)
        rotatedRect = (center, (self.width / resolution, self.height /
            resolution), self.angle)
        vertices = cv2.boxPoints(rotatedRect)
        factor = 1 << self.shift
        for i in range(vertices.shape[0]):
            for j in range(vertices.shape[1]):
                vertices[i, j] = round(vertices[i, j] * factor)
        vertices = vertices.astype('int32')
        cv2.fillConvexPoly(img, vertices, value, shift=self.shift)

    def plot(self, ax, relCenter, resolution, color='black'):
        # Calculate anchor
        angleRad = self.angle * math.pi / 180
        c = (relCenter[0] / resolution, relCenter[1] / resolution)
        w = self.width / resolution
        h = self.height / resolution
        x = c[0] - w / 2 * math.cos(angleRad) + h / 2 * math.sin(angleRad)
        y = c[1] - w / 2 * math.sin(angleRad) - h / 2 * math.cos(angleRad)
        anchor = (x, y)

        rect = patches.Rectangle(anchor, self.width / resolution, self.height /
                resolution, self.angle, fill=False, linewidth=1, color=color)
        self.artist = ax.add_patch(rect)

class Obstacle2d:
    def __init__(self, shape):
        self.shape = shape

    '''
    img - 2D Numpy array image
    origin - Point of image origin
    '''
    def draw(self, img, origin, resolution, value=100):
        relCenter = (self.shape.center.x - origin.x,
                self.shape.center.y - origin.y)
        self.shape.draw(img, relCenter, resolution, value)

    def plot(self, ax, origin, resolution, color='red'):
        relCenter = (self.shape.center.x - origin.x,
                self.shape.center.y - origin.y)
        self.shape.plot(ax, relCenter, resolution, color=color)

    def plotRemove(self):
        self.shape.plotRemove()

class Footprint(Obstacle2d):
    def __init__(self, shape):
        super().__init__(shape)

    def plot(self, ax, origin, resolution, color='green'):
        Obstacle2d.plot(self, ax, origin, resolution, color=color)

def plotCostmap(costmap):
    plt.imshow(costmap, cmap='binary', vmin=0, vmax=100, origin='lower')
    plt.show()

def publishCostmap(costmap, width=100, height=100, resolution=0.2):
    assert costmap.shape == (width, height), 'costmap shape and dimension' \
            + ' arguments do not agree'

    rospy.init_node('costmap_publisher')
    rate = rospy.Rate(1) # Hz

    pub = rospy.Publisher('costmap', OccupancyGrid, queue_size=1)
    msg = OccupancyGrid()
    msg.header.frame_id = 'base_link'
    msg.info.resolution = resolution
    msg.info.width = width # cells
    msg.info.height = height # cells
    msg.info.origin.position.x = -width / 2 * resolution
    msg.info.origin.position.y = -height / 2 * resolution
    msg.info.origin.orientation.w = 1.0
    
    for row in range(width):
        for col in range(height):
            msg.data.append(costmap[row, col])

    while not rospy.is_shutdown():
        pub.publish(msg)
        rate.sleep()

class Environment2d:
    '''
    start - Pose2d start pose
    '''
    def __init__(self, start, costmapWidth=100, costmapHeight=100,
            costmapResolution=0.2):
        self.start = start
        self.obstacles = []
        self.footprint = None
        self.costmap = None
        self.costmapWidth = costmapWidth
        self.costmapHeight = costmapHeight
        self.costmapResolution = costmapResolution
        self.calcCostmapProperties()

    def calcCostmapProperties(self):
        self.costmapDistance = np.zeros((self.costmapHeight, self.costmapWidth))
        self.costmapHeading = np.zeros((self.costmapHeight, self.costmapWidth))
        
        rowC = (self.costmapHeight - 1) / 2
        colC = (self.costmapWidth - 1) / 2
        for row in range(self.costmapHeight):
            for col in range(self.costmapWidth):
                self.costmapDistance[row, col] = ((row - rowC) ** 2 + (col -
                    colC) ** 2) ** 0.5
                self.costmapHeading[row, col] = math.atan2(row - rowC, col -
                        colC)

        # normalize
        #self.costmapDistance /= ((rowC ** 2 + colC ** 2) ** 0.5)
        #self.costmapHeading /= np.pi

    def getCostmapProperties(self):
        return self.costmapDistance, self.costmapHeading
    
    def addObstacle(self, obstacle):
        self.obstacles.append(obstacle)
        #print('Added obstacle')

    def clearObstacles(self):
        self.obstacles = []
    
    def addRandomObstacles(self, number=1, xBounds=(0, 100 * 0.2), yBounds=(0,
        100 * 0.2)):
        radiusMin = 2
        radiusMax = 6
        for _ in range(number):
            center = Point(random.uniform(xBounds[0], xBounds[1]),
                    random.uniform(yBounds[0], yBounds[1]))
            radius = random.uniform(radiusMin, radiusMax)
            ob = Obstacle2d(Circle(center, radius))
            self.obstacles.append(ob)

    def updateFootprint(self, shape):
        self.footprint = Footprint(shape)

    '''
    center - Point center of costmap
    width - cells wide
    height - cells high
    resolution - meters per cell
    '''
    def updateCostmap(self, center, width=None, height=None, resolution=None):
        # Update costmap variables
        update = False
        if width is not None:
            self.costmapWidth = width
            update = True
        if height is not None:
            self.costmapHeight = height
            update = True
        if resolution is not None:
            self.costmapResolution = resolution
            update = True
        if update:
            self.calcCostmapProperties()

        self.costmap = np.zeros((self.costmapHeight, self.costmapWidth),
                dtype=np.int8)
        self.costmapOrigin = Point(center.x - self.costmapWidth / 2 *
                self.costmapResolution, center.y - self.costmapHeight / 2 *
                self.costmapResolution)
        for ob in self.obstacles:
            ob.draw(self.costmap, self.costmapOrigin, self.costmapResolution)

    def getCostmap(self, dtype=np.int8):
        if not dtype == np.int8:
            return self.costmap.astype(dtype)
        return self.costmap

    def detectCollision(self):
        numOccupied = (self.costmap != 0).sum() # count occupied cells
        self.footprint.draw(self.costmap, self.costmapOrigin,
                self.costmapResolution, value=0)
        newNumOccupied = (self.costmap != 0).sum()

        if newNumOccupied != numOccupied:
            return True
        return False

    '''
    pose - Pose2d relative pose from start
    '''
    def updateOdom(self, pose, dtype=np.int8):
        absPose = Pose2d(pose.center + self.start.center, pose.yaw +
                self.start.yaw)
        self.footprint.shape.updatePose(absPose)
        self.updateCostmap(absPose.center)
        collision = self.detectCollision()

        return collision, self.getCostmap(dtype)

    '''
    Obsolete...
    '''
    def getCostmapMultiplier(self):
        multiplier = np.zeros((self.costmapWidth, self.costmapHeight))

        rowC = (self.costmapHeight - 1) / 2
        colC = (self.costmapWidth - 1) / 2
        for row in range(self.costmapHeight):
            for col in range(self.costmapWidth):
                multiplier[row, col] = ((rowC - abs(row - rowC)) ** 2 + (colC -
                    abs(col - colC)) ** 2) ** 1

        return multiplier

    def plotInit(self, grid=False):
        #plt.ion() # interactive on
        fig, self.ax = plt.subplots()
        if grid:
            self.ax.xaxis.set_major_locator(MultipleLocator(10))
            self.ax.yaxis.set_major_locator(MultipleLocator(10))
            self.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
            self.ax.yaxis.set_minor_locator(AutoMinorLocator(1))
            self.ax.grid(which='both')
        self.grid = grid
        plt.draw()
        plt.pause(0.1)
    
    def plotClear(self):
        if self.footprint is not None:
            self.footprint.plotRemove()
        for ob in self.obstacles:
            if ob is not None:
                ob.plotRemove()

    def plot(self):
        # Clear axis/artists
        plt.cla() # clear entire axis
        #self.plotClear() # clear artists (slow)

        self.ax.imshow(self.costmap, cmap='binary', vmin=0, vmax=100,
                origin='lower')
        if self.grid:
            self.ax.xaxis.set_major_locator(MultipleLocator(5))
            self.ax.yaxis.set_major_locator(MultipleLocator(5))
            self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            self.ax.grid(which='both')
        if self.footprint is not None:
            self.footprint.plot(self.ax, self.costmapOrigin,
                    self.costmapResolution)
        for ob in self.obstacles:
            ob.plot(self.ax, self.costmapOrigin, self.costmapResolution)
        plt.draw()
        plt.pause(1e-6) # 0.1

class Test(unittest.TestCase):
    def testInterface(self):
        env = Environment2d(start=Pose2d(50 * 0.2, 50 * 0.2, 0))

        o1 = Obstacle2d(Circle(Point(50 * 0.2, 50 * 0.2), 2))
        env.addObstacle(o1)

        env.updateFootprint(Rectangle(Point(50 * 0.2, 50 * 0.2), 4.0, 4.0, 0))
        # TODO: initialize rectangle with Pose2d instead of Point

        #o2 = Obstacle2d(Rectangle(Point(50.0 * 0.2, 50.0 * 0.2), 16, 8, 0))
        #env.addObstacle(o2)

        env.addRandomObstacles(10)

        env.updateCostmap(env.start.center)
        plotCostmap(env.getCostmap())
        result, cm = env.updateOdom(Pose2d(0, 0, 0))
        print(result)
        plotCostmap(cm)

    def testObstacles(self):
        env = Environment2d(start=Pose2d(5.0, 5.0, 0),
                costmapWidth=50, costmapHeight=50, costmapResolution=0.2)

        env.addObstacle(Obstacle2d(Circle(Point(3.5, 3.5), 0.25)))
        env.addObstacle(Obstacle2d(Circle(Point(6.5, 3.5), 0.25)))
        env.addObstacle(Obstacle2d(Circle(Point(6.5, 6.5), 0.25)))
        env.addObstacle(Obstacle2d(Circle(Point(3.5, 6.5), 0.25)))

        env.addObstacle(Obstacle2d(Rectangle(Point(5.0, 5.0), 0.310, 0.420,
            0.0)))

        env.updateCostmap(Point(5.0, 5.0))
        plotCostmap(env.costmap)
        
        env.obstacles[4].shape.updateCenter(Point(6.5, 5.0))
        env.updateCostmap(Point(6.5, 5.0))
        plotCostmap(env.costmap)

    def testUpdateOdom(self):
        env = Environment2d(start=Pose2d(5.0, 5.0, 0),
                costmapWidth=50, costmapHeight=50, costmapResolution=0.2)

        env.addObstacle(Obstacle2d(Circle(Point(3.5, 3.5), 10)))
        env.updateFootprint(Rectangle(Point(5.0, 5.0), 3, 1, 0.0))

        odom = Pose2d(0.0, 0.0, 0.0)
        env.plotInit()
        for i in range(50):
            odom.center.x += 0.0
            odom.center.y += 0.0
            odom.yaw += 0.2
            env.updateOdom(odom)
            env.plot()

    def testPlot(self):
        cellDimension = 40 # 400
        resolution = 0.1 # 0.01

        env = Environment2d(start=Pose2d(5.0, 5.0, 0),
                costmapWidth=cellDimension, costmapHeight=cellDimension,
                costmapResolution=resolution)
        
        env.addObstacle(Obstacle2d(Circle(Point(3.5, 3.5), 0.25)))
        env.addObstacle(Obstacle2d(Circle(Point(6.5, 3.5), 0.25)))
        env.addObstacle(Obstacle2d(Circle(Point(6.5, 6.5), 0.25)))
        env.addObstacle(Obstacle2d(Circle(Point(3.5, 6.5), 0.25)))

        env.updateFootprint(Rectangle(Point(5.0, 5.0), 0.310, 0.420, 0.0))

        odom = Pose2d(0.0, 0.0, 0.0)
        env.plotInit()
        for i in range(50):
            odom.center.x += .03
            odom.center.y += .03
            odom.yaw += 0.0
            env.updateOdom(odom)
            env.plot()

    def testMultiplier(self):
        cellDimension = 8 # 400
        resolution = 0.1 # 0.01

        env = Environment2d(start=Pose2d(5.0, 5.0, 0),
                costmapWidth=cellDimension, costmapHeight=cellDimension,
                costmapResolution=resolution)

        cmm = env.getCostmapMultiplier()
        cm = np.zeros((cellDimension, cellDimension))
        cm[3, 3] = 1

        mat = np.multiply(cm, cmm)
        print(cmm)
        print(mat)
        print(np.sum(mat))

    def testCostmapProperties(self):
        cellDimension = 4
        resolution = 1.0

        env = Environment2d(start=Pose2d(2.0, 2.0, 0),
                costmapWidth=cellDimension, costmapHeight=cellDimension,
                costmapResolution=resolution)

        env.calcCostmapProperties()
        print(env.costmapDistance)
        print(env.costmapHeading)

        cm = np.zeros((env.costmapHeight, env.costmapWidth))
        cm[2, 2] = 1
        print(cm)

if __name__ == '__main__':
    unittest.main()
