#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import queue

class CommandPublisher:
    def __init__(self, command_queue, rate=50):
        self.queue = command_queue
        self.rate = rate

    def ros_init(self):
        rospy.init_node('command_publisher')
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def publish(self):
        self.ros_init()
        rate = rospy.Rate(self.rate)
        msg = Twist()
        while not rospy.is_shutdown():
            try:
                # check queue
                try:
                    values = self.queue.get(False) # no wait
                    msg.linear.x = values[0]
                    msg.angular.z = values[1]
                except queue.Empty:
                    pass
                self.pub.publish(msg) # publish command
                rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                rospy.logdebug('Caught ROSTimeMovedBackwardsException')
