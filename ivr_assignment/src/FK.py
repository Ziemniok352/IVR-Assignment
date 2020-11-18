#!/usr/bin/env python
from __future__ import print_function

import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import Float64MultiArray


class FK:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named forward_kinematics
        rospy.init_node('forward_kinematics', anonymous=True)
        # initialize a publisher to publish the end effector position calculated by FK
        self.end_effector_pub = rospy.Publisher("/end_effector_position", Float64MultiArray, queue_size=10)
        self.end_effector_position = Float64MultiArray()
        self.joints = np.array([0.1, 0.2, 0.3, 0.4]) #set this to the value of rostopic inputs to validate FK

    def fk_end_effector_estimate(self):
        
        end_effector = self.calculate_fk(self.joints)
	#Compare the fk end effector to something? Sin signals?
        print("FK  x: {:.2f}, y: {:.2f}, z: {:.2f}".format(end_effector[0], end_effector[1], end_effector[2]), end='\r')
        self.end_effector_position.data = end_effector
        self.end_effector_pub.publish(self.end_effector_position)

    def calculate_fk(self, joints):
        s1 = np.sin(joints[0])
        s2 = np.sin(joints[1])
        s3 = np.sin(joints[2])
        s4 = np.sin(joints[3])
        c1 = np.cos(joints[0])
        c2 = np.cos(joints[1])
        c3 = np.cos(joints[2])
        c4 = np.cos(joints[3])
        
        # CALC FK HERE
        end_effector = np.array([x, y, z])
        
        return end_effector


# call the class
def main(args):
    fk = FK()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
© 2020 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
