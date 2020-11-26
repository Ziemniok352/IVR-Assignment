#!/usr/bin/env python3
import roslib
import sys
import rospy
import cv2
import numpy as np
from math import pi
from numpy import sin
from numpy import cos
from functools import reduce
from std_msgs.msg import String
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError

class control:

    def __init__(self):
        # initialize the node named control
        rospy.init_node('control', anonymous=True)

        # initialize a publisher to send joints' angular position to the robot
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        #use actual pos for now
        self.robot_pos_sub = rospy.Subscriber("/robot/joint_states",JointState, self.callback)   
        self.target_pos_sub = rospy.Subscriber("/target/joint_states",JointState, self.targetCallback)
        
        #self.robot_pos_sub = rospy.Subscriber(get from img processing)
        #self.target_pos_sub = rospy.Subscriber(get from img processing) or directly from target move
        
        # record the begining time
        self.time_trajectory = rospy.get_time()
        # initialize everything
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')     
        self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')   

        self.error = np.array([0.0,0.0,0.0], dtype='float64')  
        self.error_d = np.array([0.0,0.0,0.0], dtype='float64') 

        self.target_position = np.array([0.0,0.0,0.0])

    # this gets target coords in right format    
    def targetCallback(self,data):
        self.target_position = np.asarray(data.position)

    def fk(self, angles):
        theta1 = angles[0] + pi/2
        theta2 = angles[1] + pi/2
        theta3 = angles[2]
        theta4 = angles[3]

        s1,s2,s3,s4 = sin(theta1), sin(theta2), sin(theta3), sin(theta4)
        c1,c2,c3,c4 = cos(theta1), cos(theta2), cos(theta3), cos(theta4)

        x = 3.5*c1*c2*c3 + 3.5*s1*s3 + 3*c4*(c1*c2*c3 + s1*s3) - 3*c1*s2*s4
        y = 3.5*s1*c2*c3 + 3*c4*(s1*c2*c3 - c1*s3) - 3*s1*s2*s4 - 3.5*c1*s3
        z = 3.5*s2*c3 + 3*s2*c3*c4 + 3*c2*s4 + 2.5
        #print(x,y,z)
        return np.array([x,y,z])


    def calculate_jacobian(self, angles):
        theta1 = angles[0]
        theta2 = angles[1]
        theta3 = angles[2]
        theta4 = angles[3]

        s1,s2,s3,s4 = sin(theta1), sin(theta2), sin(theta3), sin(theta4)
        c1,c2,c3,c4 = cos(theta1), cos(theta2), cos(theta3), cos(theta4)
        
        jacobian = np.array([[3.5*c1*c3*s2 + 3*c1*c2*s4 + 3*(c1*c3*s2 - s1*s3)*c4 - 3.5*s1*s3,
                3*c2*c3*c4*s1 + 3.5*c2*c3*s1 - 3*s1*s2*s4,-3.5*s1*s2*s3 + 3.5*c1*c3 - 3*(s1*s2*s3 - c1*c3)*c4,
                3*c2*c4*s1 - 3*(c3*s1*s2 + c1*s3)*s4],
                [3.5*c3*s1*s2 + 3*c2*s1*s4 + 3*(c3*s1*s2 + c1*s3)*c4 + 3.5*c1*s3,
                -3*c1*c2*c3*c4 - 3.5*c1*c2*c3 + 3*c1*s2*s4,
                3.5*c1*s2*s3 + 3*(c1*s2*s3 + c3*s1)*c4 + 3.5*c3*s1,-3*c1*c2*c4 + 3*(c1*c3*s2 - s1*s3)*s4],
                [0,-3*c3*c4*s2 - 3.5*c3*s2 - 3*c2*s4,-3*c2*c4*s3 - 3.5*c2*s3,
                -3*c2*c3*s4 - 3*c4*s2]])
        return jacobian

    def closed_loop_control(self,position):
        #P gain
        K_p = np.array([[1,0,0],[0,1,0],[0,0,1]])
        #D gain
        K_d = np.array([[0.1,0,0],[0,0.1,0],[0,0,0.1]])
        #estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        self.time_previous_step = cur_time

        #robot end-effector position (xyz)
        pos = self.fk(position)
        #desired trajectory
        pos_d = self.target_position
        #estimate derivative of error
        self.error_d = ((pos_d - pos) - self.error)/dt
        #estimate error
        self.error = pos_d-pos

        #end effector position (angles)
        q = position
        #print("end effector: {}".format(pos))
        #print("target: {}".format(self.target_position))
        J_inv = np.linalg.pinv(self.calculate_jacobian(q))  # calculating the psudeo inverse of Jacobian
        dq_d =np.dot(J_inv, (np.dot(K_d,self.error_d.transpose()) + np.dot(K_p,self.error.transpose())))  # control input (angular velocity of joints)
        q_d = q + (dt * dq_d)  # control input (angular position of joints)
        return q_d


    def callback(self,data):
        position = data.position
        
        q_d = self.closed_loop_control(position)
        #print(q_d)

        self.joint1 = Float64()
        self.joint2 = Float64()
        self.joint3 = Float64()
        self.joint4 = Float64()

        self.joint1.data = q_d[0]       
        self.joint2.data = q_d[1]       
        self.joint3.data = q_d[2]       
        self.joint4.data = q_d[3]       

        self.robot_joint1_pub.publish(self.joint1)
        self.robot_joint2_pub.publish(self.joint2)
        self.robot_joint3_pub.publish(self.joint3)
        self.robot_joint4_pub.publish(self.joint4)
        print("published")

# call class
def main(args):
    c = control()
    c.fk([])
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

    # run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)