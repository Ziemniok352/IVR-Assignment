#!/usr/bin/env python3
print('Starting...')
import roslib
import sys
import rospy
import cv2
import numpy as np
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
#print("Imports done :)")

class image_converter:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send messages to a topic named image_topic
        self.image_pub = rospy.Publisher("image_topic", Image, queue_size=1)
        # initialize a publisher to send joints' angular position to a topic called joints_pos
        self.joints_pub = rospy.Publisher("joints_pos", Float64MultiArray, queue_size=10)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        # initialize a subscribers and time synchronizer
        self.image_sub1 = message_filters.Subscriber("/camera1/robot/image_raw", Image)
        self.image_sub2 = message_filters.Subscriber("/camera2/robot/image_raw", Image)
        ts = message_filters.TimeSynchronizer([self.image_sub1, self.image_sub2], 5)
        print('filter made')
        ts.registerCallback(self.callback)
        
        
    def publish_angles(self, angles):
      #print('Publishing...')
      # Initializes publishers 
      robot_joint2_est = rospy.Publisher("/image1/joint2_position_estimator", Float64, queue_size=10)
      robot_joint3_est = rospy.Publisher("/image1/joint3_position_estimator", Float64, queue_size=10)
      robot_joint4_est = rospy.Publisher("/image1/joint4_position_estimator", Float64, queue_size=10)
      #print(angles[0], angles[1], angles[2])
      # Publishes results
      joint2 = Float64()
      joint2.data = angles[0]
      robot_joint2_est.publish(joint2)
      joint3 = Float64()
      joint3.data = angles[1]
      robot_joint3_est.publish(joint3)
      joint4 = Float64()
      joint4.data = angles[2]
      robot_joint4_est.publish(joint4)
      print(joint2, joint3, joint4)

    # In this method you can focus on detecting the centre of the red circle
    def detect_red(self, image):
        # Isolate the blue colour in the image as a binary image
        mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
        # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        # Obtain the moments of the binary image
        M = cv2.moments(mask)
        # Calculate pixel coordinates for the centre of the blob
        if M['m00'] == 0.0: # if red is occluded
        # TODO: Maybe if occluded cx and cy should fall back on the previous values? Would need some restructuring for that though
          cx = 0.0
          cy = 0.0
        else:
          cx = int(M['m10'] / M['m00'])
          cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])

    # Detecting the centre of the green circle
    def detect_green(self, image):
        mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])

    # Detecting the centre of the blue circle
    def detect_blue(self, image):
        mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])

    # Detecting the centre of the yellow circle
    def detect_yellow(self, image):
        mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])
        
    def detect_orange(self, image):
        # TODO: Check that this is working correctly and detecting the correct color
        # TODO: Add function to distinguish target via Chamfer matching (presumably)
        # TODO: Copy coordinate functions to get coordinates of target
        mask = cv2.inRange(image, (0, 50, 110), (50, 200, 220))
        cv2.imshow('window2', mask)
        return mask
        

    # Calculate the conversion from pixel to meter
    def pixel2meter(self, image):
        # Obtain the centre of each coloured blob
        circle1Pos = self.detect_blue(image)
        circle2Pos = self.detect_green(image)
        dist = np.sum((circle1Pos - circle2Pos) ** 2)
        return 3.5 / np.sqrt(dist)
        

    # Calculate the relevant joint vectors from the image
    def detect_joint_angles(self, image):
        a = self.pixel2meter(image)
        b = self.pixel2meter(image)
        
        # Obtain the centre of each coloured blob
        center = a * self.detect_yellow(image)
        circle1Pos = a * self.detect_blue(image)
        circle2Pos = a * self.detect_green(image)
        circle3Pos = a * self.detect_red(image)
        
        # Get vector for each joint
        ytob = center - circle1Pos
        btog = center - circle2Pos
        gtor = center - circle3Pos

        # Return 2d coordinates
        return np.array([ytob[0], ytob[1], btog[0], btog[1], gtor[0], gtor[1], center[0], center[1]])
        
        
    def get_3d_coords(self, image1, image2):
    
        # Get 2d coordinates
        self.image1_coords = self.detect_joint_angles(image1)
        self.image2_coords = self.detect_joint_angles(image2)
        
        # Get 3d coordinates
        # TODO: how to get z-coord???
        self.yellow = np.array([0,0,0])
        self.blue = np.array([self.image1_coords[0], self.image2_coords[1], max(self.image1_coords[1], self.image2_coords[0])])
        self.green = np.array([self.image1_coords[2], self.image2_coords[3], max(self.image1_coords[3], self.image2_coords[2])])
        self.red = np.array([self.image1_coords[4], self.image2_coords[5], max(self.image1_coords[5], self.image2_coords[4])])
        
        # Return 3d coordinates
        return [self.blue, self.green, self.red]
        
        
    def get_angles(self, image1, image2):
        # TODO: ANGLES
        self.coords = self.get_3d_coords(image1, image2)
        return self.coords
        

    # Recieve data, process it, and publish
    def callback(self, image1, image2):
        print('Running callback')
        # Recieve the image
        try:
            cv_image1 = self.bridge.imgmsg_to_cv2(image1, "bgr8")
            cv_image2 = self.bridge.imgmsg_to_cv2(image2, "bgr8")
            print('Images received')
        except CvBridgeError as e:
            print(e)

        #cv2.imshow('window', cv_image)
        cv2.waitKey(3)

        #self.joints = Float64MultiArray()
        #self.joints.data = a

        # Publish the results
        try:
            #self.detect_orange(cv_image1)
            self.publish_angles(self.get_angles(cv_image1, cv_image2))
            #self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            #self.joints_pub.publish(self.joints)
        except CvBridgeError as e:
            print(e)


# call the class
def main():
    print('Running main...')
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main()
