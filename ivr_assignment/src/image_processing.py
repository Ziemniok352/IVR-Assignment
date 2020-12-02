#!/usr/bin/env python3
#print('Starting...')
import roslib
import sys
import rospy
import cv2
import os
import numpy as np
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
from scipy.optimize import least_squares
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
        # initialize a publisher to send end effector position to a topic called end_effector_position
        self.end_effector_pos = rospy.Publisher("end_effector_pos", Float64MultiArray, queue_size=10)
        # initialize a publisher to send target position to a topic called target_pos
        self.target_pos = rospy.Publisher("target_pos", Float64MultiArray, queue_size=10)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        # initialize a subscribers and time synchronizer
        self.image_sub1 = message_filters.Subscriber("/camera1/robot/image_raw", Image)
        self.image_sub2 = message_filters.Subscriber("/camera2/robot/image_raw", Image)
        ts = message_filters.TimeSynchronizer([self.image_sub1, self.image_sub2], 5)
        ts.registerCallback(self.callback)
        
        
    def publish_angles(self, angles):
      # Initializes publishers 
      robot_joint2_est = rospy.Publisher("/image1/joint2_position_estimator", Float64, queue_size=10)
      robot_joint3_est = rospy.Publisher("/image1/joint3_position_estimator", Float64, queue_size=10)
      robot_joint4_est = rospy.Publisher("/image1/joint4_position_estimator", Float64, queue_size=10)

      # Publishes results
      joint2 = Float64()
      joint2.data = self.angles[0] * -1
      robot_joint2_est.publish(joint2)
      joint3 = Float64()
      joint3.data = self.angles[1] * -1
      robot_joint3_est.publish(joint3)
      joint4 = Float64()
      joint4.data = self.angles[2] * -1
      robot_joint4_est.publish(joint4)
      #print(joint2, joint3, joint4)

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
        if M['m00'] == 0.0: 
          cx = 0.0
          cy = 0.0
        else:
          cx = int(M['m10'] / M['m00'])
          cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])

    # Detecting the centre of the blue circle
    def detect_blue(self, image):
        mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        if M['m00'] == 0.0: 
          cx = 0.0
          cy = 0.0
        else:
          cx = int(M['m10'] / M['m00'])
          cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])

    # Detecting the centre of the yellow circle
    def detect_yellow(self, image):
        mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        if M['m00'] == 0.0: 
          cx = 0.0
          cy = 0.0
        else:
          cx = int(M['m10'] / M['m00'])
          cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])
        
    def detect_orange(self, image):
        mask = cv2.inRange(image, (0, 40, 100), (100, 100, 255))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        return mask
        
    def detect_target(self, image1, image2, template, version=1):
        # Use version 1 to detect sphere, version 2 to detect box
        a1 = self.pixel2meter(image1)
        a2 = self.pixel2meter(image2)
               
        # Chamfer matching
        masked1 = self.detect_orange(image1)
        masked2 = self.detect_orange(image2)
        matched1 = cv2.matchTemplate(masked1, template, 1)
        matched2 = cv2.matchTemplate(masked2, template, 1)
        coord1x, coord1y = cv2.minMaxLoc(matched1)[2]
        coord2x, coord2y = cv2.minMaxLoc(matched2)[2]
        
        # Scale to images
        center1 = self.detect_yellow(image1) * a1
        center2 = self.detect_yellow(image2) * a2
        coords1 = np.array([coord1x, coord1y])
        coords2 = np.array([coord2x, coord2y])
        coords1 = coords1 * a1
        coords1 = center1 - coords1
        coords1 = coords1 * np.array([-1,1])
        coords2 = coords2 * a2
        coords2 = center2 - coords2
        coords2 = coords2 * np.array([-1,1])
        
        # Coordinates
        if version == 1:
            coords = np.array([coords1, coords2])
            self.target = np.array([coords[1][0], coords[0][0], np.mean([coords[0][1], coords[1][1]])])
            print('sphere: ' + str(self.target))
        
            # Publish results
            self.target_end_effector_pos()
            
        elif version == 2:
            coords = np.array([coords1, coords2])
            self.box = np.array([coords[1][0], coords[0][0], np.mean([coords[0][1], coords[1][1]])])
            print('box:', str(self.box))
            self.publish_box_pos()
            
        return self.target
        

    # Calculate the conversion from pixel to meter
    def pixel2meter(self, image):
        # Obtain the centre of each coloured blob
        circle1Pos = self.detect_yellow(image)
        circle2Pos = self.detect_blue(image)
        dist = np.sum((circle1Pos - circle2Pos) ** 2)
        return 2.5 / np.sqrt(dist)
        

    # Calculate the relevant joint vectors from the image
    def detect_joint_locations(self, image):
        a = self.pixel2meter(image)
        
        # Obtain the centre of each coloured blob
        center = a * self.detect_yellow(image)
        circle1Pos = a * self.detect_blue(image)
        circle2Pos = a * self.detect_green(image)
        #circle3Pos = a * self.detect_red(image)
        
        # Get position with respect to yellow
        blue_to_yellow = center - circle1Pos
        blue_to_yellow = blue_to_yellow * np.array([-1,1])
        green_to_yellow = center - circle2Pos
        green_to_yellow = green_to_yellow * np.array([-1,1])
        
        # Get end effector position
        circle3Pos = self.detect_red(image)
        if circle3Pos[0] == 0 and circle3Pos[1] == 0:
            red_to_yellow = circle3Pos
        else:
            red_to_yellow = a * (center - circle3Pos)
            red_to_yellow = red_to_yellow * np.array([-1,1])
        

        # Return 2d coordinates
        return np.array([center, blue_to_yellow, green_to_yellow, red_to_yellow])
        
        
    def get_3d_coords(self, image1, image2):
    
        # Get 2d coordinates
        self.image1_coords = self.detect_joint_locations(image1)
        self.image2_coords = self.detect_joint_locations(image2)
        
        # Get 3d coordinates
        # If yellow's angle is fixed, so are the coords of yellow and blue
        self.yellow = np.array([0,0,0])
        self.blue = np.array([0,0,2])
        
        # Calculate green
        if self.image1_coords[2][0] == 0:
            self.green = np.array([self.image2_coords[2][0], 0, self.image2_coords[2][1]])
        if self.image2_coords[2][0] == 0:
            self.green = np.array([0, self.image1_coords[2][0], self.image1_coords[2][1]])
        if self.image1_coords[2][0] != 0 and self.image2_coords[2][0] != 0:
            self.green = np.array([self.image2_coords[2][0], self.image1_coords[1][0], np.mean([self.image1_coords[2][1], self.image2_coords[2][1]])])
            
        # Calculate red
        if self.image1_coords[3][0] == 0:
            self.red = np.array([self.image2_coords[3][0], self.green[1], self.image2_coords[3][1]])
        if self.image2_coords[3][0] == 0:
            self.red = np.array([self.green[0], self.image1_coords[3][0], self.image1_coords[3][1]])
        if self.image1_coords[3][0] != 0 and self.image2_coords[3][0] != 0:
            self.red = np.array([self.image2_coords[3][0], self.image1_coords[3][0], np.mean([self.image1_coords[3][1], self.image2_coords[3][1]])])       
        
        # Return 3d coordinates
        return np.array([self.yellow, self.blue, self.green, self.red])
        
        
    def get_angles(self, image1, image2):
        self.coords = self.get_3d_coords(image1, image2)
        self.angles = np.array([self.estimate_angle(self.coords[2], self.coords[1], 3.5, 'x'), self.estimate_angle(self.coords[2], self.coords[1], 0, 'y'), self.estimate_angle(self.coords[3], self.coords[2], 3, 'x')])
        return self.angles
        
        
    def estimate_angle(self, coord1, coord2, h, axis):
        a = (coord1-coord2)/np.linalg.norm(coord1)
        b = coord2/np.linalg.norm(coord2)
        
        if axis == 'x':
          try:
            return least_squares(self.x_fun, [0.0], args = (a,b), bounds = (-np.pi/2, np.pi/2)).x
          except:
            return np.array[0.0]
            
        elif axis == 'y':
          try:
            return least_squares(self.y_fun, [0.0], args = (a,b), bounds = (-np.pi/2, np.pi/2)).x
          except:
            return np.array[0.0]
            
        else:
          print('axis must be either \'x\' or \'y\'. ')
          return np.array[0.0]
        
    def x_fun(self, theta, a, b):
        m = np.array([[1,0,0], [0,np.cos(theta),-np.sin(theta)], [0,np.sin(theta),np.cos(theta)]])
        return np.sum(np.abs(m.dot(a) - b))
        
    def y_fun(self, theta, a, b):
        m = np.array([[np.cos(theta),0,-np.sin(theta)], [0,1,0], [np.sin(theta),0,np.cos(theta)]])
        return np.sum(np.abs(m.dot(a) - b))

    def detect_end_effector_pos(self):
        #find end effector coordinates from img and publish in topic(end_effector_pos) so control.py can use it
        robot_end_pos_pub = rospy.Publisher("/image_processing/end_effector_pos", Float64, queue_size=10)
        end_pos = Float64()
        end_pos.data = self.red
        robot_end_pos_pub.publish(end_pos)

    def target_end_effector_pos(self):
        #find target coordinates from img and publish in topic(target_pos_x, _y, and _z) so control.py can use it
        target_pos_pub_x = rospy.Publisher("/image_processing/target_position_x", Float64, queue_size=10)
        target_pos = Float64()
        target_pos.data = self.target[0]
        target_pos_pub_x.publish(target_pos)

        target_pos_pub_y = rospy.Publisher("/image_processing/target_position_y", Float64, queue_size=10)
        target_pos.data = self.target[1]
        target_pos_pub_y.publish(target_pos)
        
        target_pos_pub_z = rospy.Publisher("/image_processing/target_position_z", Float64, queue_size=10)
        target_pos.data = self.target[2]
        target_pos_pub_z.publish(target_pos)
        
        target_pos_pub = rospy.Publisher("/image_processing/target_position", Float64, queue_size=10)
        target_pos = Float64()
        target_pos.data = self.target
        target_pos_pub.publish(target_pos)
        
    def publish_box_pos(self):
        #find target coordinates from img and publish in topic(target_pos_x, _y, and _z) so control.py can use it
        box_pos_pub_x = rospy.Publisher("/image_processing/box_position_x", Float64, queue_size=10)
        box_pos = Float64()
        box_pos.data = self.box[0]
        box_pos_pub_x.publish(box_pos)

        box_pos_pub_y = rospy.Publisher("/image_processing/box_position_y", Float64, queue_size=10)
        box_pos = Float64()
        box_pos.data = self.box[1]
        box_pos_pub_y.publish(box_pos)
        
        box_pos_pub_z = rospy.Publisher("/image_processing/box_position_z", Float64, queue_size=10)
        box_pos = Float64()
        box_pos.data = self.box[2]
        box_pos_pub_z.publish(box_pos)
        
        box_pos_pub = rospy.Publisher("/image_processing/box_position", Float64, queue_size=10)
        box_pos = Float64()
        box_pos.data = self.box
        box_pos_pub.publish(box_pos)        
        
        
    # Recieve data, process it, and publish
    def callback(self, image1, image2):
        # Recieve the image
        try:
            cv_image1 = self.bridge.imgmsg_to_cv2(image1, "bgr8")
            cv_image2 = self.bridge.imgmsg_to_cv2(image2, "bgr8")
            imgPath = os.path.join(os.getcwd(), '/home/tully/catkin_ws/src/ivr_assignment/src/template.png')
            template = cv2.imread(imgPath, 1)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            imgPath2 = os.path.join(os.getcwd(), '/home/tully/catkin_ws/src/ivr_assignment/src/template2.png')
            template2 = cv2.imread(imgPath, 1)
            template2 = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
            print('Images loaded!')
        except CvBridgeError as e:
            print(e)

        cv2.waitKey(3)

        # Publish the results
        try:
            self.publish_angles(self.get_angles(cv_image1, cv_image2))
            # Target detection currently prints output as well for debug purposes
            self.detect_target(cv_image1, cv_image2, template)
            self.detect_target(cv_image1, cv_image2, template2, version=2)
            self.detect_end_effector_pos()
            
        except CvBridgeError as e:
            print(e)
        
        


# call the class
def main():
    #print('Running main...')
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main()
