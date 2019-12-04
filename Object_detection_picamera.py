######## Picamera Object Detection Using Tensorflow Classifier #########
#
# Author: Evan Juras
# Date: 4/15/18
# Description: 
# This program uses a TensorFlow classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from
# the Picamera. It also can be used with a webcam by adding "--usbcam"
# when executing this script from the terminal.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys


#servo control imports and vars
import RPi.GPIO as GPIO
from time import sleep
from Object_detection_servo_GPIO import SetPW


GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(3, GPIO.OUT)
pwm = GPIO.PWM(3, 50)
pwm.start(0)
GPIO.setwarnings(False)
#pwm.ChangeDutyCycle(0)
###########################

# Set up camera constants
#IM_WIDTH = 1280
#IM_HEIGHT = 720
IM_WIDTH = 640    #Use smaller resolution for
IM_HEIGHT = 480   #slightly faster framerate

k_counter = 0
f_counter = 0
s_counter = 0     #spoon

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','backup2.pbtx')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 2
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX
previous_pw = 0

# Initialize camera and perform object detection.
#sleep(1)

SetPW(2.2)
SetPW(0)# overflow
print("overflow")
sleep(1)
SetPW(1.8)
SetPW(0)
print("fork")
sleep(1)
SetPW(1.5)
SetPW(0)
print("knife")
sleep(1)
SetPW(1.2)
print("spoon")
sleep(1)
#SetPW(2.2)
#print("back to overflow")

if camera_type == 'picamera':
    
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        
        """
        print("classes")
        print(classes[0][0])
        print("boxes")
        print(boxes[0][0])
        print("scores")
        print(scores[0][0])
        print("num detections")
        print(num)
        
        box_index = -1
        # get number of items detected - 1
        for i in classes[0][0]:
            if i == 48 or i == 49 or i == 50:
                box_index = box_index + 1
            # need to figure out how to deal with misidentified objects
                
        # get first detected object        
        first_detected = classes[0][box_index]
        
        #object with the largest boxes[][][] value gets priority
        largest_val = boxes[0][box_index][2]
        
        for i in boxes[0][box_index]:
            if boxes[0][box_index][2] > largest_val:
                first_detected = classes[0][box_index]
                largest_val = boxes[0][box_index][2]
            # decrement box_index
            box_index = box_index - 1
            #don't need to access array element [-1]
            if box_index = -1:
                break
        """
        #idea 2
        
        cnt = 0
        for i in classes[0]:
            if i != 48 and i != 49 and i != 50 and i != 1:
                classes[0][cnt] = 1
            cnt += 1
        
        #following logic not necessary if objects are spaced at proper intervals
        
        first_detected = classes[0][0] #first detected defaults to first element in array
        largest_val = boxes[0][0][2] #largest value defaults to first element in array
        
        counter = 0
        
        for i in classes[0]:
            if i == 48 or i ==49 or i == 50:
                if boxes[0][counter][2] > largest_val:
                    print("reached if statement")
                    print("largest value: ")
                    print(boxes[0][counter][2])
                    print()
                    largest_val = boxes[0][counter][2]
                    first_detected = i
            counter = counter + 1
                
        print("first detected: ")
        print(first_detected)
        
        wait_flag = 0
        
        if 48 == classes[0][0] and scores[0][0] > 0.4:
            if previous_pw != 1.8:
                SetPW(1.8)
                #SetPW(0)
                previous_pw = 1.8
            print("found fork")
            print()
            wait_flag = 1
            f_counter = f_counter + 1
            
        elif 49 == classes[0][0] and scores[0][0] > 0.4:
            if previous_pw != 1.5: 
                SetPW(1.5)
                #SetPW(0)
                previous_pw = 1.5
            print("found knife")
            print()
            wait_flag = 1
            k_counter = k_counter + 1
            
            
        elif 50 == classes[0][0] and scores[0][0] > 0.4:
            if previous_pw != 1.2:
                SetPW(1.2)
                #etPW(0)
                previous_pw = 1.2
            print("found spoon")
            print()
            wait_flag = 1
            s_counter = s_counter + 1
            
        else:
            if previous_pw != 2.2:
                SetPW(2.2)
                #SetPW(0)
                previous_pw = 2.2
            print("overflow bin")
            print()
            #wait_flag = 1

        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        #cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        cv2.putText(frame,f"fork count: {f_counter}", (20, 100),font,1,(255, 255, 0),2,cv2.LINE_AA)
        cv2.putText(frame,f"knife count: {k_counter}", (20, 200),font,1,(255, 255, 0),2,cv2.LINE_AA)
        cv2.putText(frame,f"spoon count: {s_counter}", (20, 300),font,1,(255, 255, 0),2,cv2.LINE_AA)

        
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
        
        if wait_flag == 1:
            sleep(5)
        
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

cv2.destroyAllWindows()

