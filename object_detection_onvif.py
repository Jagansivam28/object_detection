######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a webcam feed.
# It draws boxes, scores, and labels around the objects of interest in each frame
# from the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
c = 'person'
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from onvif import *

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph1.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 3

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)


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

# Initialize webcam feed
mycam = ONVIFCamera('rtsp://192.168.55.177:554', 80, 'user', 'User'0)
media_controll = mycam.create_media_service()
print('cam initialising')
cam_config = media_controll.GetProfiles()[0].token
stream_uri_obj = media_controll.create_type("GetStreamUri")
stream_uri_obj.ProfileToken = cam_config
stream_uri_obj.StreamSetup = {"Stream" : "RTP-Unicast", "Transport" : {"Protocol" : "RTSP"}}
connection_data = media_controll.ws_client.GetStreamUri(stream_uri_obj.StreamSetup, stream_uri_obj.ProfileToken)
video = cv2.VideoCapture('rtsp://user:User@192.168.55.117:554/1/')
#fps=video.set(cv2.CAP_PROP_FPS, 10)
#ret = video.set(3,1280)
#ret.set(CV_CAP_PROP_FRAME_WIDTH,640)
#ret.set(CV_CAP_PROP_FRAME_HEIGHT,480)
#ret = video.set(4,720)
#fps = video.get(cv2.CAP_PROP_FPS)
#print(fps)

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=-0)
    print ("camera running") 

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor:frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=1,
        min_score_thresh=0.60)
    #####################
    final_score = np.squeeze(scores)
    count = 0
    for i in range(100):
            if scores is None or final_score[i] > 0.5:
                count = count + 1
    a = count
    printcount = 0;
    for i in classes[0]:
            printcount = printcount +1
            try:b = category_index[int(i)]['name']
            except:pass
        
            if(printcount == count):
                break


##################


# All the results have been drawn on image. Now display the image.
    cv2.imshow('Object detector', frame)
    if b == c:
        print("Person is entering into a restricted zone")
    else:
        print("Tested OK")

    # All the results have been drawn on the frame, so it's time to display it.
	

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        

# Clean up
video.release()
cv2.destroyAllWindows()
