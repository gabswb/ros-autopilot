#!/usr/bin/env python
import sys

import numpy as np
import cv2
import rospy
import yaml
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from perception.msg import ObjectBoundingBox



class ObjectDetector(object):
    def __init__(self, config, publish=False, visualize=False):
        self.config = config
        self.image_topic = self.config["node"]["image-topic"]
        self.bbox_topic = self.config["node"]["object-bbox-topic"]
        self.object_detection_viz_topic = self.config["node"]["object-detection-viz-topic"]
        self.model_config_path = self.config["model"]["detection-model-config-path"]
        self.model_weights_path = self.config["model"]["detection-model-weights-path"]
        self.model_class_names_path = self.config["model"]["detection-model-class-names-path"]
        
        self.publish = publish

        self.classes = None
        with open(self.model_class_names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # read pre-trained model and config file
        self.net = cv2.dnn.readNet(self.model_weights_path, self.model_config_path)
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        self.cvbridge = CvBridge()

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        if self.publish:
            self.image_subscriber = rospy.Subscriber(self.image_topic, Image, self.detect)
            self.object_detection_publisher = rospy.Publisher(self.bbox_topic, ObjectBoundingBox, queue_size=10)

        if visualize:
            self.visualization_publisher = rospy.Publisher(self.object_detection_viz_topic, Image, queue_size=10)

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def detect(self, data):
        # create input blob
        img = None
        if self.publish:
            img = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, 3))
        else:
            img =data

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)
        height, width, _ = img.shape

        # set input blob for the network
        self.net.setInput(blob)
        # run inference through the network
        # and gather predictions from output layers
        outs = self.net.forward(self.output_layers)

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.3)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    class_ids.append(int(class_id))
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # go through the detections remaining
        # after nms and draw bounding box
        rospy.loginfo(f'detection') 

        obj_bboxes = []
        
        for i in range(len(boxes)):
            for i in indices:
                x, y, w, h = boxes[i]
                if self.publish:
                    obb = ObjectBoundingBox()
                    obb.class_id = class_ids[i]
                    obb.x = x
                    obb.y = y
                    obb.w = w
                    self.object_detection_publisher.publish(obb)
                else:
                    obb = {}
                    obb["class_id"] = class_ids[i]
                    obb["x"] = x
                    obb["y"] = y
                    obb["w"] = w
                    obj_bboxes.extend(obb)
                
                if self.visualization_publisher:
                    self.draw_bounding_box(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        #display output image   
        if self.visualization_publisher:
            self.visualization_publisher.publish(self.cvbridge.cv2_to_imgmsg(img, encoding="passthrough"))
        
        if not self.publish:
            return obb

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <config-file>")
    else:
        with open(sys.argv[1], "r") as config_file:
            config = yaml.load(config_file, yaml.Loader)
    rospy.init_node('object_detector')
    node = ObjectDetector(config, visualize=True)
    rospy.spin()


