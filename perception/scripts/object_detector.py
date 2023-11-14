#!/usr/bin/env python
import sys

import numpy as np
import cv2
import rospy
import yaml

from perception.msg import Object, ObjectBoundingBox

class ObjectDetector(object):
    def __init__(self, config):
        self.config = config
        self.bbox_topic = self.config["node"]["object-bbox-topic"]
        self.model_config_path = self.config["model"]["detection-model-config-path"]
        self.model_weights_path = self.config["model"]["detection-model-weights-path"]

        # read pre-trained model and config file
        self.net = cv2.dnn.readNet(self.model_weights_path, self.model_config_path)
        self.output_layers = self.net.getUnconnectedOutLayersNames()

        rospy.loginfo("Object detector ready")	


    def detect(self, img):
        # create input blob
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)
        height, width, _ = img.shape

        # set input blob for the network
        self.net.setInput(blob)
        # run inference through the network and gather predictions from output layers
        outs = self.net.forward(self.output_layers)

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # for each detetion from each output layer get the confidence, class id, bounding box params and ignore weak detections (confidence < 0.3)
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
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    class_ids.append(int(class_id))
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # go through the detections remaining after nms and draw bounding box

        bbox_list = []
        for i in range(len(boxes)):
            for i in indices:
                x, y, w, h = boxes[i]
                msg_bbox = ObjectBoundingBox()
                msg_bbox.x = x
                msg_bbox.y = y
                msg_bbox.w = w
                msg_bbox.h = h
                msg_bbox.class_id = class_ids[i]
                bbox_list.append(msg_bbox)
        return bbox_list


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <config-file>")
    else:
        with open(sys.argv[1], "r") as config_file:
            config = yaml.load(config_file, yaml.Loader)
        rospy.init_node('object_detector')
        node = ObjectDetector(config, visualize=True)
        rospy.spin()


