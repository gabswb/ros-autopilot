#!/usr/bin/env python
import sys

import numpy as np
import cv2
import rospy
import yaml
from ultralytics import YOLO

from perception.msg import ObjectBoundingBox

MODEL_INPUT_WIDTH = MODEL_INPUT_HEIGHT = 416

class ObjectDetector(object):
    def __init__(self, config, yolov8l):
        self.config = config
        self.yolov8l = yolov8l
        
        self.bbox_topic = self.config["topic"]["object-bbox"]
        self.yolov8n_path = self.config['model']['yolov8n-path']
        self.yolov8l_path = self.config['model']['yolov8l-path']
        self.image_topic_width = self.config['topic']['forward-camera-width']
        self.image_topic_height = self.config['topic']['forward-camera-height']

        # read pre-trained model and config file
        self.net = None
        if self.yolov8l:
            self.net = YOLO(self.yolov8l_path)
        else:
            self.net = YOLO(self.yolov8n_path)

        self.confidence_threshold = 0.4
        rospy.loginfo("Object detector ready")	

    def create_bbox_msg(self, boxes, class_id, instance_id = 0):
        msg_bbox = ObjectBoundingBox()
        msg_bbox.x = boxes[0]
        msg_bbox.y = boxes[1]
        msg_bbox.w = boxes[2]
        msg_bbox.h = boxes[3]
        msg_bbox.class_id = class_id
        msg_bbox.instance_id = instance_id
        return msg_bbox
    
    def detect(self, img):
        """Detect objects in the image"""
        bbox_list = []

        # yolov8 processing
        outputs = self.net.track(img, persist=True, verbose=False)
        for box in outputs[0].boxes:
            if box.conf > self.confidence_threshold:
                xywh = box.xywh.int().cpu().tolist()[0]
                xywh[0] = xywh[0]-xywh[2]//2
                xywh[1] = xywh[1]-xywh[3]//2
                bbox_list.append(self.create_bbox_msg(xywh, box.cls.int().cpu().tolist()[0], box.id.int().cpu().tolist()[0] if box.id is not None else 0))
              
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


