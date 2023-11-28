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
    def __init__(self, config, yolov5, yolov8l, yolov8n):
        self.config = config
        self.bbox_topic = self.config["node"]["object-bbox-topic"]
        self.yolov4_config_path = self.config["model"]["yolov4-config-path"]
        self.yolov4_weights_path = self.config["model"]["yolov4-weights-path"]
        self.yolov5_onnx_path = self.config['model']['yolov5n-onnx-model-path']
        self.image_topic_width = self.config['node']['image-topic-width']
        self.image_topic_height = self.config['node']['image-topic-height']
        self.yolov5 = yolov5
        self.yolov8l = yolov8l
        self.yolov8n = yolov8n
        self.yolov4 = not(self.yolov5 or self.yolov8l or self.yolov8n)

        # read pre-trained model and config file
        self.net = None
        if self.yolov5:
            self.net = cv2.dnn.readNetFromONNX(self.yolov5_onnx_path)
        elif self.yolov8l:
            self.net = YOLO("src/models/yolov8/yolov8l.pt")
        elif self.yolov8n:
            self.net = YOLO("src/models/yolov8/yolov8n.pt")
        else:
            self.net = cv2.dnn.readNet(self.yolov4_weights_path, self.yolov4_config_path)

        if self.yolov5 or self.yolov4:
            # get output layer names
            self.output_layers = self.net.getUnconnectedOutLayersNames()
            self.nms_threshold = 0.45
            # object association variables
            self.confidence_matching = 300
            self.kalman_filter_persistence = 2 # seconds
            self.obj_id = 0
            self.kalman_filters = []

        self.confidence_threshold = 0.4

        rospy.loginfo("Object detector ready")	

    def kalman_init(self, measurement, h , w):
        k = cv2.KalmanFilter(6, 2) # Position-velocity-acceleration state model
        k.measurementMatrix = np.array([    
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], np.float32) # we only measure x and y, not the velocities
        k.measurementNoiseCov = np.array([ 
            [0.1, 0],
            [0, 0.1]
        ], np.float32) # we trust our measurements 
        k.transitionMatrix = np.array([
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        k.processNoiseCov = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], np.float32) * 1e-1 # for fast model adaptation
        k.statePost = np.array([measurement[0],measurement[1],[0],[0],[0],[0]], np.float32)
        self.obj_id += 1
        return {
            'instance_id': self.obj_id,
            'filter': k,
            'last_seen': rospy.get_time(),
            'h': h,
            'w': w,
        }
    
    def create_bbox_msg(self, boxes, class_id, instance_id = None):
        msg_bbox = ObjectBoundingBox()
        msg_bbox.x = boxes[0]
        msg_bbox.y = boxes[1]
        msg_bbox.w = boxes[2]
        msg_bbox.h = boxes[3]
        msg_bbox.class_id = class_id
        msg_bbox.instance_id = instance_id
        return msg_bbox
    
    def yolov5_output_to_bbox(self, outputs):
        class_ids = []
        confidences = []
        boxes = []
        # Rows.
        rows = outputs[0].shape[1]
        # Resizing factor.
        x_factor = self.image_topic_width / MODEL_INPUT_WIDTH
        y_factor =  self.image_topic_height / MODEL_INPUT_HEIGHT

        # Iterate through detections.
        for r in range(rows):
                row = outputs[0][0][r]
                confidence = row[4]
                # Discard bad detections and continue.
                if confidence >= self.confidence_threshold:
                    classes_scores = row[5:]
                    # Get the index of max class score.
                    class_id = np.argmax(classes_scores)
                    #  Continue if the class score is above threshold.
                    if classes_scores[class_id] > 0.5:
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        left = int((cx - w/2) * x_factor)
                        top = int((cy - h/2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        if left < self.image_topic_width and top < self.image_topic_height:
                            confidences.append(confidence)
                            class_ids.append(class_id)
                            box = np.array([left, top, width, height])
                            boxes.append(box)
                        else:
                            rospy.loginfo(f"Detection out of bounds: {left}, {top}, {width}, {height}")
        
        return boxes, class_ids, confidences

    def yolov4_output_to_bbox(self, outputs):
        class_ids = []
        confidences = []
        boxes = []
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * self.image_topic_width)
                    center_y = int(detection[1] * self.image_topic_height)
                    w = np.uint32(detection[2] * self.image_topic_width)
                    h = np.uint32(detection[3] * self.image_topic_height)
                    x = np.uint32(center_x - w / 2)
                    y = np.uint32(center_y - h / 2)
                    if x < self.image_topic_width and y < self.image_topic_height:
                        class_ids.append(np.uint32(class_id))
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])
                    else:
                        rospy.loginfo(f"Detection out of bounds: {x}, {y}, {w}, {h}")

        return boxes, class_ids, confidences

    def detect(self, img):
        """Detect objects in the image"""
        bbox_list = []

        if self.yolov8l or self.yolov8n:
            # yolov8 processing
            outputs = self.net.track(img, persist=True, verbose=False)
            for box in outputs[0].boxes:
                if box.conf > self.confidence_threshold:
                    # rospy.loginfo(f"box:{box.xywh.int().cpu().tolist()}")
                    # rospy.loginfo(f"cls:{box.cls.int().cpu().tolist()}")
                    # if box.id is not None:
                    #     rospy.loginfo(f"id:{box.id.int().cpu().tolist()}")
                    # else:
                    #     rospy.loginfo(f"id:None")
                    xywh = box.xywh.int().cpu().tolist()[0]
                    xywh[0] = xywh[0]-xywh[2]//2
                    xywh[1] = xywh[1]-xywh[3]//2
                    bbox_list.append(self.create_bbox_msg(xywh, box.cls.int().cpu().tolist()[0], box.id.int().cpu().tolist()[0] if box.id is not None else None))
        else:
            # yolov4 and yolov5 processing
            blob = cv2.dnn.blobFromImage(img, 0.00392, (MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH), (0,0,0), True, crop=False)
            # set input blob for the network
            self.net.setInput(blob)
            # run inference through the network and gather predictions from output layers
            outputs = self.net.forward(self.output_layers)
            boxes, class_ids, confidences = self.yolov5_output_to_bbox(outputs) if self.yolov5 else self.yolov4_output_to_bbox(outputs)     
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
            # go through the detections remaining after nms and draw bounding box
            for i in indices:
                bbox_list.append(self.create_bbox_msg(boxes[i], class_ids[i]))
        
        return bbox_list

    def bbox_associations_and_predictions(self, bbox_list):
        """Associate bounding boxes with Kalman filter and predict bounding boxes when no measurement is available"""     
        predictions = [] 

        for k in self.kalman_filters:
            # remove old filters
            if rospy.get_time() - k['last_seen'] > self.kalman_filter_persistence:
                # rospy.loginfo(f"Removing filter {k['instance_id']}")
                self.kalman_filters.remove(k)
                continue
            # predict
            x, y, _, _, _, _ = k['filter'].predict()
            predictions.append(np.array([x, y], np.float32))
        predictions = np.array(predictions)
        # rospy.loginfo(f"p:{predictions}")

        measurements = np.zeros((len(bbox_list), 2, 1), np.float32)
        for i, bbox in enumerate(bbox_list):
            measurements[i] = np.array([[bbox.x+bbox.w/2], [bbox.y+bbox.h/2]], np.float32)
        # rospy.loginfo(f"m:{measurements}")

        if len(predictions) == 0:
            # initialize kalman filters with measurements
            for i in range(len(bbox_list)):
                self.kalman_filters.append(self.kalman_init(measurements[i], bbox_list[i].h, bbox_list[i].w))
                bbox_list[i].instance_id = self.kalman_filters[-1]['instance_id']
        elif len(bbox_list) > 0:
            # Find best matches between measurements and predictions
            distances = np.linalg.norm(
                measurements.reshape(measurements.shape[0], measurements.shape[1])[:, np.newaxis, :]
                - predictions.reshape(predictions.shape[0], predictions.shape[1])[np.newaxis, :, :]
                , axis=2)
            # rospy.loginfo(f"distances:{distances}")
            sort_axis = int(len(predictions) > len(measurements))
            matches = distances.argsort(axis=sort_axis)
            # rospy.loginfo(f"matches:{matches}")
            best_matches = [[i, j] for i in range(matches.shape[0]) for j in range(matches.shape[1]) if matches[i][j] == 0]
            measurements_affected = [False]*measurements.shape[0]
            # rospy.loginfo(f"best_matches:{best_matches}")

            # correct filter with closest measurement
            for m in best_matches:
                # rospy.loginfo(f"best match distance : {distances[m[0]][m[1]]}")
                measurements_affected[m[0]] = True
                if distances[m[0]][m[1]] < self.confidence_matching:
                    # rospy.loginfo(f"meas:{measurements[m[0]].dtype}")
                    self.kalman_filters[m[1]]['filter'].correct(measurements[m[0]])
                    bbox_list[m[0]].instance_id = self.kalman_filters[m[1]]['instance_id']
                    self.kalman_filters[m[1]]['last_seen'] = rospy.get_time()
                    self.kalman_filters[m[1]]['h'] = bbox_list[m[0]].h
                    self.kalman_filters[m[1]]['w'] = bbox_list[m[0]].w
                else:
                    self.kalman_filters.append(self.kalman_init(measurements[m[0]], bbox_list[m[0]].h, bbox_list[m[0]].w))
                    bbox_list[m[0]].instance_id = self.kalman_filters[-1]['instance_id']
        
            if len(predictions) < len(measurements):
                # more measurements than kalman filter = create new kalman filters for unmatched measurements
                measurement_to_affect =  [i for i in range(len(measurements_affected)) if not measurements_affected[i]]
                for m in measurement_to_affect:
                    self.kalman_filters.append(self.kalman_init(measurements[m], bbox_list[m].h, bbox_list[m].w))
                    bbox_list[m].instance_id = self.kalman_filters[-1]['instance_id']
            elif len(predictions) > len(measurements):
                # more kalman filters than measurements = create new objects for unmatched kalman filters
                predictions_to_affect =  [i for i in range(len(predictions)) if i not in np.array(best_matches)[:,1]]
                # print(f'predictions_to_affect:{predictions_to_affect}')
                for p in predictions_to_affect:
                    # rospy.loginfo('Prediction: {}, {}'.format(int(predictions[p][0][0]), int(predictions[p][1][0])))
                    boxes = [
                        int(predictions[p][0][0]-self.kalman_filters[p]['w']/2),
                        int(predictions[p][1][0]-self.kalman_filters[p]['h']/2),
                        self.kalman_filters[p]['w'],
                        self.kalman_filters[p]['h']
                    ]
                    # print(f'boxes:{boxes}')
                    bbox_list.append(
                        self.create_bbox_msg(boxes, 80, self.kalman_filters[p]['instance_id'])
                    )
        else:
            # no measurement -> predict
            for k in self.kalman_filters:
                x, y, _, _, _, _ = k['filter'].predict()
                # rospy.loginfo('Prediction: {}, {}'.format(x, y))
                bbox_list.append(
                    self.create_bbox_msg([int(x-k['w']/2),int(y-k['h']/2),k['w'],k['h']], 80, k['instance_id'])
                )

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


