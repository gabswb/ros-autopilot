#!/usr/bin/env python
import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from object_detection.msg import ObjectBoundingBox


class ObjectDetector(object):
    def __init__(self, visualize=False):
        self.classes = None
        with open("src/object_detection/yolov4-tiny/classes.names", 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # read pre-trained model and config file
        self.net = cv2.dnn.readNet("src/object_detection/yolov4-tiny/yolov4-tiny.weights", "src/object_detection/yolov4-tiny/yolov4-tiny.cfg")
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        self.cvbridge = CvBridge()

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.image_subscriber = rospy.Subscriber("/forwardCamera/image_raw",Image,self.detect)
        self.object_detection_publisher = rospy.Publisher("/object_bounding_box", ObjectBoundingBox, queue_size=10)

        if visualize:
            self.visualization_publisher = rospy.Publisher("/object_detector_viz_topic", Image, queue_size=10)

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def detect(self, data):
        # create input blob
        img = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, 3))
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
        
        for i in range(len(boxes)):
            for i in indices:
                x, y, w, h = boxes[i]
                obb = ObjectBoundingBox()
                obb.class_id = class_ids[i]
                obb.x = x
                obb.y = y
                obb.w = w
                self.object_detection_publisher.publish(obb)
                
                if self.visualization_publisher:
                    self.draw_bounding_box(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        # display output image   
        if self.visualization_publisher:
            self.visualization_publisher.publish(self.cvbridge.cv2_to_imgmsg(img, encoding="passthrough"))
        

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    # print(f'pwd: {os.getcwd()}')
    od = ObjectDetector(visualize=True)
    rospy.spin()


