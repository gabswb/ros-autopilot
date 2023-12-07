import cv2 as cv
import numpy as np
import rospy
from math import sqrt


HAS_LIGHT = (2, 5, 6, 7, 8)

BLINK_HEIGHT = [0.3, 0.55]
LEFT_BLINK = [0, 0.3]
RIGHT_BLINK = [0.7, 1]
BLINK_TO_TURN_ON = 2
BLINK_TO_TURN_OFF = 6

class VehicleLightsDetector(object):
    def __init__(self):
        self.previous_obj_list = None
        self.previous_image = None
        self.blink_dict_history = {}
        rospy.loginfo("Vehicle light detector initialized")		

    def check_vehicle_lights_on(self, previous_img, previous_obj, current_image, current_obj):
        new_image = np.zeros((current_image.shape[0], current_image.shape[1]), dtype=np.uint8)

        light_blink = {}

        for obj in current_obj:
            if obj.bbox.class_id in HAS_LIGHT:
                corresponding_previous_bbox = [prev_obj for prev_obj in previous_obj if
                                            (prev_obj.bbox.class_id == obj.bbox.class_id and
                                                prev_obj.bbox.instance_id == obj.bbox.instance_id)]

                if len(corresponding_previous_bbox) >= 1:
                    dst_corners, current_diagonal = self.get_bbox_corners(obj.bbox)
                    src_corners, previous_diagonal = self.get_bbox_corners(corresponding_previous_bbox[0].bbox)


                    M = cv.getPerspectiveTransform(src_corners, dst_corners)

                    previous_isolate_bbox = self.isolate_bbox(previous_img, corresponding_previous_bbox[0].bbox)
                    current_isolate_bbox = self.isolate_bbox(current_image, obj.bbox)

                    previous_isolate_bbox = cv.warpPerspective(
                        previous_isolate_bbox, M, (previous_isolate_bbox.shape[1], previous_isolate_bbox.shape[0]))

                    x1, y1, x2, y2 = np.int32((dst_corners[0][0], dst_corners[0][1],
                                            dst_corners[2][0], dst_corners[2][1]))

                    image_to_analyse = cv.absdiff(current_isolate_bbox[y1:y2, x1:x2], previous_isolate_bbox[y1:y2, x1:x2])
                    image_hsv = cv.cvtColor(image_to_analyse, cv.COLOR_RGB2HSV)
                    th_image = cv.inRange(image_hsv, (0, 43, 140), (255, 255, 255))

                    if 0.96 * previous_diagonal <= current_diagonal <= 1.04 * previous_diagonal:
                        light_blink[obj.bbox.instance_id] = self.check_light_blink(th_image)
                    else:
                        light_blink[obj.bbox.instance_id] = [False, False]

                        new_image[y1:y2, x1:x2] = th_image

        return light_blink, new_image


    def isolate_bbox(self, image_with_bbox, bbox):
        black_image = np.zeros_like(image_with_bbox, dtype=np.uint8)
        x = bbox.x
        y = bbox.y
        x_plus_w = x + bbox.w
        y_plus_h = y + bbox.h

        black_image[y:y_plus_h, x:x_plus_w] = image_with_bbox[y:y_plus_h, x:x_plus_w]

        return black_image


    def get_bbox_corners(self, bbox):
        x = bbox.x
        y = bbox.y
        x_plus_w = x + bbox.w
        y_plus_h = y + bbox.h

        l_x = x_plus_w - x
        l_y = y_plus_h - y

        diagonal = sqrt(l_x ** 2 + l_y ** 2)

        corners = np.float32([(x, y), (x_plus_w, y), (x_plus_w, y_plus_h), (x, y_plus_h)]), diagonal

        return corners


    def check_light_blink(self, img):
        y_min = int(img.shape[0] * BLINK_HEIGHT[0])
        y_max = int(img.shape[0] * BLINK_HEIGHT[1])

        x_min_left = int(img.shape[1] * LEFT_BLINK[0])
        x_max_left = int(img.shape[1] * LEFT_BLINK[1])

        x_min_right = int(img.shape[1] * RIGHT_BLINK[0])
        x_max_right = int(img.shape[1] * RIGHT_BLINK[1])

        blinks_on = []

        if cv.countNonZero(img[y_min:y_max, x_min_left:x_max_left]) == 0:
            blinks_on.append(False)
        else:
            blinks_on.append(True)

        if cv.countNonZero(img[y_min:y_max, x_min_right:x_max_right]) == 0:
            blinks_on.append(False)
        else:
            blinks_on.append(True)

        return blinks_on

    def update_blink_history(self, current_blink):
        for instance_id in list(self.blink_dict_history):
            if instance_id in current_blink:
                if current_blink[instance_id][0]:
                    if not self.blink_dict_history[instance_id]["left_blink"]:
                        self.blink_dict_history[instance_id]["count_left"] += 1
                        if self.blink_dict_history[instance_id]["count_left"] >= BLINK_TO_TURN_ON:
                            self.blink_dict_history[instance_id]["left_blink"] = True
                            self.blink_dict_history[instance_id]["count_left"] = 0
                    else:
                        self.blink_dict_history[instance_id]["count_left"] = 0

                if current_blink[instance_id][1]:
                    if not self.blink_dict_history[instance_id]["right_blink"]:
                        self.blink_dict_history[instance_id]["count_right"] += 1
                        if self.blink_dict_history[instance_id]["count_right"] >= BLINK_TO_TURN_ON:
                            self.blink_dict_history[instance_id]["right_blink"] = True
                            self.blink_dict_history[instance_id]["count_right"] = 0
                    else:
                        self.blink_dict_history[instance_id]["count_right"] = 0
                else:
                    if self.blink_dict_history[instance_id]["left_blink"]:
                        self.blink_dict_history[instance_id]["count_left"] += 1
                        if self.blink_dict_history[instance_id]["count_left"] >= BLINK_TO_TURN_OFF:
                            self.blink_dict_history[instance_id]["left_blink"] = False
                            self.blink_dict_history[instance_id]["count_left"] = 0
                    else:
                        self.blink_dict_history[instance_id]["count_left"] = 0

                    if self.blink_dict_history[instance_id]["right_blink"]:
                        self.blink_dict_history[instance_id]["count_right"] += 1
                        if self.blink_dict_history[instance_id]["count_right"] >= BLINK_TO_TURN_OFF:
                            self.blink_dict_history[instance_id]["right_blink"] = False
                            self.blink_dict_history[instance_id]["count_right"] = 0
                    else:
                        self.blink_dict_history[instance_id]["count_right"] = 0
                current_blink.pop(instance_id)
            else:
                self.blink_dict_history.pop(instance_id)

        for id_instance in current_blink:
            self.blink_dict_history[id_instance] = {"left_blink": False, "right_blink": False, "count_left": 0, "count_right": 0}
            if current_blink[id_instance][0]:
                self.blink_dict_history[id_instance]["count_left"] += 1
            if current_blink[id_instance][1]:
                self.blink_dict_history[id_instance]["count_right"] += 1

    def check_lights(self, img, obj_list):
        current_blink_dict = {}
        if obj_list is not None and self.previous_image is not None:
            current_blink_dict, th_image = self.check_vehicle_lights_on(self.previous_image, self.previous_obj_list, img, obj_list)
            self.update_blink_history(current_blink_dict)

        self.previous_image = img.copy()
        self.previous_obj_list = obj_list
        
        for obj in obj_list:
            if obj.bbox.class_id in HAS_LIGHT and obj.bbox.instance_id in self.blink_dict_history:
                obj.left_blink = self.blink_dict_history[obj.bbox.instance_id]["left_blink"]
                obj.right_blink = self.blink_dict_history[obj.bbox.instance_id]["right_blink"]

        return obj_list