import cv2 as cv
import numpy as np
import rospy
from math import sqrt

HAS_LIGHT = (2, 5, 6, 7, 8)

BLINK_HEIGHT = [0.25, 0.55]
LEFT_BLINK = [0, 0.3]
RIGHT_BLINK = [0.7, 1]
BLINK_TO_TURN_ON = 2
BLINK_TO_TURN_OFF = 6

DIAGONAL_MAX_INCREASE = 0.04
HSV_VALUE_THRESHOLD = 140


class VehicleLightsDetector(object):
    def __init__(self):
        self.previous_obj_list = None
        self.previous_image = None
        self.blink_dict_history = {}
        rospy.loginfo("VehicleLightsDetector initialized")

    def check_vehicle_lights_on(self, current_image, current_obj):
        vehicles_list_blinking = {}

        for obj in current_obj:
            if obj.bbox.class_id in HAS_LIGHT:
                corresponding_previous_bbox = [prev_obj for prev_obj in self.previous_obj_list if
                                               prev_obj.bbox.instance_id == obj.bbox.instance_id]

                if len(corresponding_previous_bbox) >= 1:
                    previous_isolate_bbox, src_corners, previous_diagonal = isolate_bbox(
                        self.previous_image, corresponding_previous_bbox[0].bbox)
                    current_isolate_bbox, dst_corners, current_diagonal = isolate_bbox(current_image, obj.bbox)

                    x1, y1, x2, y2 = np.int32((dst_corners[0][0], dst_corners[0][1],
                                               dst_corners[2][0], dst_corners[2][1]))

                    M = cv.getPerspectiveTransform(src_corners, dst_corners)

                    previous_isolate_bbox = cv.warpPerspective(
                        previous_isolate_bbox, M, (previous_isolate_bbox.shape[1], previous_isolate_bbox.shape[0]))

                    image_abs_difference = cv.absdiff(current_isolate_bbox[y1:y2, x1:x2],
                                                      previous_isolate_bbox[y1:y2, x1:x2])
                    image_hsv = cv.cvtColor(image_abs_difference, cv.COLOR_RGB2HSV)
                    threshold_image = cv.inRange(image_hsv, (0, 43, HSV_VALUE_THRESHOLD), (255, 255, 255))

                    if ((1 - DIAGONAL_MAX_INCREASE) * previous_diagonal <= current_diagonal <=
                            (1 + DIAGONAL_MAX_INCREASE) * previous_diagonal):
                        vehicles_list_blinking[obj.bbox.instance_id] = check_light_blink(threshold_image)
                    else:
                        vehicles_list_blinking[obj.bbox.instance_id] = [False, False]

        return vehicles_list_blinking

    def update_blink_history(self, current_blink):
        # Mise à jour des véhicules déjà présents sur l'image
        for instance_id in list(self.blink_dict_history):
            if instance_id in current_blink:
                if current_blink[instance_id][0]:  # Le feu gauche clignotte sur la frame
                    if not self.blink_dict_history[instance_id]["left_blink"]:  # Si le voyant était éteint
                        self.blink_dict_history[instance_id]["count_left"] += 1

                        if self.blink_dict_history[instance_id]["count_left"] >= BLINK_TO_TURN_ON:
                            self.blink_dict_history[instance_id]["left_blink"] = True
                            self.blink_dict_history[instance_id]["count_left"] = 0
                    else:
                        self.blink_dict_history[instance_id]["count_left"] = 0
                else:
                    if self.blink_dict_history[instance_id]["left_blink"]:  # Si le voyant était allumé
                        self.blink_dict_history[instance_id]["count_left"] += 1

                        if self.blink_dict_history[instance_id]["count_left"] >= BLINK_TO_TURN_OFF:
                            self.blink_dict_history[instance_id]["left_blink"] = False
                            self.blink_dict_history[instance_id]["count_left"] = 0
                    else:
                        self.blink_dict_history[instance_id]["count_left"] = 0

                if current_blink[instance_id][1]:  # Le feu droit clignotte sur la frame
                    if not self.blink_dict_history[instance_id]["right_blink"]:
                        self.blink_dict_history[instance_id]["count_right"] += 1

                        if self.blink_dict_history[instance_id]["count_right"] >= BLINK_TO_TURN_ON:
                            self.blink_dict_history[instance_id]["right_blink"] = True
                            self.blink_dict_history[instance_id]["count_right"] = 0
                    else:
                        self.blink_dict_history[instance_id]["count_right"] = 0
                else:
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

        # Ajout des nouveaux véhicules
        for id_instance in current_blink:
            self.blink_dict_history[id_instance] = {
                "left_blink": False,
                "right_blink": False,
                "count_left": 0,
                "count_right": 0
            }

            if current_blink[id_instance][0]:
                self.blink_dict_history[id_instance]["count_left"] += 1
            if current_blink[id_instance][1]:
                self.blink_dict_history[id_instance]["count_right"] += 1

    def check_lights(self, img, obj_list):
        if obj_list is not None and self.previous_image is not None:
            current_blink_dict = self.check_vehicle_lights_on(img,
                                                              obj_list)
            self.update_blink_history(current_blink_dict)

        self.previous_image = img.copy()
        self.previous_obj_list = obj_list

        for obj in obj_list:
            if obj.bbox.class_id in HAS_LIGHT and obj.bbox.instance_id in self.blink_dict_history:
                obj.left_blink = self.blink_dict_history[obj.bbox.instance_id]["left_blink"]
                obj.right_blink = self.blink_dict_history[obj.bbox.instance_id]["right_blink"]

        return obj_list


def isolate_bbox(image_with_bbox, bbox):
    black_image = np.zeros_like(image_with_bbox, dtype=np.uint8)
    x = bbox.x
    y = bbox.y
    x_plus_w = x + bbox.w
    y_plus_h = y + bbox.h

    l_x = x_plus_w - x
    l_y = y_plus_h - y

    diagonal = sqrt(l_x ** 2 + l_y ** 2)

    corners = np.array([(x, y), (x_plus_w, y), (x_plus_w, y_plus_h), (x, y_plus_h)], dtype=np.float32)

    black_image[y:y_plus_h, x:x_plus_w] = image_with_bbox[y:y_plus_h, x:x_plus_w]

    return black_image, corners, diagonal


def check_light_blink(img):
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
