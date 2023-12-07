import cv2 as cv
import numpy as np
import rospy
from math import sqrt

HAS_LIGHT = (2, 5, 6, 7, 8)

BLINK_HEIGHT = [0.3, 0.55]
LEFT_BLINK = [0, 0.3]
RIGHT_BLINK = [0.7, 1]


def check_vehicle_lights_on(previous_img, previous_bbox, current_image, current_bbox):
    new_image = np.zeros((current_image.shape[0], current_image.shape[1]), dtype=np.uint8)

    light_blink = {}

    for bbox in current_bbox:
        if bbox.class_id in HAS_LIGHT:
            corresponding_previous_bbox = [prev_bbox for prev_bbox in previous_bbox if
                                           (prev_bbox.class_id == bbox.class_id and
                                            prev_bbox.instance_id == bbox.instance_id)]

            if len(corresponding_previous_bbox) >= 1:
                dst_corners, current_diagonal = get_bbox_corners(bbox)
                src_corners, previous_diagonal = get_bbox_corners(corresponding_previous_bbox[0])


                M = cv.getPerspectiveTransform(src_corners, dst_corners)

                previous_isolate_bbox = isolate_bbox(previous_img, corresponding_previous_bbox[0])
                current_isolate_bbox = isolate_bbox(current_image, bbox)

                previous_isolate_bbox = cv.warpPerspective(
                    previous_isolate_bbox, M, (previous_isolate_bbox.shape[1], previous_isolate_bbox.shape[0]))

                x1, y1, x2, y2 = np.int32((dst_corners[0][0], dst_corners[0][1],
                                          dst_corners[2][0], dst_corners[2][1]))

                image_to_analyse = cv.absdiff(current_isolate_bbox[y1:y2, x1:x2], previous_isolate_bbox[y1:y2, x1:x2])
                image_hsv = cv.cvtColor(image_to_analyse, cv.COLOR_RGB2HSV)
                th_image = cv.inRange(image_hsv, (0, 43, 140), (255, 255, 255))

                if 0.96 * previous_diagonal <= current_diagonal <= 1.04 * previous_diagonal:
                    light_blink[bbox.instance_id] = check_light_blink(th_image)
                else:
                    light_blink[bbox.instance_id] = [False, False]

                    new_image[y1:y2, x1:x2] = th_image

    return light_blink, new_image


def isolate_bbox(image_with_bbox, bbox):
    black_image = np.zeros_like(image_with_bbox, dtype=np.uint8)
    x = bbox.x
    y = bbox.y
    x_plus_w = x + bbox.w
    y_plus_h = y + bbox.h

    black_image[y:y_plus_h, x:x_plus_w] = image_with_bbox[y:y_plus_h, x:x_plus_w]

    return black_image


def get_bbox_corners(bbox):
    x = bbox.x
    y = bbox.y
    x_plus_w = x + bbox.w
    y_plus_h = y + bbox.h

    l_x = x_plus_w - x
    l_y = y_plus_h - y

    diagonal = sqrt(l_x ** 2 + l_y ** 2)

    corners = np.float32([(x, y), (x_plus_w, y), (x_plus_w, y_plus_h), (x, y_plus_h)]), diagonal

    return corners


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
