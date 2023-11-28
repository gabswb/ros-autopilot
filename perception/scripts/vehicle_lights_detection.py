
from Sensors import Sensors
import config
import cv2 as cv
import numpy as np
from statistics import mean

LIGHT_OFF_MAX_COUNT = 8
LIGHT_ON_MIN_COUNT = 3

MEAN_MIN_STEP = 3
THRESH_METHOD = True
HAS_LIGHT = (2, 5, 6, 7, 8) # car, bus, train, truck, boat

class VehicleLight:
    is_on = False

    light_on_count = 0
    light_off_count = 0

    previous_mean = 0

    previous_image = []
    current_image = []

    def __init__(self, side, box, id):
        self.side = side
        self.box = box
        self.id = id

    def update_info(self):
        try:
            x1, y1, x2, y2 = self.box

            self.previous_image = self.current_image.copy()
            self.current_image = config.s.camera_image[y1:y2, x1:x2]
        except ValueError:
            pass

        if self.previous_image != []:
            self.check_state()

    def check_state(self):
        if not self.get_new_image_state():
            self.light_on_count = 0
            self.light_off_count += 1
            if self.light_off_count >= LIGHT_OFF_MAX_COUNT and self.is_on:
                self.is_on = False
        else:
            self.light_on_count += 1
            self.light_off_count = 0
            if self.light_on_count >= LIGHT_ON_MIN_COUNT and not self.is_on:
                self.is_on = True

    def get_new_image_state(self):
        light_on = False
        if THRESH_METHOD:
            cv.cvtColor(self.current_image, cv.COLOR_RGB2HSV)
            binary_image = cv.inRange(self.current_image, (0, 43, 220), (180, 255, 255))

            kernel = np.ones((5, 5), np.uint8)
            opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel, iterations=1)

            result = cv.bitwise_and(self.current_image, self.current_image, mask=opening)

            cv.imshow(self.side, result)

        else:
            processed_image = self.process_images()

            (numLabels, labels, stats, centroids) = cv.connectedComponentsWithStats(processed_image)

            if numLabels == 1:
                light_on = False
            else:
                light_on = True

        return light_on

    def process_images(self):
        prev_img_bw = cv.cvtColor(self.previous_image, cv.COLOR_RGB2GRAY)
        prev_img_blur = cv.GaussianBlur(prev_img_bw, (11, 11), 0)

        current_img_bw = cv.cvtColor(self.current_image, cv.COLOR_RGB2GRAY)
        current_img_blur = cv.GaussianBlur(current_img_bw, (11, 11), 0)

        img_sub = cv.absdiff(prev_img_blur, current_img_blur)
        _, img_th = cv.threshold(img_sub, 15, 255, cv.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        opening = cv.morphologyEx(img_th, cv.MORPH_OPEN, kernel, iterations=2)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=2)

        # dilate = cv.morphologyEx(closing, cv.MORPH_DILATE, kernel, iterations=3)
        # cv.imshow("Test", dilate)
        # cv.waitKey(0)

        return closing

class VehicleLightsDetection:
    def __init__(self):
        self.vehicle_lights = []

    def detect(self, img, obj_list):
        existing_light_ids = [light.id for light in self.vehicle_lights]
        # for each object in the list, check if it has a light and an instance id
        for obj in obj_list:
            if obj.bbox.class_id in HAS_LIGHT and obj.bbox.instance_id is not None:
                # update the object's light
                if obj.bbox.instance_id in existing_light_ids:
                    self.vehicle_lights[existing_light_ids.index(obj.bbox.instance_id)].update_info()
                else:
                    self.vehicle_lights.append(VehicleLight(obj.bbox.side, obj.bbox.box))
        return self.vehicle_lights