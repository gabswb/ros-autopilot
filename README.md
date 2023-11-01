# ros-autopilot
## Build
### Requirements
- install packages listed in `requirements.txt`
### With catkin
```sh
mkdir -p catkin_ws/src 
git clone https://github.com/gabswb/ros-autopilot.git catkin_ws/src
cd catkin_ws/
catkin_make
```

### Usage
- **Object detection**:
    - pulish the object bouding box and id of the detcted object with yolo throught the topic `/object_bounding_box` under the `ObjectBoundingBox.msg` message type (to see msg attributes: `rostopic echo /object_bounding_box`)
    - to retrieve the class name with `class_id` use:
        ```py
        with open("src/object_detection/yolov4-tiny/classes.names", 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        rospy.loginfo(f"{classes[class_id]}")
        ```

