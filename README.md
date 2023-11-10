# ros-autopilot
## Build
### Requirements
- install packages listed in `requirements.txt`
- install OpenBLAS and Armadillo for transform track package
```sh
sudo apt-get install libopenblas-dev
# Download Armadillo sources from https://arma.sourceforge.net/download.html
tar -xvf armadillo-9.880.1.tar.gz 
cd armadillo-9.880.1 
./configure 
make 
sudo make install
```
### With catkin
```sh
mkdir -p catkin_ws/src
git clone https://github.com/gabswb/ros-autopilot.git catkin_ws/src
cd catkin_ws/
catkin_make
cythonize -3 -a -i src/perception/scripts/fish2bird.pyx
mv src/perception/scripts/fish2bird*.so devel/lib/python3/dist-packages/
rm src/perception/scripts/fish2bird.c fish2bird.html
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

