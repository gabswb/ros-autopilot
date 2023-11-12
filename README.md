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
# if fatal error: numpy/arrayobject.h: No such file or directory => python src/perception/scripts/setup.py build_ext --inplace (setuptools package required)
mv src/perception/scripts/fish2bird*.so devel/lib/python3/dist-packages/
rm src/perception/scripts/fish2bird.c fish2bird.html
```

### Usage
Run at the root dir (the one with src/ build/ etc ...)  :
```sh
roscore
simulator # launch utac simulator
rosrun perception perception.py src/config-utac.yml
```