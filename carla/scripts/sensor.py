import rospy
import tf2_ros

from std_msgs.msg import Header


class Sensor:
    def __init__(self, carla_object, frame_id, child_frame_id, topic_name, position):
        self.carla_object = carla_object
        self.frame_id = frame_id
        self.child_frame_id = child_frame_id
        self.topic_name = topic_name

        # Publish les transform de la camera et du lidar
        self.tf = self.get_tf(position, frame_id, child_frame_id)

        # Lance le publish de la camera
        self.carla_object.listen(lambda data: self.sensor_callback(data))

    def get_msg_header(self):
        """
        Get a filled ROS message header
        :return: ROS message header
        :rtype: std_msgs.msg.Header
        """
        header = Header()
        header.frame_id = self.frame_id

        header.stamp = rospy.Time.from_sec(rospy.get_time())

        return header

    def get_tf(self, pose, frame_id, child_frame_id=None):
        transform = tf2_ros.TransformStamped()
        transform.header.stamp = rospy.Time.from_sec(rospy.get_time())
        transform.header.frame_id = frame_id
        if child_frame_id:
            transform.child_frame_id = child_frame_id

        transform.transform.translation.x = pose#.position.x
        transform.transform.translation.y = pose#.position.y
        transform.transform.translation.z = pose#.position.z

        transform.transform.rotation.x = pose#.orientation.x
        transform.transform.rotation.y = pose#.orientation.y
        transform.transform.rotation.z = pose#.orientation.z
        transform.transform.rotation.w = pose#.orientation.w

        return transform

    def sensor_callback(self, data):
        """
        Function overloaded in each sensors
        :param data:
        :return:
        """
        pass
