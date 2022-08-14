from time import sleep
import rclpy
from std_msgs.msg import String, Int32, Float32
from geometry_msgs.msg import Twist

'''
sudo apt-get install python3-catkin-pkg

ROS issues w python3:
https://discourse.ros.org/t/should-we-warn-new-users-about-difficulties-with-python-3-and-alternative-python-interpreters/3874/3
'''

class RosPubisher(object):
    '''
    A ROS node to pubish to a data stream
    '''
    #def __init__(self, node_name, channel_name, stream_type=String, anonymous=True):
    def __init__(self,node_name,topic_name,args=None,stream_type=Float32, anonymous=True):
        #rclpy.init(args=args)
        self.node = rclpy.create_node(node_name)
        self.publisher = self.node.create_publisher(stream_type, topic_name, 10)
        self.msg = Float32()

        #self.data = ""
        #self.pub = rospy.Publisher(channel_name, stream_type)
        #rospy.init_node(node_name, anonymous=anonymous)

    def run(self, data):
        '''
        only publish when data stream changes.
        '''
        #print('test')
        #while rclpy.ok():
        self.msg.data = float(data)            
        self.publisher.publish(self.msg)



class RosSubscriber(object):
    '''
    A ROS node to subscribe to a data stream
    '''

    def __init__(self, node_name, channel_name, stream_type=String, anonymous=True):
        self.data = ""
        rospy.init_node(node_name, anonymous=anonymous)
        self.pub = rospy.Subscriber(channel_name, stream_type, self.on_data_recv)        

    def on_data_recv(self, data):
        self.data = data.data

    def run(self):
        return self.data

class RosTwistPubisher(object):
    '''
    A ROS node to pubish to a data stream
    '''
    #def __init__(self, node_name, channel_name, stream_type=String, anonymous=True):
    def __init__(self,node_name,topic_name,args=None,stream_type=Twist, anonymous=True):
        #rclpy.init(args=args)
        self.node = rclpy.create_node(node_name)
        #self.publisher = self.node.create_publisher(stream_type, topic_name, 10)
        self.publisher = self.node.create_publisher(stream_type,'cmd_vel',10)
        self.twist = Twist()

        #self.data = ""
        #self.pub = rospy.Publisher(channel_name, stream_type)
        #rospy.init_node(node_name, anonymous=anonymous)

    def run(self, x,z):
        '''
        only publish when data stream changes.
        '''
        #print('test')
        #while rclpy.ok():
        
        self.twist.linear.x = float(x)
        self.twist.angular.z = float(z)

        self.publisher.publish(self.twist)
