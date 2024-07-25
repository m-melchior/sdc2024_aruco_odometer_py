from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
from collections import deque
from geometry_msgs.msg import Point, TransformStamped, PoseStamped
import json
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as SST_Rotation
from sensor_msgs.msg import Image
import time
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformBroadcaster


PATH_PARAMETERS = "../assets/parameters.json"

PATH_OAK_CALIB = "../assets/oak_1.json"

PATH_MARKER_SIGNS = "../assets/marker_signs.json"
PATH_MARKER_BOXES = "../assets/marker_boxes.json"
marker_list = [
	["sign", PATH_MARKER_SIGNS],
	["tgt", PATH_MARKER_BOXES]
]

TGT_IDS_BLUE = {61, 62, 63, 71, 72, 73}
TGT_IDS_RED = {64, 65, 66, 74, 75, 76}

SIZE_LOS = 100
LIFETIME_LOS_S = 3

class RollingWindowVec:
    def __init__(self, window_size=10):
        self.items = deque(maxlen=window_size)

    def add_element(self, element):
        if isinstance(element, numpy.ndarray):
            self.items.append(element)
        else:
            raise ValueError("Element must be a numpy array")

    def get_median(self):
        if not self.items:
            return None
        return numpy.median(numpy.array(self.items), axis=0)

    def get_mean(self):
        if not self.items:
            return None
        return numpy.mean(numpy.array(self.items), axis=0)

    def get_std(self):
        if not self.items:
            return None
        return numpy.std(numpy.array(self.items), axis=0)

    def __str__(self):
        return f"RollingWindow({list(self.items)})"


# ******************************************
# required for publish marker arrays
def euler_to_quaternion(roll, pitch, yaw):
	rotation = SST_Rotation.from_euler('xyz', [roll, pitch, yaw])
	return rotation.as_quat()

# ******************************************
def rotate(points, angles):
	rot_x, rot_y, rot_z = angles

	Rx = numpy.array([
		[1, 0, 0],
		[0, numpy.cos(rot_x), -numpy.sin(rot_x)],
		[0, numpy.sin(rot_x), numpy.cos(rot_x)]
	])

	Ry = numpy.array([
		[numpy.cos(rot_y), 0, numpy.sin(rot_y)],
		[0, 1, 0],
		[-numpy.sin(rot_y), 0, numpy.cos(rot_y)]
	])

	Rz = numpy.array([
		[numpy.cos(rot_z), -numpy.sin(rot_z), 0],
		[numpy.sin(rot_z), numpy.cos(rot_z), 0],
		[0, 0, 1]
	])

	R = Rz @ Ry @ Rx
	return points @ R.T

# ******************************************
def get_data(path_in):
	data = None

	try:
		with open(path_in, "r") as file:
			data = json.load(file)
	except Exception as ex:
		print (f"ERROR: failed to get data: {ex}")

	return data
	
# ******************************************
def evaluate_parametric_value(value, parameters):
	if isinstance(value, str):
		return eval(value, {}, parameters)
	return value

# ******************************************
def get_markers(type, path_markerdata):
	ret = {}

	parameters = get_data(PATH_PARAMETERS)
	marker_data = get_data(path_markerdata)

	size_markers = marker_data["size_markers"]
	path_mat = marker_data["path_mat"]
	size_tags = marker_data["size_tags"]
	markers = marker_data["markers"]

	for marker in markers:
		id_marker, position, orientation = marker
		position = [evaluate_parametric_value(val, parameters) for val in position]

		ret[id_marker] = {
			"position": (float(position[0]), float(position[1]), float(position[2])),
			"orientation": (math.radians(orientation[0]), math.radians(orientation[1]), math.radians(orientation[2]))
		}
		# print(f"marker: {id_marker}, {position}, {orientation}")

	# print(f"ret: {ret}")

	return ret, size_markers

# ******************************************
class ArucoDetector(Node):
	# ******************************************
	def __init__(self):
		super().__init__('aruco_detector')

		self.logger = self.get_logger()

		self.cam_matrix = None
		self.cam_distortion = None

		self.rolling_window_vec_r = None
		self.rolling_window_vec_t = None

		self.rolling_window_vec_pos = None

		self.markers_nav = {}
		self.markers_nav_size = 0
		self.markers_tgt = {}
		self.markers_tgt_size = 0

		self.subscriber_cam = None
		self.publisher_image_aruco = None
		self.publisher_rviz_markers_nav = None
		self.publisher_rviz_markers_tgt = None
		self.publisher_rviz_position = None

		self.publisher_rviz_camera = None

		self.publisher_rviz_pose = None

		self.cv_bridge = None

		self.aruco_dict_nav = None
		self.aruco_dict_tgt = None
		self.aruco_detector_parameters = None

		self.fps = 0
		self.frame_count = 0
		self.time_start = None

		self.fig = None
		self.ax = None

		self.avg_deg = None
		self.avg_ry = None

		self.tfbroadcaster = TransformBroadcaster(self)

	# ******************************************
	def start(self):
		self.rolling_window_vec_r = RollingWindowVec(50)
		self.rolling_window_vec_t = RollingWindowVec(50)

		self.rolling_window_vec_pos = RollingWindowVec(50)

		self.markers_nav, self.markers_nav_size = get_markers("sign", PATH_MARKER_SIGNS)
		self.markers_tgt, self.markers_tgt_size = get_markers("tgt", PATH_MARKER_BOXES)

		with open(PATH_OAK_CALIB, "r") as file:
			data = json.load(file)

		self.cam_matrix = numpy.array(data["mtx"])
		self.cam_distortion = numpy.array(data["dist"])

		# print(f"data:{data}")
		# print(f"self.cam_matrix: {self.cam_matrix}")
		# print(f"self.cam_distortion: {self.cam_distortion}")

		self.subscriber_cam = self.create_subscription(
			Image,
			'/quadrotor_00/Camera/rgb',
			self.callback_cam,
			10
		)

		self.publisher_image_aruco = self.create_publisher(Image, '/image_aruco', 10)
		self.publisher_rviz_markers_nav = self.create_publisher(MarkerArray, 'rviz_markers_nav', 10)
		self.publisher_rviz_markers_tgt = self.create_publisher(MarkerArray, 'rviz_markers_tgt', 10)

		self.publisher_rviz_los = self.create_publisher(Marker, 'rviz_los', 10)

		self.publisher_rviz_position = self.create_publisher(Marker, 'rviz_position', 10)

		self.publisher_rviz_camera = self.create_publisher(Marker, 'rviz_camera', 10)

		self.publisher_rviz_pose = self.create_publisher(PoseStamped, 'rviz_pose', 10)

		self.cv_bridge = CvBridge()

		self.aruco_dict_nav = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
		self.aruco_dict_tgt = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
		self.aruco_detector_parameters = aruco.DetectorParameters()

		self.time_start = time.time()
		self.publish_markers_nav(None)
		self.publish_markers_tgt({64, 65, 66, 71, 72, 73})

		self.avg_deg = deque(maxlen = 10)
		self.avg_ry = deque(maxlen = 10)

	# ******************************************
	def callback_cam(self, msg):
		self.frame_count += 1
		time_now = time.time()
		time_delta = time_now - self.time_start

		if time_delta >= 1.0:
			self.fps = self.frame_count / time_delta
			self.frame_count = 0
			self.time_start = time_now
			# self.logger.info(f"fps: {self.fps}")

		# NAV Markers
		cv_image_raw = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
		cv_image_gray = cv2.cvtColor(cv_image_raw, cv2.COLOR_BGR2GRAY)

		marker_corners_nav, marker_ids_nav, rejected = aruco.detectMarkers(cv_image_gray, self.aruco_dict_nav, parameters = self.aruco_detector_parameters)
		# print(f"marker_corners_nav: {marker_corners_nav}")

		if marker_ids_nav is not None:
			cv_image_raw = aruco.drawDetectedMarkers(cv_image_raw, marker_corners_nav, marker_ids_nav)

			marker_ids = set()
			for index, marker_id in enumerate(marker_ids_nav):
				if(marker_id[0] not in self.markers_nav):
					print(f"ERROR: Unknown marker id: {marker_id}")
					continue

				id_aruco = int(marker_id[0])
				marker_ids.add(id_aruco)

				rotations, translations, _ = aruco.estimatePoseSingleMarkers(marker_corners_nav, self.markers_nav_size, self.cam_matrix, self.cam_distortion)
				rotation = rotations[index].flatten()
				translation = translations[index].flatten()

				self.rolling_window_vec_r.add_element(numpy.array(rotation))
				self.rolling_window_vec_t.add_element(numpy.array(translation))

				marker_position = numpy.array(self.markers_nav[id_aruco]["position"])
				marker_orientation = self.markers_nav[id_aruco]["orientation"]

				rotation_matrix = cv2.Rodrigues(self.rolling_window_vec_r.get_median())[0]
				translation_vector = self.rolling_window_vec_t.get_median().reshape(3, 1)
				inverse_rotation_matrix = rotation_matrix.T
				inverse_translation_vector = -inverse_rotation_matrix @ translation_vector
				inverse_translation_vector = inverse_translation_vector.flatten()

				pos_rot = rotate(inverse_translation_vector, marker_orientation)
				pos_rot_trans = pos_rot + marker_position

				self.rolling_window_vec_pos.add_element(pos_rot_trans)

				rolling_window_vec_pos = self.rolling_window_vec_pos.get_median()


				sst_rotation = SST_Rotation.from_matrix(inverse_rotation_matrix)
				sst_rotation_quat = sst_rotation.as_quat()

				marker = Marker()
				marker.header.frame_id = "map"
				marker.header.stamp = self.get_clock().now().to_msg()
				marker.ns = "cam"
				marker.id = 0
				marker.type = Marker.SPHERE
				marker.action = Marker.ADD
				marker.pose.position.x = rolling_window_vec_pos[0]
				marker.pose.position.y = rolling_window_vec_pos[1]
				marker.pose.position.z = rolling_window_vec_pos[2]
				marker.scale.x = 0.3
				marker.scale.y = 0.3
				marker.scale.z = 0.3
				
				marker.color.a = 1.0
				marker.color.r = 0.0
				marker.color.g = 1.0
				marker.color.b = 0.0

				self.publisher_rviz_camera.publish(marker)
				
				# start_time = time.time()
				rotation_matrix, _ = cv2.Rodrigues(rotation)
				angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)
				# elapsed_time = time.time() - start_time
				# print(f"elapsed_time: {elapsed_time:.6f}")
				# print(f"angles: {angles}")
				# yaw_angle = angles[1]

				self.publish_los(id_aruco, angles[1])

				cv2.drawFrameAxes(cv_image_raw, self.cam_matrix, self.cam_distortion, rotations[index], translations[index], 0.1)

			self.publish_markers_nav(marker_ids)


		# Target Markers
		marker_corners_tgt, marker_ids_tgt, rejected = aruco.detectMarkers(cv_image_gray, self.aruco_dict_tgt, parameters = self.aruco_detector_parameters)
		if marker_ids_tgt is not None:
			marker_ids = set()
			cv_image_raw = aruco.drawDetectedMarkers(cv_image_raw, marker_corners_tgt, marker_ids_tgt)

			for marker_id in range(len(marker_ids_tgt)):
				marker_ids.add(marker_ids_tgt[marker_id][0])
				corners = marker_corners_tgt[marker_id].reshape((4, 2))

				x = [corner[0] for corner in corners]
				y = [corner[1] for corner in corners]
				p_center = (int(sum(x) / len(corners)), int(sum(y) / len(corners)))

				cv2.circle(cv_image_raw, p_center, 8, (0, 0, 255), -1)

				rotations, translations, markerPoints = aruco.estimatePoseSingleMarkers(marker_corners_tgt[marker_id], self.markers_tgt_size, self.cam_matrix, self.cam_distortion)
				(rotations - translations).any()

			# print(f"marker_ids: {marker_ids}")
			self.publish_markers_tgt(marker_ids)

		# Text and markers_nav on the image
		t_proc = time.time() - time_now		
		cv2.putText(cv_image_raw, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
		cv2.putText(cv_image_raw, f"t_proc: {t_proc:.4f}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)

		# does not work in docker, thus the publishing
		# cv2.imshow("ArUco", cv_image_raw)

		msg_image_processed = self.cv_bridge.cv2_to_imgmsg(cv_image_raw, "bgr8")
		self.publisher_image_aruco.publish(msg_image_processed)

	# ******************************************
	def publish_markers_nav(self, marker_ids_in):
		marker_array = MarkerArray()
		for marker in self.markers_nav.items():
			marker_id, marker_data = marker
			# print(f"marker_data: {marker_data}")
			marker_msg = Marker()
			marker_msg.header.frame_id = "map"
			marker_msg.id = marker_id
			marker_msg.type = Marker.CUBE
			marker_msg.action = Marker.ADD

			marker_msg.pose.position.x = marker_data['position'][0]
			marker_msg.pose.position.y = marker_data['position'][1]
			marker_msg.pose.position.z = marker_data['position'][2]

			q_x, q_y, q_z, q_w = euler_to_quaternion(marker_data["orientation"][0], marker_data["orientation"][1], marker_data["orientation"][2])
			marker_msg.pose.orientation.x = q_x
			marker_msg.pose.orientation.y = q_y
			marker_msg.pose.orientation.z = q_z
			marker_msg.pose.orientation.w = q_w

			marker_msg.scale.x = 0.5
			marker_msg.scale.y = 0.5
			marker_msg.scale.z = 0.01

			marker_msg.color.a = 1.0

			if (marker_ids_in and (marker_id in marker_ids_in)):
				marker_msg.color.r = 0.0
				marker_msg.color.g = 1.0
				marker_msg.color.b = 0.0
			else:
				marker_msg.color.r = 1.0
				marker_msg.color.g = 1.0
				marker_msg.color.b = 1.0

			marker_array.markers.append(marker_msg)
		
		self.publisher_rviz_markers_nav.publish(marker_array)

	# ******************************************
	def publish_markers_tgt(self, marker_ids_in):
		marker_array = MarkerArray()
		for marker_id in marker_ids_in:
			# print(f"marker_id: {marker_id}")
			if (marker_id not in self.markers_tgt.keys()):
				continue
			marker_data = self.markers_tgt[marker_id]
			# print(f"marker_data: {marker_data}")
			marker_msg = Marker()
			marker_msg.header.frame_id = "map"
			if ((marker_id == 61) or (marker_id == 64)):
				marker_msg.id = 1
			if ((marker_id == 62) or (marker_id == 65)):
				marker_msg.id = 2
			if ((marker_id == 63) or (marker_id == 66)):
				marker_msg.id = 3
			if ((marker_id == 71) or (marker_id == 74)):
				marker_msg.id = 4
			if ((marker_id == 72) or (marker_id == 75)):
				marker_msg.id = 5
			if ((marker_id == 73) or (marker_id == 76)):
				marker_msg.id = 6
			marker_msg.type = Marker.CUBE
			marker_msg.action = Marker.ADD

			marker_msg.pose.position.x = marker_data['position'][0]
			marker_msg.pose.position.y = marker_data['position'][1]
			marker_msg.pose.position.z = marker_data['position'][2]

			q_x, q_y, q_z, q_w = euler_to_quaternion(0, 0, marker_data["orientation"][2])
			marker_msg.pose.orientation.x = q_x
			marker_msg.pose.orientation.y = q_y
			marker_msg.pose.orientation.z = q_z
			marker_msg.pose.orientation.w = q_w

			marker_msg.scale.x = 0.2
			marker_msg.scale.y = 0.01
			marker_msg.scale.z = 0.2

			marker_msg.color.a = 1.0

			if (marker_id in TGT_IDS_BLUE):
				marker_msg.color.r = 0.0
				marker_msg.color.g = 0.0
				marker_msg.color.b = 1.0
			else:
				marker_msg.color.r = 1.0
				marker_msg.color.g = 0.0
				marker_msg.color.b = 0.0

			marker_array.markers.append(marker_msg)
		
		if(len(marker_array.markers) > 0):
			self.publisher_rviz_markers_tgt.publish(marker_array)

	# ******************************************
	def publish_los(self, marker_id_in, angle_in):
		marker = Marker()
		marker.header.frame_id = "map"
		marker.id = int(marker_id_in)
		marker.type = Marker.LINE_STRIP
		marker.action = Marker.ADD

		marker.scale.x = 0.1

		marker.color.a = 1.0
		marker.color.r = 0.0
		marker.color.g = 0.0
		marker.color.b = 1.0

		marker_position = self.markers_nav[marker_id_in]["position"]
		angle = self.markers_nav[marker_id_in]["orientation"][2] - angle_in

		start = Point()
		start.x = float(marker_position[0])
		start.y = float(marker_position[1])
		start.z = float(marker_position[2])

		end = Point()
		end.x = float(marker_position[0] + SIZE_LOS * numpy.sin(numpy.radians(angle)))
		end.y = float(marker_position[1] + SIZE_LOS * numpy.cos(numpy.radians(angle)))
		end.z = float(marker_position[2])

		marker.points.append(start)
		marker.points.append(end)
		
		self.publisher_rviz_los.publish(marker)
	
	# ******************************************
	def publish_positions(self, positions_in):
		marker_array = MarkerArray()
		for position in positions_in:
			marker_msg = Marker()
			marker_msg.header.frame_id = "map"
			marker_msg.id = marker_id
			marker_msg.type = Marker.SPHERE
			marker_msg.action = Marker.ADD

			marker_msg.pose.position.x = position[0]
			marker_msg.pose.position.y = position[1]
			marker_msg.pose.position.z = position[2]

			marker_msg.pose.orientation.w = 1

			marker_msg.scale.x = 1.0
			marker_msg.scale.y = 1.0
			marker_msg.scale.z = 1.0

			marker_msg.color.a = 1.0
			marker_msg.color.r = 1.0
			marker_msg.color.g = 1.0
			marker_msg.color.b = 1.0

			marker_array.markers.append(marker_msg)
		
		if(len(marker_array) > 0):
			self.publisher_rviz_positions.publish(marker_array)

	# ******************************************
	def publish_position(self, marker_id_in, position_in):
		marker_msg = Marker()
		marker_msg.header.frame_id = "map"
		marker_msg.id = int(marker_id_in)
		marker_msg.type = Marker.SPHERE
		marker_msg.action = Marker.ADD

		marker_msg.pose.position.x = position_in[0]
		marker_msg.pose.position.y = position_in[1]
		marker_msg.pose.position.z = position_in[2]

		marker_msg.pose.orientation.w = 1.0

		marker_msg.scale.x = 1.0
		marker_msg.scale.y = 1.0
		marker_msg.scale.z = 1.0

		marker_msg.color.a = 1.0

		marker_msg.color.r = 1.0
		marker_msg.color.g = 1.0
		marker_msg.color.b = 1.0
	
		self.publisher_rviz_position.publish(marker_msg)

# ******************************************
def main(args=None):
	rclpy.init(args=args)

	aruco_detector = ArucoDetector()
	aruco_detector.start()

	rclpy.spin(aruco_detector)

	aruco_detector.destroy_node()
	rclpy.shutdown()

# ******************************************
if __name__ == '__main__':
	main()
