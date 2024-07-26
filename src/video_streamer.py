#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

import signal, sys
from threading import Event, Lock, Thread

from flask import Flask, render_template, Response

import time

class ImageSubscriber(Node):

	# ******************************************
	def __init__(self):
		super().__init__('image_subscriber')
		self.image_aruco = None
		self.bridge = CvBridge()
		self.lock = Lock()

		self.fps = 0
		self.frame_count = 0
		self.time_start = time.time()
		
		self.subscription = self.create_subscription(
			Image,
			"/image_aruco",
			self.callback_an_image_processed,
			10)

	# ******************************************
	def callback_an_image_processed(self, msg):
		self.frame_count += 1
		time_now = time.time()
		time_delta = time_now - self.time_start

		if time_delta >= 1.0:
			self.fps = self.frame_count / time_delta
			self.frame_count = 0
			self.time_start = time_now
			self.get_logger().info(f"callback_an_image_processed fps: {self.fps}")
			
		with self.lock:
			self.image_aruco = msg

	# ******************************************
	def get_frame(self):
		image_aruco = None
		with self.lock:
			image_aruco = self.image_aruco
			self.image_aruco = None

		if (image_aruco is None):
			return None

		cv_image = self.bridge.imgmsg_to_cv2(image_aruco, desired_encoding = "passthrough")
		frame = cv2.imencode(".jpg", cv_image)[1].tobytes()
		return frame

rclpy.init(args = None)
image_subscriber = ImageSubscriber()
thread_image_subscriber = Thread(target = rclpy.spin, args = (image_subscriber,))
thread_image_subscriber.start()

def generate_frames():
	# frame_count = 0
	# fps = 0
	# time_start = time.time()

	while True:
		frame = image_subscriber.get_frame()

		if (frame is not None):
			# frame_count += 1
			# time_now = time.time()
			# time_delta = time_now - time_start

			# if time_delta >= 1.0:
			# 	fps = frame_count / time_delta
			# 	frame_count = 0
			# 	time_start = time_now
			# 	print(f"generate_frames fps: {fps}")
				
			yield (	b'--frame\r\n'
						b'Content-Type: image/jpeg\r\n\r\n' + frame + 
						b'\r\n')
		time.sleep(0.02)

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/video_feed')
def video_feed():
	return Response(
		generate_frames(),
		mimetype='multipart/x-mixed-replace; boundary=frame'
	)

def signal_handler(signal, frame):
	rclpy.shutdown()
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main(args=None):
	app.run(host='0.0.0.0', port=8080)

if __name__ == '__main__':
	main()
