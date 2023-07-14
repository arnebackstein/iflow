# -*- coding:utf-8 -*-

# -----------------------------------
# 3D Skeleton Marker Publisher
# Author: souljaboy764
# Date: 2021/5/23
# -----------------------------------

import numpy as np


from read_hh_hr_data import *

import rospy
import tf2_ros
from tf2_msgs.msg import *
from geometry_msgs.msg import *
from visualization_msgs.msg import *

def normal_skeleton(data):
	#  use as center joint
	center_joint = data[0]
	
	center_jointx = np.mean(center_joint[:, 0])
	center_jointy = np.mean(center_joint[:, 1])
	center_jointz = np.mean(center_joint[:, 2])

	center = np.array([center_jointx, center_jointy, center_jointz])
	data = data - center

	return data

def rotation(data, alpha=0, beta=0):
	# rotate the skeleton around x-y axis
	r_alpha = alpha * np.pi / 180
	r_beta = beta * np.pi / 180

	rx = np.array([[1, 0, 0],
					[0, np.cos(r_alpha), -1 * np.sin(r_alpha)],
					[0, np.sin(r_alpha), np.cos(r_alpha)]]
					)

	ry = np.array([
		[np.cos(r_beta), 0, np.sin(r_beta)],
		[0, 1, 0],
		[-1 * np.sin(r_beta), 0, np.cos(r_beta)],
	])

	r = ry.dot(rx)
	data = data.dot(r)

	return data

if __name__ == '__main__':
	rospy.init_node('butepage_skeleton_node')
	skeleton_marker_array_pub = rospy.Publisher("/butepage_skeleton_viz", MarkerArray, queue_size=10)
	br = tf2_ros.TransformBroadcaster()
	rate = rospy.Rate(10)
	tfs = []
	marker_array = MarkerArray()
	for j in joints:
		tfs.append(TransformStamped())
		tfs[-1].header.seq = 0
		tfs[-1].header.frame_id = "base_footprint"
		tfs[-1].child_frame_id = "butepage_skeleton_1_"+j

		marker = Marker()
		marker.ns = "nturgbd_skeleton"
		marker.id = len(tfs)
		marker.lifetime = rospy.Duration(0.5)
		
		marker.header.frame_id = tfs[-1].child_frame_id
		marker.frame_locked = False
		marker.type = Marker.CUBE
		marker.action = Marker.ADD

		marker.color.a = 1
		marker.color.r = 1
		marker.color.g = 0
		marker.color.b = 0

		if j in joints[-4:]:
			marker.color.r = 0
			marker.color.g = 1

		marker.scale.x = 0.05
		marker.scale.y = 0.05
		marker.scale.z = 0.05

		marker_array.markers.append(marker)
	

	
	data_p, data_q, names, times = read_data('hh/p1/hand_shake_s1_1.csv')
	segments = np.load('hh/segmentation/hand_shake_1.npy')
	for s in segments:
		data = data_p[s[0]:s[1]]

		print('data',data.shape)
		idx_list = np.linspace(0,data.shape[0],70,endpoint=False).astype(int)
		data = data[idx_list]
		data = normal_skeleton(data)

		for frame_idx in range(data.shape[0]):
			y = -data[frame_idx, :, 0]
			z = data[frame_idx, :, 1]
			x = data[frame_idx, :, 2]
			stamp = rospy.Time.now()
			
			for joint_idx in range(len(joints)):
				tfs[joint_idx].transform.translation.x = x[joint_idx] - x[0]
				tfs[joint_idx].transform.translation.y = y[joint_idx] - y[0]
				tfs[joint_idx].transform.translation.z = z[joint_idx] - z[0]
				
				tfs[joint_idx].transform.rotation.x = tfs[joint_idx].transform.rotation.y = tfs[joint_idx].transform.rotation.z = 0
				tfs[joint_idx].transform.rotation.w = 1.

				tfs[joint_idx].header.stamp = stamp
			
			br.sendTransform(tfs)
			skeleton_marker_array_pub.publish(marker_array)
			rate.sleep()
			if rospy.is_shutdown():
				break

		if rospy.is_shutdown():
			break
