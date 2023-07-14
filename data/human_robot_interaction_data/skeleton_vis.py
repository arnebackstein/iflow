# -*- coding:utf-8 -*-

# -----------------------------------
# 3D Skeleton Display
# Author: souljaboy764
# Date: 2021/5/23
# -----------------------------------

import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from read_hh_hr_data import *

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

def visualize_skeleton(data, variant=None, action=None, save_path=None):
	fig = plt.figure()
	ax = Axes3D(fig)
	plt.ion()

	print('data',data.shape)
	idx_list = np.linspace(0,data.shape[0],70,endpoint=False).astype(int)
	data = data[idx_list]
	data = normal_skeleton(data)
	
	ax.view_init(0, -90)
	# ax.grid(False)
	ax.set_xlabel('X')
	ax.set_ylabel('Z')
	ax.set_zlabel('Y')
	# ax.set_axis_bgcolor('white')w
	for frame_idx in range(data.shape[0]):
		ax.cla()
		ax.set_xlabel('X')
		ax.set_ylabel('Z')
		ax.set_zlabel('Y')
		ax.set_facecolor('none')
		ax.set_xlim3d([-1, 1])
		ax.set_ylim3d([-1, 1])
		ax.set_zlim3d([-0.8, 0.8])

		# ax.axis('off')
		if variant is not None and action is not None:
			# ax.set_title('_'.join(variant.split('/')[0]) + " " + action)
			ax.set_title(variant + " " + action)
			print(variant,action,"Frame: {}".format(frame_idx))

		x = data[frame_idx, :, 0]
		y = data[frame_idx, :, 1]
		z = data[frame_idx, :, 2]
		ax.scatter(x, z, y, color='r', marker='o')
		
		# for part in body:
		for part in connections:
			x_plot = x[part]
			y_plot = y[part]
			z_plot = z[part]
			ax.plot(x_plot, z_plot, y_plot, color='b')
		

		if save_path is not None:
			plt.savefig(os.path.join(save_path, '%.4d.png'%(frame_idx)))
		plt.pause(0.01)
		if not plt.fignum_exists(1):
			break
	plt.ioff()
	plt.show()
	

if __name__ == '__main__':
	data_p, data_q, names, times = read_data('hh/p1/rocket_s1_1.csv')
	segments = np.load('hh/segmentation/rocket_1.npy')
	for s in segments:
		visualize_skeleton(data_p[s[0]:s[1]])
