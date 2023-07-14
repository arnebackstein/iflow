# -*- coding:utf-8 -*-

# -----------------------------------
# HRI Data Plotter
# Author: souljaboy764
# Date: 2022/07/06
# -----------------------------------

import numpy as np
from read_hh_hr_data import *
import os
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import grid_sample, affine_grid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

src_dir = './hr'
actions = ['hand_wave', 'hand_shake', 'rocket', 'parachute']
idx_list = np.array([joints_dic[joint] for joint in ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']])
# action = actions[3]
for action in actions:
	data_h, data_r = read_hri_data(action, src_dir = './hr/')
	seq_len, _, dims = data_h.shape
	data_h = data_h[:, idx_list].reshape((-1, len(idx_list)*dims))
	segments_file_h = os.path.join(src_dir, 'segmentation', action+'_p1.npy')
	segments_file_r = os.path.join(src_dir, 'segmentation', action+'_r2.npy')
	segments_h = np.load(segments_file_h)
	segments_r = np.load(segments_file_r)

	for i in range(len(segments_h)):
		s_h = segments_h[i]
		s_r = segments_r[i]
		ax1 = plt.subplot(211)
		ax2 = plt.subplot(212)
		for dim in range(len(data_h.T)):
			ax1.plot(data_h[s_h[0]:s_h[1], dim])
		for dim in range(len(data_r.T)):
			ax2.plot(data_r[s_r[0]:s_r[1], dim])
		plt.show()
