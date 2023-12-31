import csv
import numpy as np


joints = ['Root', 'Hips', 'LeftThigh', 'LeftShin', 'LeftFoot', 'LeftToe',
		  'LeftToeTip', 'RightThigh', 'RightShin', 'RightFoot', 'RightToe',
		  'RightToeTip', 'Spine1', 'Spine2', 'Spine3', 'Spine4',
		  'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'Neck',
		  'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']

joints_dic = {}
c = 0
for j in joints:
	joints_dic[j] = c
	c = c + 1 

connections = [[1,12],[12,13],[13,14],[14,15],[15,20],[20,21],  # torso
			   [15,16],[16,17],[17,18],[18,19],                 # left arm
			   [15,22],[22,23],[23,24],[24,25],                 # right arm
			   [1,2],[2,3],[3,4],[4,5],[5,6],                   # left leg
			   [1,7],[7,8],[8,9],[9,10],[10,11]]                # right leg

	
 
def extract_indices(names):
	
	joints_p = []
	joints_q = []
	c = 0
	for i in names:
		if i[-3]  == 'Q':   
			joints_q.append(c)
		if i[-3]  == 'P':   
			joints_p.append(c)
		c = c + 1
	return joints_p, joints_q

def extract_robot_indices(names):
	
	L_joints_p = []
	L_joints_v = []
	L_cart     = []
	R_joints_p = []
	R_joints_v = []
	R_cart     = []
	c = 0
	for i in names:
		if len(i)>0:
			if i[0]  == 'L':
				if i[-2]  == 'V':   
					L_joints_v.append(c)
				if i[-2]  == 'P':   
					L_joints_p.append(c)
				if i[-2]  == 'T':   
					L_cart.append(c)
			if i[0]  == 'R':
				if i[-2]  == 'V':   
					R_joints_v.append(c)
				if i[-2]  == 'P':   
					R_joints_p.append(c)
				if i[-2]  == 'T':   
					R_cart.append(c)
		c = c + 1
	return L_joints_p, L_joints_v, L_cart, R_joints_p, R_joints_v, R_cart

	
 
def read_data(path):
	data_p = []
	data_q = []
	with open(path, 'rt') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		c = 0
		times = []
		for row in reader:
			if c == 0:
				names = row[0].split(',')
				p_idx, q_idx = extract_indices(names)
				c = 1
			else:
				new_row  = np.array(row[0].split(',')).astype('float')
				new_row_p = np.reshape(new_row[p_idx],(-1,3))
				new_row_q = np.reshape(new_row[q_idx],(-1,4))
				data_p.append(new_row_p)
				data_q.append(new_row_q)
				times.append( np.reshape(new_row[0],(1)))
	data_p = np.array(data_p)
	data_q = np.array(data_q)
	times = np.array(times)
	return  data_p, data_q, names, times



def read_robot_data(path):
	data_p_L = []
	data_p_R = []
	data_v_L = []
	data_v_R = []
	data_c_L = []
	data_c_R = []
	with open(path, 'rt') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		c = 0
		times = []
		for row in reader:
			if c == 0:
				names = row[0].split(',')
				L_p_idx, L_v_idx, L_c_idx, R_p_idx, R_v_idx, R_c_idx = extract_robot_indices(names)
				c = 1
			else:
				new_row   = np.array(row[0].split(',')[:-1]).astype('float')
				data_p_L.append(new_row[L_p_idx])
				data_v_L.append(new_row[L_v_idx])
				data_c_L.append(new_row[L_c_idx])
				data_p_R.append(new_row[R_p_idx])
				data_v_R.append(new_row[R_v_idx])
				data_c_R.append(new_row[R_c_idx])
				times.append( np.reshape(new_row[0],(1)))
	data_p_L = np.array(data_p_L)
	data_p_R = np.array(data_p_R)
	data_v_L = np.array(data_v_L) 
	data_v_R = np.array(data_v_R)
	data_c_L = np.array(data_c_L)
	data_c_R = np.array(data_c_R)
	times = np.array(times)
	return  data_p_L, data_p_R, data_v_L, data_v_R, data_c_L, data_c_R, names, times


def downsample(data, downsample_len):
	"""
	data: Input trajectory of Shape (LEN x NUM_JOINTS x JOINT_DIMS)
	"""
	import torch
	from torch.nn.functional import grid_sample, affine_grid
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	seq_len, joints, dims = data.shape
	theta = torch.Tensor(np.array([[[1,0,0.], [0,1,0]]])).to(device).repeat(joints,1,1)

	traj = data.transpose(1,2,0) # joints, dims, seq_len
	traj = torch.Tensor(traj).to(device).unsqueeze(2) # joints, dims, 1 seq_len
	traj = torch.concat([traj, torch.zeros_like(traj)], dim=2) # joints, dims, 2 seq_len
	
	grid = affine_grid(theta, torch.Size([joints, dims, 2, downsample_len]), align_corners=True)
	traj = grid_sample(traj.type(torch.float32), grid, align_corners=True) # joints, dims, 2 new_length
	traj = traj[:, :, 0].cpu().detach().numpy() # joints, dims, downsample_len
	return traj.transpose(2,0,1) # downsample_len, joints, dims

def read_hri_data(action, src_dir = './hr/'):
	import os
	data_file_p1 = os.path.join(src_dir, 'p1', action+'_s1_1.csv')
	data_file_r2 = os.path.join(src_dir, 'r2', action+'.csv')
	_, data_r2, _, _, _, _, _, _ = read_robot_data(data_file_r2)
	data_p1, _, _, _ = read_data(data_file_p1)
	data_p1 = downsample(data_p1, len(data_r2))
	return data_p1, data_r2