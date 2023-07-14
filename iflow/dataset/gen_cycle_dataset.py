import numpy as np
import os
import torch
from iflow.dataset.generic_dataset import CycleDataset
from sklearn.decomposition import PCA
import scipy.fftpack


class GENCYCLE():

    def __init__(self, trajs_np, dim, device = torch.device('cpu')):
        ## Define Variables and Load trajectories ##
        self.type = type
        self.dim = dim
        self.dt = .01

        self.trajs_real=[]
        for i in range(trajs_np.shape[0]):
            self.trajs_real.append(trajs_np[i, :, :])
        self.n_trajs = trajs_np.shape[0]
        self.n_steps = trajs_np.shape[1]
        self.n_dims = trajs_np.shape[2]

        ## Normalize Trajectories
        trajs_np = np.reshape(trajs_np, (self.n_trajs * self.n_steps, self.n_dims))
        self.mean = np.mean(trajs_np, axis=0)
        self.std = np.std(trajs_np, axis=0)
        self.trajs_normalized = self.normalize(self.trajs_real)


        ## Build Train Dataset
        self.train_data = []
        for i in range(self.trajs_normalized.shape[0]):
            self.train_data.append(self.trajs_normalized[i, ...])
        ### Mean Angular velocity ###
        self.w = self.get_mean_ang_vel()

        self.train_phase_data = []
        for i in range(len(self.train_data)):
            trj = self.train_data[0]
            N = trj.shape[0]
            t = np.linspace(0,N*self.dt,N)
            phase_trj = np.arctan2(np.sin(self.w*t),np.cos(self.w*t))
            self.train_phase_data.append(phase_trj)

        self.dataset = CycleDataset(trajs=self.train_data, device=device, trajs_phase=self.train_phase_data)

    def get_mean_ang_vel(self):
        ########## PCA trajectories and Fourier Transform #############
        self.pca = PCA(n_components=2)
        self.pca.fit(self.train_data[0])
        pca_trj = self.pca.transform(self.train_data[0])

        ### Fourier Analysis
        N = pca_trj.shape[0]
        yf = scipy.fftpack.fft(pca_trj[:, 1])
        xf = np.linspace(0.0, 1. / (2 * self.dt), N // 2)

        max_i = np.argmax(np.abs(yf[:N // 2]))

        self.freq = xf[max_i]
        w = 2*np.pi * self.freq
        return w

    def normalize(self, X):
        Xn = (X - self.mean) / self.std
        return Xn

    def unormalize(self, Xn):
        X = Xn * self.std + self.mean
        return X
