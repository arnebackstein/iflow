import numpy as np
import torch
from sklearn.decomposition import PCA
import scipy.fftpack


class HRVelCycleDataset(torch.utils.data.Dataset):


    """
    trajs_h: np.array of shape (n_demonstrations, n_timesteps, n_dimensions)
    trajs_r: np.array of shape (n_demonstrations, n_timesteps, n_dimensions)
    """
    def __init__(self, trajs_h, trajs_r, dim_h, dim_r, device = torch.device('cpu'), steps=20):

        self.trajs_h = trajs_h
        self.trajs_r = trajs_r
        self.dim_h = dim_h
        self.dim_r = dim_r

        
        dim = dim_h + dim_r

        trajs_np = np.concatenate([trajs_h, trajs_r], axis=2)

        ## Define Variables and Load trajectories ##
        self.type = type
        self.dim = dim
        self.dt = .01

        self.trajs_real = np.array(trajs_np, copy=True)
        self.n_trajs = trajs_np.shape[0]
        self.n_steps = trajs_np.shape[1]
        self.n_dims = trajs_np.shape[2]

        ## Normalize Trajectories
        trajs_np = np.reshape(trajs_np, (self.n_trajs * self.n_steps, self.n_dims))
        self.mean = np.mean(trajs_np, axis=0)
        self.std = np.std(trajs_np, axis=0)

        self.train_data = self.normalize(self.trajs_real)

        ### Mean Angular velocity ###
        self.w = self.get_mean_ang_vel()

        self.train_phase_data = []
        for i in range(len(self.train_data)):
            trj = self.train_data[0]
            N = trj.shape[0]
            t = np.linspace(0,N*self.dt,N)
            phase_trj = np.arctan2(np.sin(self.w*t),np.cos(self.w*t))
            self.train_phase_data.append(phase_trj)

        # initialization
        self.x = []
        self.x_n = np.zeros((0, dim))
        for i in range(steps):
            tr_i_all = np.zeros((0,dim))
            for tr_i in self.train_data:
                _trj = tr_i[i:i-steps,:]
                tr_i_all = np.concatenate((tr_i_all, _trj), 0)
                self.x_n = np.concatenate((self.x_n, tr_i[-1:,:]),0)
            self.x.append(tr_i_all)

        self.x = torch.from_numpy(np.array(self.x)).float().to(device)
        self.x_n = torch.from_numpy(np.array(self.x_n)).float().to(device)

        ## Phase ordering ##
        trp_all = np.zeros((0))
        for trp_i in self.train_phase_data:
            _trjp = trp_i[:-steps]
            trp_all = np.concatenate((trp_all, _trjp), 0)
        self.trp_all = torch.from_numpy(trp_all).float().to(device)

        self.len_n = self.x_n.shape[0]
        self.len = self.x.shape[1]
        self.steps_length = steps
        self.step = steps - 1


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

    def get_human_data(self):
        return self.train_data[:, :, :self.dim_h]

    def get_robot_data(self):
        return self.train_data[:, :, self.dim_h:]
    
    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def set_step(self, step=None):
        if step is None:
            self.step = np.random.randint(1, self.steps_length-1)

    def __getitem__(self, index):
        'Generates one sample of data'

        X = self.x[0, index, :]
        X_human = self.x[0, index, :self.dim_h]
        X_robot = self.x[0, index, self.dim_h:]

        X_1 = self.x[self.step, index, :]
        X_1_human = self.x[self.step, index, :self.dim_h]
        X_1_robot = self.x[self.step, index, self.dim_h:]
        phase = self.trp_all[index]

        return (X_human, X_robot), [(X_1_human, X_1_robot), int(self.step), phase]
