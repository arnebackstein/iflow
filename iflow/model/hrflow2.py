import torch
import torch.nn as nn


class HumanRobotFlow2(nn.Module):
    def __init__(self, model_human, model_robot, dynamics, device=None, dt=0.01):
        super().__init__()
        self.device = device
        self.flow_human = model_human
        self.flow_robot = model_robot
        self.flow_backward_h, self.flow_forward_h = self.get_transforms(model_human)
        self.flow_backward_r, self.flow_forward_r = self.get_transforms(model_robot)
        self.dynamics = dynamics

    def get_transforms(self, model):

        def sample_fn(z, logpz=None, context=None):
            if logpz is not None:
                return model(z, logpz, reverse=True)
            else:
                return model(z,  reverse=True)

        def density_fn(x, logpx=None, context=None):
            if logpx is not None:
                return model(x, logpx, reverse=False)
            else:
                return model(x, reverse=False)

        return sample_fn, density_fn

    def forward(self, yt, context=None):
        yt_h, yt_r = yt

        #human
        zero = torch.zeros(yt_h.shape[0], 1).to(yt_h)
        xt_h, log_detjacobians_h = self.flow_forward_h(yt_h, zero, context=context)
        
        #robot
        zero = torch.zeros(yt_r.shape[0], 1).to(yt_r)
        xt_r, log_detjacobians_r = self.flow_forward_r(yt_r, zero, context=context)


        #zt, log_p = self.dynamics(xt,log_detjacobians)
        return xt_h, xt_r, log_detjacobians_h, log_detjacobians_r

    def generate_trj(self, y0_h, y0_r, T=100, noise=False, reverse=False):
        z0_h = self.flow_forward_h(y0_h)
        z0_r = self.flow_forward_r(y0_r)

        #z0 = torch.cat([z0_h, z0_r], dim=1)

        trj_z_h = self.dynamics.generate_trj(z0_h, T=T, reverse = reverse, noise = noise)
        trj_z_r = self.dynamics.generate_trj(z0_r, T=T, reverse = reverse, noise = noise)

        trj_y_h = self.flow_backward_h(trj_z_h[:, 0, :])
        trj_y_r = self.flow_backward_r(trj_z_r[:, 0, :])
        return trj_y_h, trj_y_r

    def generate_trj_hr(self, y0_h, y0_r, y_h, noise=False, reverse=False):
        T=y_h.shape[0]
        z_h = self.flow_forward_h(y_h)
        z0_h = self.flow_forward_h(y0_h)
        z0_r = self.flow_forward_r(y0_r)

        dim_h = y0_h.shape[1]
        dim_r = y0_r.shape[1]

        z0 = torch.cat([z0_h, z0_r], dim=1)

        #mu
        mu = self.dynamics.generate_trj(z0, T=T, reverse = reverse, noise = noise)
        mu_h = mu[:, :, :dim_h]
        mu_r = mu[:, :, dim_h:]

        #sigma
        sigma_hh = self.dynamics.var[:dim_h, :dim_h]
        sigma_hr = self.dynamics.var[:dim_h, dim_h:]
        sigma_rh = self.dynamics.var[dim_h:, :dim_h]
        sigma_rr = self.dynamics.var[dim_h:, dim_h:]


        z_mu_h = (z_h.squeeze() - mu_h.squeeze())
        z_mu_h = z_mu_h.reshape(*z_mu_h.shape, 1)

        z_r = mu_r.squeeze() + ((sigma_rh @ torch.inverse(sigma_hh)) @ z_mu_h).squeeze()
        
        y_r = self.flow_backward_r(z_r)

        return y_r

    def evolve(self, y0, T=100, noise=False, reverse=False):
        z0 = self.flow_forward(y0)
        z1 = self.dynamics.evolve(z0, T=T, reverse=reverse, noise=noise)
        y1 = self.flow_backward(z1)
        return y1
