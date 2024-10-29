import numpy as np
from numpy.polynomial.polynomial import polyval
from sklearn.preprocessing import StandardScaler
import pyro.distributions as dist
import torch

from src.models.utils import lb_cdf_norm, ub_cdf_norm, lb_icdf_norm, ub_icdf_norm, lb_cdf_generic, ub_cdf_generic, lb_icdf_generic, ub_icdf_generic

class Sine:
    """
    Synthetic data using the SCM from https://arxiv.org/pdf/2311.11321.pdf
    """

    def __init__(self, n_samples_train=1000, n_samples_test=1000, mode='normal', **kwargs):
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.mode = mode

        if self.mode == 'multimodal':
            weigths_dist = dist.OneHotCategorical(probs=torch.tensor([0.7, 0.3]))
            comp_dist = dist.Normal(torch.tensor([-0.5, 1.5]), torch.tensor([1.5, 0.5]))
            self.eps0 = dist.MixtureSameFamily(weigths_dist._categorical, comp_dist)

            weigths_dist = dist.OneHotCategorical(probs=torch.tensor([0.3, 0.4, 0.3]))
            comp_dist = dist.Normal(torch.tensor([-2.5, 0.5, 2.0]), torch.tensor([0.35, 0.75, 0.5]))
            self.eps1 = dist.MixtureSameFamily(weigths_dist._categorical, comp_dist)

            self.out_f_bounds = None

        self.out_scaler = StandardScaler()

    def get_data(self):
        data_dicts = []
        for i, n_samples in enumerate([self.n_samples_train, self.n_samples_test]):
            u = np.random.normal(0.0, 1.0, n_samples)
            x = np.random.uniform(-2.0, 2.0, n_samples)
            t = np.random.binomial(1, 1.0 / (1.0 + np.exp(- (0.75 * x - u + 0.5))), n_samples)

            if self.mode == 'normal':
                y_pot0 = self.get_mu(0, x, u) + np.random.normal(0.0, 1.0, n_samples)
                y_pot1 = self.get_mu(1, x, u) + np.random.normal(0.0, 1.0, n_samples)
            elif self.mode == 'multimodal':
                y_pot0 = self.get_mu(0, x, u) + self.eps0.sample((n_samples,)).cpu().numpy()
                y_pot1 = self.get_mu(1, x, u) + self.eps1.sample((n_samples,)).cpu().numpy()
            elif self.mode == 'exp':
                y_pot0 = dist.Exponential(torch.tensor(1 / np.abs(self.get_mu(0, x, u)))).sample().cpu().numpy()
                y_pot1 = dist.Exponential(torch.tensor(1 / np.abs(self.get_mu(1, x, u)))).sample().cpu().numpy()
            else:
                raise NotImplementedError()

            y = y_pot0 * (1 - t) + y_pot1 * t

            if i == 0:
                self.out_scaler.fit(y.reshape(-1, 1))
                self.out_f_bounds = (np.concatenate([y_pot0, y_pot1]).min(), np.concatenate([y_pot0, y_pot1]).max())

            y_scaled = self.out_scaler.transform(y.reshape(-1, 1))

            # y = (2 * t - 1) * x + (2 * t - 1) - 2 * np.sin(2 * x + u) - 2 * u * (1 + 0.5 * x) + y_eps
            data_dicts.append({
                'cov_f': np.stack([x, u], -1),
                'treat_f': t,
                'out_f': y,
                'out_f_scaled': y_scaled,
                'out_scaler.scale_': float(self.out_scaler.scale_),
                'out_pot0': y_pot0,
                'out_pot1': y_pot1,
                'mu0': self.get_mu(0, x, u).reshape(-1),
                'mu1': self.get_mu(1, x, u).reshape(-1),
            })
        return data_dicts

    def get_mu(self, treat, x, u):
        if treat == 0:
            return - 1 * x - 2 * np.sin(2 * x + u) - 2 * u * (1 + 0.5 * x)
        else:
            return 1 * x + 1 - 2 * np.sin(2 * x + u) - 2 * u * (1 + 0.5 * x)

    def get_bounds(self, delta_or_alpha, data_dict, mode='cdf', n_grid=500):
        if self.mode == 'normal':
            if mode == 'cdf':
                return lb_cdf_norm(delta_or_alpha, loc0=data_dict['mu0'], scale0=1.0, loc1=data_dict['mu1'], scale1=1.0), \
                       ub_cdf_norm(delta_or_alpha, loc0=data_dict['mu0'], scale0=1.0, loc1=data_dict['mu1'], scale1=1.0)
            elif mode == 'icdf':
                return lb_icdf_norm(delta_or_alpha, loc0=data_dict['mu0'], scale0=1.0, loc1=data_dict['mu1'], scale1=1.0), \
                       ub_icdf_norm(delta_or_alpha, loc0=data_dict['mu0'], scale0=1.0, loc1=data_dict['mu1'], scale1=1.0)
            else:
                raise NotImplementedError()
        elif self.mode == 'multimodal':
            delta_or_alpha = delta_or_alpha.reshape(-1, 1, 1)
            mu0 = torch.tensor(data_dict['mu0']).reshape(1, 1, -1).repeat(delta_or_alpha.shape[0], n_grid, 1)
            mu1 = torch.tensor(data_dict['mu1']).reshape(1, 1, -1).repeat(delta_or_alpha.shape[0], n_grid, 1)
            p0, p1 = (self.eps0.rv + mu0).dist, (self.eps1.rv + mu1).dist
            out_bounds = (min(self.out_f_bounds[0], delta_or_alpha.min()), max(self.out_f_bounds[1], delta_or_alpha.max()))

            if mode == 'cdf':
                return lb_cdf_generic(delta_or_alpha, p0=p0, p1=p1, out_bounds=out_bounds, n_grid=n_grid), \
                       ub_cdf_generic(delta_or_alpha, p0=p0, p1=p1, out_bounds=out_bounds, n_grid=n_grid)
            elif mode == 'icdf':
                return np.array(np.nan), np.array(np.nan)
            else:
                raise NotImplementedError()
        elif self.mode == 'exp':
            delta_or_alpha = delta_or_alpha.reshape(-1, 1, 1)
            mu0 = torch.tensor(data_dict['mu0']).reshape(1, 1, -1).repeat(delta_or_alpha.shape[0], n_grid, 1)
            mu1 = torch.tensor(data_dict['mu1']).reshape(1, 1, -1).repeat(delta_or_alpha.shape[0], n_grid, 1)
            p0 = dist.Exponential(1 / mu0.abs())
            p1 = dist.Exponential(1 / mu1.abs())
            # out_bounds = (min(self.out_f_bounds[0], delta_or_alpha.min()), max(self.out_f_bounds[1], delta_or_alpha.max()))
            if mode == 'cdf':
                return lb_cdf_generic(delta_or_alpha, p0=p0, p1=p1, out_bounds=self.out_f_bounds, n_grid=n_grid, set_neg_to_zero=True), \
                       ub_cdf_generic(delta_or_alpha, p0=p0, p1=p1, out_bounds=self.out_f_bounds, n_grid=n_grid, set_neg_to_zero=True)
            elif mode == 'icdf':
                return lb_icdf_generic(delta_or_alpha, p0=p0, p1=p1, n_grid=n_grid), \
                       ub_icdf_generic(delta_or_alpha, p0=p0, p1=p1, n_grid=n_grid)
            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError()
