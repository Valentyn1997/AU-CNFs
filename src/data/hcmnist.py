# Credits to https://github.com/anndvision/quince

import torch
import typing
import numpy as np
from torchvision import datasets
from sklearn import model_selection, preprocessing
from src.models.utils import lb_cdf_norm, ub_cdf_norm, lb_icdf_norm, ub_icdf_norm

from src import ROOT_PATH


class HCMNISTSubset(datasets.MNIST):
    """
    HC-MNIST semi-synthetic dataset
    Jesson, A., Mindermann, S., Gal, Y., and Shalit, U. Quantifying ignorance in individual-level causal-effect estimates under
    hidden confounding. In International Conference on Machine Learning, 2021.
    """

    def __init__(
        self,
        root: str = f'{ROOT_PATH}/data/hcmnist',
        gamma_star: float = np.exp(1),
        split: str = "train",
        mode: str = "mu",
        p_u: str = "bernoulli",
        theta: float = 4.0,
        beta: float = 0.75,
        sigma_y: float = 1.0,
        domain: float = 2.0,
        seed: int = 1331,
        transform: typing.Optional[typing.Callable] = None,
        target_transform: typing.Optional[typing.Callable] = None,
        download: bool = True,
        **kwargs
    ) -> None:
        train = split == "train" or split == "valid"
        self.__class__.__name__ = "MNIST"
        super(HCMNISTSubset, self).__init__(root, train=train, transform=transform, target_transform=target_transform,
                                      download=download)
        self.data = self.data.view(len(self.targets), -1).numpy()
        self.targets = self.targets.numpy()

        if train:
            data_train, data_valid, targets_train, targets_valid = \
                model_selection.train_test_split(self.data, self.targets, test_size=0.3, random_state=seed)
            self.data = data_train if split == "train" else data_valid
            self.targets = targets_train if split == "train" else targets_valid

        self.mode = mode
        self.dim_input = [1, 28, 28]
        self.dim_treatment = 1
        self.dim_output = 1

        self.phi_model = fit_phi_model(root=root, edges=torch.arange(-domain, domain + 0.1, (2 * domain) / 10))

        size = (self.__len__(), 1)
        rng = np.random.RandomState(seed=seed)
        if p_u == "bernoulli":
            self.u = rng.binomial(1, 0.5, size=size).astype("float32")
        elif p_u == "uniform":
            self.u = rng.uniform(size=size).astype("float32")
        elif p_u == "beta_bi":
            self.u = rng.beta(0.5, 0.5, size=size).astype("float32")
        elif p_u == "beta_uni":
            self.u = rng.beta(2, 5, size=size).astype("float32")
        else:
            raise NotImplementedError(f"{p_u} is not a supported distribution")

        phi = self.phi
        self.pi = (complete_propensity(x=phi, u=self.u, gamma=gamma_star, beta=beta).astype("float32").ravel())
        self.t = rng.binomial(1, self.pi).astype("float32")
        self.mu0 = (f_mu(x=phi, t=0.0, u=self.u, theta=theta).astype("float32").ravel())
        self.mu1 = (f_mu(x=phi, t=1.0, u=self.u, theta=theta).astype("float32").ravel())
        self.y0 = self.mu0 + (sigma_y * rng.normal(size=self.t.shape)).astype("float32")
        self.y1 = self.mu1 + (sigma_y * rng.normal(size=self.t.shape)).astype("float32")
        self.y = self.t * self.y1 + (1 - self.t) * self.y0
        self.tau = self.mu1 - self.mu0
        self.y_mean = np.array([0.0], dtype="float32")
        self.y_std = np.array([1.0], dtype="float32")


        self.out_scaler = preprocessing.StandardScaler()
        self.out_scaler.fit(np.concatenate([self.y0, self.y1]).reshape(-1, 1))
        self.y_scaled = self.out_scaler.transform(self.y.reshape(-1, 1))
        # self.y0 = self.out_scaler.transform(self.y0.reshape(-1, 1))
        # self.y1 = self.out_scaler.transform(self.y1.reshape(-1, 1))
        # self.mu0 = self.out_scaler.transform(self.mu0.reshape(-1, 1))
        # self.mu1 = self.out_scaler.transform(self.mu1.reshape(-1, 1))

    def __getitem__(self, index):
        x = ((self.data[index].astype("float32") / 255.0) - 0.1307) / 0.3081
        t = self.t[index: index + 1]
        if self.mode == "pi":
            return x, t
        elif self.mode == "mu":
            return np.hstack([x, t]), self.y[index: index + 1]
        else:
            raise NotImplementedError(
                f"{self.mode} not supported. Choose from 'pi'  for propensity models or 'mu' for expected outcome models"
            )

    def get_data(self) -> dict:
        return {
            'cov_f': np.concatenate([self.x, self.u], axis=1),
            'treat_f': self.t,
            'out_f': self.y,
            'out_f_scaled': self.y_scaled,
            'out_scaler.scale_': float(self.out_scaler.scale_),
            'out_pot0': self.y0,
            'out_pot1': self.y1,
            'mu0': self.mu0,
            'mu1': self.mu1,
        }

    @property
    def phi(self):
        x = ((self.data.astype("float32") / 255.0) - 0.1307) / 0.3081
        z = np.zeros_like(self.targets.astype("float32"))
        for k, v in self.phi_model.items():
            ind = self.targets == k
            x_ind = x[ind].reshape(ind.sum(), -1)
            means = x_ind.mean(axis=-1)
            z[ind] = linear_normalization(np.clip((means - v["mu"]) / v["sigma"], -1.4, 1.4), v["lo"], v["hi"])
        return np.expand_dims(z, -1)

    @property
    def x(self):
        return ((self.data.astype("float32") / 255.0) - 0.1307) / 0.3081


def fit_phi_model(root, edges):
    ds = datasets.MNIST(root=root)
    data = (ds.data.float().div(255) - 0.1307).div(0.3081).view(len(ds), -1)
    model = {}
    digits = torch.unique(ds.targets)
    for i, digit in enumerate(digits):
        lo, hi = edges[i: i + 2]
        ind = ds.targets == digit
        data_ind = data[ind].view(ind.sum(), -1)
        means = data_ind.mean(dim=-1)
        mu = means.mean()
        sigma = means.std()
        model.update({digit.item(): {"mu": mu.item(), "sigma": sigma.item(), "lo": lo.item(), "hi": hi.item()}})
    return model


class HCMNIST:

    def __init__(self, **kwargs):
        self.train_subset = HCMNISTSubset(split='train')
        self.test_subset = HCMNISTSubset(split='test')

    def get_data(self):
        return [self.train_subset.get_data(), self.test_subset.get_data()]

    def get_bounds(self, delta_or_alpha, data_dict, mode='cdf'):
        if mode == 'cdf':
            return lb_cdf_norm(delta_or_alpha, loc0=data_dict['mu0'], scale0=1.0, loc1=data_dict['mu1'], scale1=1.0), \
                   ub_cdf_norm(delta_or_alpha, loc0=data_dict['mu0'], scale0=1.0, loc1=data_dict['mu1'], scale1=1.0)
        elif mode == 'icdf':
            return lb_icdf_norm(delta_or_alpha, loc0=data_dict['mu0'], scale0=1.0, loc1=data_dict['mu1'], scale1=1.0), \
                   ub_icdf_norm(delta_or_alpha, loc0=data_dict['mu0'], scale0=1.0, loc1=data_dict['mu1'], scale1=1.0)
        else:
            raise NotImplementedError()


def alpha_fn(pi, lambda_):
    return (pi * lambda_) ** -1 + 1.0 - lambda_ ** -1


def beta_fn(pi, lambda_):
    return lambda_ * (pi) ** -1 + 1.0 - lambda_


def complete_propensity(x, u, gamma, beta=0.75):
    logit = beta * x + 0.5
    nominal = (1 + np.exp(-logit)) ** -1
    alpha = alpha_fn(nominal, gamma)
    beta = beta_fn(nominal, gamma)
    return (u / alpha) + ((1 - u) / beta)


def f_mu(x, t, u, theta=4.0):
    mu = ((2 * t - 1) * x + (2.0 * t - 1) - 2 * np.cos((4 * t - 2) * x) - (theta * u - 2) * (1 + 0.5 * x))
    return mu


def linear_normalization(x, new_min, new_max):
    return (x - x.min()) * (new_max - new_min) / (x.max() - x.min()) + new_min