import numpy as np
from torch import nn
import torch
from copy import deepcopy
from sklearn.gaussian_process.kernels import RBF
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from ray import tune
from copy import deepcopy
from scipy.stats import norm, expon


def subset_by_indices(data_dict: dict, indices: list):
    subset_data_dict = {}
    for (k, v) in data_dict.items():
        if not isinstance(data_dict[k], float):
            subset_data_dict[k] = np.copy(data_dict[k][indices])
        else:
            subset_data_dict[k] = deepcopy(data_dict[k])
    return subset_data_dict


def fit_eval_kfold(args: dict, orig_hparams: DictConfig, model_cls, train_data_dict: dict, val_data_dict: dict,
                   kind: str = None, **kwargs):
    """
    Globally defined method, used for ray tuning
    :param args: Hyperparameter configuration
    :param orig_hparams: DictConfig of original hyperparameters
    :param model_cls: class of model
    :param kwargs: Other args
    """
    new_params = deepcopy(orig_hparams)
    model_cls.set_hparams(new_params['model'], args)
    # model_cls.set_subnet_hparams(new_params[subnet_name], args)
    new_params.exp.device = 'cpu'  # Tuning only with cpu

    if val_data_dict is None:  # KFold hparam tuning
        kf = KFold(n_splits=5, random_state=orig_hparams.exp.seed, shuffle=True)
        val_metrics = []
        for train_index, val_index in kf.split(train_data_dict['cov_f']):
            ttrain_data_dict, val_data_dict = subset_by_indices(train_data_dict, train_index), \
                                              subset_by_indices(train_data_dict, val_index)

            model = model_cls(new_params, kind=kind, **kwargs)
            model.fit(train_data_dict=ttrain_data_dict, log=False)
            log_dict = model.evaluate(data_dict=val_data_dict, log=False, prefix='val')
            val_metrics.append(log_dict[model.val_metric])
        tune.report(val_metric=np.mean(val_metrics))

    else:  # predefined hold-out hparam tuning
        model = model_cls(new_params, kind=kind, **kwargs)
        model.fit(train_data_dict=train_data_dict, log=False)
        log_dict = model.evaluate(data_dict=val_data_dict, log=False, prefix='val')
        tune.report(val_metric=log_dict[model.val_metric])


def lb_cdf_norm(delta, loc0, scale0, loc1, scale1):
    if scale0 != scale1:
        s = delta - (loc1 - loc0)
        t = np.sqrt(s ** 2 + (scale1 ** 2 - scale0 ** 2) * np.log(scale1 ** 2 / scale0 ** 2))
        return norm.cdf((scale1 * s - scale0 * t) / (scale1 ** 2 - scale0 ** 2), loc=0, scale=1) + norm.cdf(
            (scale1 * t - scale0 * s) / (scale1 ** 2 - scale0 ** 2), loc=0, scale=1) - 1
    else:
        return np.where(delta < (loc1 - loc0), 0.0, 2 * norm.cdf((delta - (loc1 - loc0)) / (2 * scale1), loc=0, scale=1) - 1)


def lb_icdf_norm(alpha, loc0, scale0, loc1, scale1):
    if scale0 == scale1:
        assert (alpha != 0.0).all()
        return loc1 - loc0 + 2 * scale0 * norm.ppf((1 + alpha) / 2, loc=0, scale=1)
    else:
        return NotImplementedError()


def lb_cdf_generic(delta, p0, p1, out_bounds, n_grid=200, set_neg_to_zero=False):
    out_grid = torch.linspace(out_bounds[0], out_bounds[1], n_grid).reshape(1, -1, 1)
    delta = torch.tensor(delta)
    if set_neg_to_zero:
        out_grid_ = torch.where(out_grid < 0.0, 0.0, out_grid)
        out_grid_min_delta_ = torch.where((out_grid - delta) < 0.0, 0.0, out_grid - delta)
        return np.maximum((p1.cdf(out_grid_) - p0.cdf(out_grid_min_delta_)).max(1)[0].cpu().numpy(), 0)
    else:
        return np.maximum((p1.cdf(out_grid) - p0.cdf(out_grid - delta)).max(1)[0].cpu().numpy(), 0)


def lb_icdf_generic(alpha, p0, p1, n_grid=200):
    quant_grid = torch.linspace(0.0, 1.0, n_grid + 2)[1:-1].reshape(1, -1, 1)
    alpha = torch.tensor(alpha)
    return (p1.icdf(quant_grid) - p0.icdf(quant_grid - alpha)).nan_to_num(1e6).min(1)[0].cpu().numpy()


def ub_icdf_norm(alpha, loc0, scale0, loc1, scale1):
    if scale0 == scale1:
        assert (alpha != 1.0).all()
        return loc1 - loc0 + 2 * scale0 * norm.ppf(alpha / 2, loc=0, scale=1)
    else:
        return NotImplementedError()


def ub_cdf_norm(delta, loc0, scale0, loc1, scale1):
    if scale0 != scale1:
        s = delta - (loc1 - loc0)
        t = np.sqrt(s ** 2 + (scale1 ** 2 - scale0 ** 2) * np.log(scale1 ** 2 / scale0 ** 2))
        return norm.cdf((scale1 * s + scale0 * t) / (scale1 ** 2 - scale0 ** 2), loc=0, scale=1) - norm.cdf(
            (scale1 * t + scale0 * s) / (scale1 ** 2 - scale0 ** 2), loc=0, scale=1) + 1
    else:
        return np.where(delta < (loc1 - loc0), 2 * norm.cdf((delta - (loc1 - loc0)) / (2 * scale1), loc=0, scale=1), 1.0)


def ub_cdf_generic(delta, p0, p1, out_bounds, n_grid=200, set_neg_to_zero=False):
    out_grid = torch.linspace(out_bounds[0], out_bounds[1], n_grid).reshape(1, -1, 1)
    delta = torch.tensor(delta)
    if set_neg_to_zero:
        out_grid_ = torch.where(out_grid < 0.0, 0.0, out_grid)
        out_grid_min_delta_ = torch.where((out_grid - delta) < 0.0, 0.0, out_grid - delta)
        return 1 + np.minimum((p1.cdf(out_grid_) - p0.cdf(out_grid_min_delta_)).min(1)[0].cpu().numpy(), 0)
    return 1 + np.minimum((p1.cdf(out_grid) - p0.cdf(out_grid - delta)).min(1)[0].cpu().numpy(), 0)


def ub_icdf_generic(alpha, p0, p1, n_grid=200):
    quant_grid = torch.linspace(0.0, 1.0, n_grid + 2)[1:-1].reshape(1, -1, 1)
    alpha = torch.tensor(alpha)
    return (p1.icdf(quant_grid) - p0.icdf(1 + quant_grid - alpha)).nan_to_num(-1e6).max(1)[0].cpu().numpy()


class NormalizedRBF:
    """
    Normalized radial basis function, used for KDE and DKME
    """
    def __init__(self, sd):
        self.sd = sd
        self.rbf = RBF(np.sqrt(self.sd / 2))

    def __call__(self, x1, x2):
        return 1 / np.sqrt(np.pi * self.sd) * self.rbf(x1, x2)

