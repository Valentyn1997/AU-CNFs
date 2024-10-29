import torch
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.distributions.transforms.spline import ConditionedSpline
from omegaconf import DictConfig
import logging
import numpy as np
from pyro.nn import DenseNN
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rv_continuous
from sklearn.gaussian_process import GaussianProcessRegressor
from functools import partial
from scipy.special import ndtr

from src.models import BoundsEstimator
from src.models.utils import NormalizedRBF

logger = logging.getLogger(__name__)


# Fixing the issue with not implemented sign property
@property
def sign(self):
    return torch.ones(1).float()


ConditionedSpline.sign = sign


class PluginNeuralConditionalDensityEstimator(BoundsEstimator):
    """
    Abstract class for neural plugin methods
    """
    val_metric = 'val_log_prob_f'

    def __init__(self, args: DictConfig = None, **kwargs):
        super(PluginNeuralConditionalDensityEstimator, self).__init__(args)

        self.device = args.exp.device
        self.has_prop_score = False
        self.prop_alpha, self.clip_prop = None, None
        self.cov_scaler = StandardScaler()
        self.scaled_out_f_bound = args.model.scaled_out_f_bound  # Support bounds for the scaled outcome

        # Model hyparams & Train params
        self.nuisance_hid_dim_multiplier = args.model.nuisance_hid_dim_multiplier
        self.noise_std_X, self.noise_std_Y = args.model.noise_std_X, args.model.noise_std_Y
        self.num_epochs = args.model.num_epochs
        self.num_burn_in_epochs = None
        self.num_train_iter, self.num_burn_in_train_iter = None, None  # Will be calculated later
        self.nuisance_lr = args.model.nuisance_lr
        self.nuisance_batch_size = args.model.batch_size

        self.dim_hid = self.nuisance_hid_dim_multiplier
        self.repr_nn = DenseNN(self.dim_cov, [self.dim_hid], param_dims=[self.dim_hid]).float()
        self.cond_dist_nn = None

    def get_train_dataloader(self, cov_f, treat_f, out_f, batch_size, bounds=None) -> DataLoader:
        if bounds is None:
            training_data = TensorDataset(cov_f, treat_f, out_f)
        else:
            training_data = TensorDataset(cov_f, treat_f, out_f, bounds[0], bounds[1])
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True,
                                      generator=torch.Generator(device=self.device))
        return train_dataloader

    def prepare_train_data(self, train_data_dict: dict) -> Tuple[torch.Tensor]:
        """
        Data pre-processing
        :param train_data_dict: Dictionary with the training data
        """
        # Scaling train data
        cov_f = self.cov_scaler.fit_transform(train_data_dict['cov_f'].reshape(-1, self.dim_cov))

        cov_f, treat_f, out_f_scaled = self.prepare_tensors(cov_f, train_data_dict['treat_f'], train_data_dict['out_f_scaled'], kind='torch')
        self.hparams.dataset.n_samples_train = cov_f.shape[0]
        self.num_train_iter = int(self.hparams.dataset.n_samples_train / self.nuisance_batch_size * self.num_epochs)
        if self.has_prop_score:
            self.num_burn_in_train_iter = int(self.hparams.dataset.n_samples_train / self.nuisance_batch_size * self.num_burn_in_epochs)

        logger.info(f'Effective number of training iterations: {self.num_train_iter}.')
        return cov_f, treat_f, out_f_scaled

    def prepare_eval_data(self, data_dict: dict) -> Tuple[torch.Tensor]:
        # Scaling eval data
        cov_f = self.cov_scaler.transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        cov_f, treat_f, out_f_scaled = self.prepare_tensors(cov_f, data_dict['treat_f'], data_dict['out_f_scaled'], kind='torch')
        return cov_f, treat_f, out_f_scaled

    def get_nuisance_optimizer(self) -> torch.optim.Optimizer:
        """
        Init optimizer for the nuisance flow
        """
        # modules = torch.nn.ModuleList([self.repr_nn, self.cond_dist_nn])
        # return torch.optim.SGD(list(modules.parameters()), lr=self.nuisance_lr, momentum=0.9)
        raise NotImplementedError()

    def fit(self, train_data_dict: dict, log: bool):
        """
        Fitting the estimator
        @param train_data_dict: Training data dictionary
        @param log: Logging to the MlFlow
        """
        # Preparing data
        cov_f, treat_f, out_f_scaled = self.prepare_train_data(train_data_dict)
        train_dataloader = self.get_train_dataloader(cov_f, treat_f, out_f_scaled, batch_size=self.nuisance_batch_size)

        # Preparing optimizers
        nuisance_optimizer = self.get_nuisance_optimizer()
        self.to(self.device)

        # Logging
        self.mlflow_logger.log_hyperparams(self.hparams) if log else None

        # Saving train train_data_dict
        self.save_train_data_to_buffer(cov_f, treat_f, out_f_scaled)

        # Conditional NFs fitting
        logger.info('Fitting nuisance models')
        for step in tqdm(range(self.num_train_iter)) if log else range(self.num_train_iter):
            cov_f, treat_f, out_f_scaled = next(iter(train_dataloader))
            # cov_f, treat_f, out_f_scaled = cov_f.to(self.device), treat_f.to(self.device), out_f_scaled.to(self.device)

            nuisance_optimizer.zero_grad()

            # Representation -> Adding noise + concat of factual treatment -> Conditional distribution
            if not self.has_prop_score:
                repr_f = self.repr_nn(cov_f)
            else:
                repr_f, prop_preds = self.repr_nn(cov_f)
            noised_out_f_scaled = out_f_scaled + self.noise_std_Y * torch.randn_like(out_f_scaled)
            noised_repr_f = repr_f + self.noise_std_X * torch.randn_like(repr_f)
            context = torch.cat([noised_repr_f, treat_f], dim=-1)
            log_prob = self._cond_log_prob(context, noised_out_f_scaled)

            if not self.has_prop_score:
                loss = -log_prob.mean()
            else:
                bce_loss = torch.binary_cross_entropy_with_logits(prop_preds, treat_f)
                if step > self.num_burn_in_train_iter:
                    prop = torch.sigmoid(prop_preds.detach())
                    ipwt = ((treat_f == 1.0) & (prop >= self.clip_prop)).float() / (prop + 1e-9) + \
                           ((treat_f == 0.0) & ((1 - prop) >= self.clip_prop)).float() / (1 - prop + 1e-9)
                    ipwt_normalized = ipwt / ipwt.mean()
                    loss = (ipwt_normalized * (- log_prob)).mean() + self.prop_alpha * bce_loss.mean()
                else:
                    loss = (- log_prob).mean() + self.prop_alpha * bce_loss.mean()

            loss.backward()

            nuisance_optimizer.step()
            self._post_nuisance_optimizer_step()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics({'train_cond_neg_log_prob': (-log_prob).mean().item()}, step=step)
                if self.has_prop_score:
                    self.mlflow_logger.log_metrics({'train_bce': bce_loss.mean().item()}, step=step)
                    self.mlflow_logger.log_metrics({'train_loss': loss.item()}, step=step)

    def _post_nuisance_optimizer_step(self):
        pass

    def _cond_log_prob(self, context, out) -> torch.Tensor:
        """
        Internal method for the conditional log-probability
        @param context: Tensor of the context
        @param out: Outcome tensor
        @return: Tensor with conditional log-probabilities
        """
        raise NotImplementedError()

    def _cond_dist(self, context) -> torch.distributions.Distribution:
        """
        Internal method for the conditional distribution
        @param context: Tensor of the context
        @return: torch.distributions.Distribution
        """
        raise NotImplementedError()

    def cond_log_prob(self, treat_f, out_f_scaled, cov_f) -> torch.Tensor:
        """
        Conditional log-probability
        @param treat_f: Tensor with factual treatments
        @param out_f_scaled: Tensor with factual outcomes
        @param cov_f: Tensor with factual covariates
        @return: Tensor with log-probabilities
        """
        if not self.has_prop_score:
            repr_f = self.repr_nn(cov_f)
        else:
            repr_f, _ = self.repr_nn(cov_f)
        context = torch.cat([repr_f, treat_f], dim=-1)
        log_prob = self._cond_log_prob(context, out_f_scaled)
        return log_prob

    def cond_dist(self, treat_f, cov_f) -> torch.distributions.Distribution:
        if not self.has_prop_score:
            repr_f = self.repr_nn(cov_f)
        else:
            repr_f, _ = self.repr_nn(cov_f)
        context = torch.cat([repr_f, treat_f], dim=-1)
        cond_dist = self._cond_dist(context)
        return cond_dist

    def evaluate(self, data_dict: dict, log: bool, prefix: str) -> dict:
        cov_f, treat_f, out_f_scaled = self.prepare_eval_data(data_dict)

        self.eval()
        with torch.no_grad():
            results = {
                f'{prefix}_log_prob_f': self.cond_log_prob(treat_f, out_f_scaled, cov_f).mean().item()
            }

        if log:
            self.mlflow_logger.log_metrics(results, step=self.num_train_iter)
        return results

    def get_plugin_bounds(self, cov_f, delta_scaled_or_alpha, mode='cdf', return_pseudo=False, n_grid=200) -> Tuple[torch.Tensor]:
        if mode == 'cdf':
            out_grid = torch.linspace(-self.scaled_out_f_bound / 2, self.scaled_out_f_bound / 2, n_grid).reshape(1, -1, 1, 1)
            out_grid = out_grid.repeat(delta_scaled_or_alpha.shape[0], 1, cov_f.shape[0], 1)
        elif mode == 'icdf':
            quant_grid = torch.linspace(0.0, 1.0, n_grid + 2)[1:-1].reshape(1, -1, 1, 1)
            quant_grid = quant_grid.repeat(delta_scaled_or_alpha.shape[0], 1, cov_f.shape[0], 1)
        else:
            raise NotImplementedError()

        if mode == 'icdf' and return_pseudo:
            cond_dist0_no_grid = self.cond_dist(torch.zeros((delta_scaled_or_alpha.shape[0], cov_f.shape[0], 1)).float(), cov_f.repeat(delta_scaled_or_alpha.shape[0], 1, 1))
            cond_dist1_no_grid = self.cond_dist(torch.ones((delta_scaled_or_alpha.shape[0], cov_f.shape[0], 1)).float(), cov_f.repeat(delta_scaled_or_alpha.shape[0], 1, 1))

        delta_scaled_or_alpha = delta_scaled_or_alpha.reshape(-1, 1, 1, 1)
        delta_scaled_or_alpha = delta_scaled_or_alpha.repeat(1, n_grid, cov_f.shape[0], 1)
        cov_f = cov_f.repeat(delta_scaled_or_alpha.shape[0], n_grid, 1, 1)

        cond_dist0 = self.cond_dist(torch.zeros((delta_scaled_or_alpha.shape[0], n_grid, cov_f.shape[2], 1)).float(), cov_f)
        cond_dist1 = self.cond_dist(torch.ones((delta_scaled_or_alpha.shape[0], n_grid, cov_f.shape[2], 1)).float(), cov_f)

        if mode == 'cdf':
            if not return_pseudo:
                lb = torch.maximum((cond_dist1.cdf(out_grid) - cond_dist0.cdf(out_grid - delta_scaled_or_alpha)).max(1)[0],
                                   torch.zeros(1)).squeeze()
                ub = 1.0 + torch.minimum((cond_dist1.cdf(out_grid) - cond_dist0.cdf(out_grid - delta_scaled_or_alpha)).min(1)[0],
                                         torch.zeros(1)).squeeze()
            else:
                lb, lb_ix = (cond_dist1.cdf(out_grid) - cond_dist0.cdf(out_grid - delta_scaled_or_alpha)).max(1, keepdim=True)
                lb_arg = torch.where(lb <= 0.0, torch.nan, torch.gather(out_grid, 1, lb_ix)).squeeze()
                lb_cdf1 = torch.where(lb <= 0.0, torch.nan, torch.gather(cond_dist1.cdf(out_grid), 1, lb_ix)).squeeze()
                lb_cdf0_min_delta = torch.where(lb <= 0.0, torch.nan, torch.gather(cond_dist0.cdf(out_grid - delta_scaled_or_alpha), 1, lb_ix)).squeeze()
                lb = torch.maximum(lb, torch.zeros(1)).squeeze()
                # assert ((lb == 0.0) == lb_arg.isnan()).all()

                ub, ub_ix = (cond_dist1.cdf(out_grid) - cond_dist0.cdf(out_grid - delta_scaled_or_alpha)).min(1, keepdim=True)
                ub_arg = torch.where(ub >= 0.0, torch.nan, torch.gather(out_grid, 1, ub_ix)).squeeze()
                ub_cdf1 = torch.where(ub >= 0.0, torch.nan, torch.gather(cond_dist1.cdf(out_grid), 1, ub_ix)).squeeze()
                ub_cdf0_min_delta = torch.where(ub >= 0.0, torch.nan, torch.gather(cond_dist0.cdf(out_grid - delta_scaled_or_alpha), 1, ub_ix)).squeeze()
                ub = 1.0 + torch.minimum(ub, torch.zeros(1)).squeeze()
                # assert ((ub == 1.0) == ub_arg.isnan()).all()

        elif mode == 'icdf':
            if not return_pseudo:
                lb = (cond_dist1.icdf(quant_grid) - cond_dist0.icdf(quant_grid - delta_scaled_or_alpha)).nan_to_num(1e6).min(1)[0].squeeze()
                ub = (cond_dist1.icdf(quant_grid) - cond_dist0.icdf(1 + quant_grid - delta_scaled_or_alpha)).nan_to_num(-1e6).max(1)[0].squeeze()
            else:
                lb, lb_ix = (cond_dist1.icdf(quant_grid) - cond_dist0.icdf(quant_grid - delta_scaled_or_alpha)).nan_to_num(1e6).min(1, keepdim=True)
                lb_arg = torch.gather(quant_grid, 1, lb_ix).squeeze()
                lb_icdf1 = torch.gather(cond_dist1.icdf(quant_grid), 1, lb_ix).squeeze()
                lb_icdf0_min_alpha = torch.gather(cond_dist0.icdf(quant_grid - delta_scaled_or_alpha), 1, lb_ix).squeeze()
                lb = lb.squeeze()
                lb_prob1 = cond_dist1_no_grid.log_prob(lb_icdf1.unsqueeze(-1)).exp().squeeze()
                lb_prob0 = cond_dist0_no_grid.log_prob(lb_icdf0_min_alpha.unsqueeze(-1)).exp().squeeze()

                ub, ub_ix = (cond_dist1.icdf(quant_grid) - cond_dist0.icdf(1 + quant_grid - delta_scaled_or_alpha)).nan_to_num(-1e6).max(1, keepdim=True)
                ub_arg = torch.gather(quant_grid, 1, ub_ix).squeeze()
                ub_icdf1 = torch.gather(cond_dist1.icdf(quant_grid), 1, ub_ix).squeeze()
                ub_icdf0_min_alpha = torch.gather(cond_dist0.icdf(1 + quant_grid - delta_scaled_or_alpha), 1, ub_ix).squeeze()
                ub = ub.squeeze()
                ub_prob1 = cond_dist1_no_grid.log_prob(ub_icdf1.unsqueeze(-1)).exp().squeeze()
                ub_prob0 = cond_dist0_no_grid.log_prob(ub_icdf0_min_alpha.unsqueeze(-1)).exp().squeeze()

        else:
            raise NotImplementedError()

        if not return_pseudo:
            return {
                'lb': lb,
                'ub': ub
            }
        else:
            if mode == 'cdf':
                return {
                    'lb': lb,
                    'lb_arg': lb_arg,
                    'lb_cdf1': lb_cdf1,
                    'lb_cdf0_min_delta': lb_cdf0_min_delta,
                    'ub': ub,
                    'ub_arg': ub_arg,
                    'ub_cdf1': ub_cdf1,
                    'ub_cdf0_min_delta': ub_cdf0_min_delta
                }
            elif mode == 'icdf':
                return {
                    'lb': lb,
                    'lb_arg': lb_arg,
                    'lb_icdf1': lb_icdf1,
                    'lb_icdf0_min_alpha': lb_icdf0_min_alpha,
                    'lb_prob1': lb_prob1,
                    'lb_prob0': lb_prob0,
                    'ub': ub,
                    'ub_arg': ub_arg,
                    'ub_icdf1': ub_icdf1,
                    'ub_icdf0_min_alpha': ub_icdf0_min_alpha,
                    'ub_prob1': ub_prob1,
                    'ub_prob0': ub_prob0
                }
            else:
                raise NotImplementedError()

    def get_bounds(self, cov_f, delta_scaled_or_alpha, mode='cdf', return_pseudo=False) -> Tuple[torch.Tensor]:
        delta_scaled_or_alpha = torch.tensor(delta_scaled_or_alpha).float()
        dict_keys = ['ub', 'lb']
        if return_pseudo:
            if mode == 'cdf':
                dict_keys += ['lb_arg', 'lb_cdf1', 'lb_cdf0_min_delta', 'ub_arg', 'ub_cdf1', 'ub_cdf0_min_delta']
            elif mode == 'icdf':
                dict_keys += ['lb_arg', 'lb_icdf1', 'lb_icdf0_min_alpha', 'lb_prob1', 'lb_prob0', 'ub', 'ub_arg', 'ub_icdf1',
                              'ub_icdf0_min_alpha', 'ub_prob1', 'ub_prob0']
            else:
                raise NotImplementedError()

        bounds_dict = {}
        if cov_f.shape[0] > 500:
            batch_size = 500
            for dict_key in dict_keys:
                bounds_dict[dict_key] = torch.zeros((self.hparams.model.n_delta, cov_f.shape[0]))

            for i in range(cov_f.shape[0] // batch_size + 1):
                bounds_dict_batch = self.get_plugin_bounds(cov_f[i * batch_size: (i + 1) * batch_size], delta_scaled_or_alpha,
                                                           mode, return_pseudo)
                for dict_key in dict_keys:
                    bounds_dict[dict_key][:, i * batch_size: (i + 1) * batch_size] = bounds_dict_batch[dict_key]
        else:
            bounds_dict = self.get_plugin_bounds(cov_f, delta_scaled_or_alpha, mode, return_pseudo)

        if return_pseudo:
            return bounds_dict
        else:
            return bounds_dict['lb'], bounds_dict['ub']


class PluginCNFs(PluginNeuralConditionalDensityEstimator):
    """
    CNF
    """

    def __init__(self, args: DictConfig = None, **kwargs):
        super(PluginCNFs, self).__init__(args)

        # Model hyparams & Train params
        self.nuisance_count_bins = args.model.nuisance_count_bins

        # Model parameters = Conditional NFs (marginalized nuisance)
        self.cond_base_dist = dist.Normal(torch.zeros(self.dim_out).float(), torch.ones(self.dim_out).float())

        self.cond_loc = torch.nn.Parameter(torch.zeros((self.dim_out, )).float())
        self.cond_scale = torch.nn.Parameter(torch.ones((self.dim_out, )).float())
        self.cond_affine_transform = T.AffineTransform(self.cond_loc, self.cond_scale)

        self.dim_hid = self.nuisance_hid_dim_multiplier

        self.cond_dist_nn = DenseNN(self.dim_hid + self.dim_treat, [self.dim_hid],
                                    param_dims=[self.nuisance_count_bins,
                                                  self.nuisance_count_bins,
                                                  (self.nuisance_count_bins - 1)]).float()
        self.cond_spline_transform = T.ConditionalSpline(self.cond_dist_nn, self.dim_out,
                                                         order='quadratic',
                                                         count_bins=self.nuisance_count_bins,
                                                         bound=self.scaled_out_f_bound).to(self.device)

        self.cond_flow_dist = dist.ConditionalTransformedDistribution(self.cond_base_dist,
                                                                      [self.cond_affine_transform, self.cond_spline_transform])

    def get_nuisance_optimizer(self) -> torch.optim.Optimizer:
        """
        Init optimizer for the nuisance flow
        """
        modules = torch.nn.ModuleList([self.repr_nn, self.cond_dist_nn])
        return torch.optim.SGD(list(modules.parameters()) + [self.cond_loc, self.cond_scale], lr=self.nuisance_lr, momentum=0.9)

    def _post_nuisance_optimizer_step(self):
        self.cond_flow_dist.clear_cache()

    def _cond_log_prob(self, context, out) -> torch.Tensor:
        return self.cond_flow_dist.condition(context).log_prob(out)

    def _cond_dist(self, context) -> torch.distributions.Distribution:
        return self.cond_flow_dist.condition(context)


class PluginDKME(BoundsEstimator):
    """
    Distributional kernel mean embeddings
    """

    val_metric = 'val_kernel_ridge_reg_neg_mse'

    def __init__(self, args: DictConfig = None, **kwargs):
        super(PluginDKME, self).__init__(args)

        self.cov_scaler = StandardScaler()
        self.scaled_out_f_bound = args.model.scaled_out_f_bound  # Support bounds for the scaled outcome

        # Model hyparams
        self.sd_x = args.model.sd_x
        self.eps = args.model.eps
        self.num_train_iter = 0

        # Model parameters
        self.normalized_rbf_y = []  # Will be initialized during the fit
        self.rbf_x = RBF(np.sqrt(self.sd_x / 2))
        self.K_inv = []

    def prepare_train_data(self, train_data_dict: dict) -> Tuple[torch.Tensor]:
        """
        Data pre-processing
        :param train_data_dict: Dictionary with the training data
        """
        # Scaling train data
        cov_f = self.cov_scaler.fit_transform(train_data_dict['cov_f'].reshape(-1, self.dim_cov))

        cov_f, treat_f, out_f_scaled = self.prepare_tensors(cov_f, train_data_dict['treat_f'], train_data_dict['out_f_scaled'], kind='numpy')
        self.hparams.dataset.n_samples_train = cov_f.shape[0]
        return cov_f, treat_f, out_f_scaled

    def prepare_eval_data(self, data_dict: dict) -> Tuple[torch.Tensor]:
        # Scaling eval data
        cov_f = self.cov_scaler.transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        cov_f, treat_f, out_f_scaled = self.prepare_tensors(cov_f, data_dict['treat_f'], data_dict['out_f_scaled'], kind='numpy')
        return cov_f, treat_f, out_f_scaled

    def set_sd_y_median_heuristic(self) -> None:
        """
        Calculate median heuristics (for DKME)
        """
        for treat_option in [0.0, 1.0]:
            distances = np.tril(squareform(pdist(self.out_f_scaled[self.treat_f.reshape(-1) == treat_option].reshape(-1, self.dim_out), 'sqeuclidean')), -1)
            sd_y = np.median(distances[distances > 0.0])
            self.normalized_rbf_y.append(NormalizedRBF(sd_y / 3))
            logger.info(f'New sd_y[{treat_option}]: {sd_y / 3}')

    def fit(self, train_data_dict: dict, log: bool):
        """
        Fitting the estimator
        @param train_data_dict: Training data dictionary
        @param log: Logging to the MlFlow
        """
        # Preparing data
        cov_f, treat_f, out_f_scaled = self.prepare_train_data(train_data_dict)

        # Logging
        self.mlflow_logger.log_hyperparams(self.hparams) if log else None

        # Saving train train_data_dict
        self.save_train_data_to_buffer(cov_f, treat_f, out_f_scaled)

        # Conditional NFs fitting
        logger.info('Fitting nuisance models')

        # Median heuristic for sd_y
        self.set_sd_y_median_heuristic()

        # Skipping fitting while hparam tuning
        if not log:
            return

        for treat_option in [0.0, 1.0]:
            K = self.rbf_x(cov_f[treat_f == treat_option], cov_f[treat_f == treat_option])
            n_cond = cov_f[treat_f == treat_option].shape[0]
            K_inv = np.linalg.inv(K + n_cond * self.eps * np.eye(n_cond))
            self.K_inv.append(K_inv)

    def kernel_ridge_reg_neg_mse(self, treat_f, out_f_scaled, cov_f) -> np.array:
        """
        Negative MSE of the kernel ridge regression with the same hyperparameters as DKME
        @param treat_f: Tensor with factual treatments
        @param out_f_scaled: Tensor with factual outcomes
        @param cov_f: Tensor with factual covariates
        @return: Negative MSE
        """
        mses = np.zeros_like(out_f_scaled)
        for treat_option in [0.0, 1.0]:
            ker_ridge_reg = GaussianProcessRegressor(alpha=self.eps, kernel=self.rbf_x, optimizer=None)
            ker_ridge_reg.fit(self.cov_f[self.treat_f == treat_option], self.out_f_scaled[self.treat_f == treat_option])
            out_pred = ker_ridge_reg.predict(cov_f[treat_f == treat_option]).reshape(-1, self.dim_out)
            mses[treat_f == treat_option] = ((out_pred - out_f_scaled[treat_f == treat_option]) ** 2)
        return - mses

    def evaluate(self, data_dict: dict, log: bool, prefix: str) -> dict:
        cov_f, treat_f, out_f_scaled = self.prepare_eval_data(data_dict)

        results = {
            f'{prefix}_log_prob_f': self.cond_log_prob(treat_f, out_f_scaled, cov_f).mean(),
            f'{prefix}_kernel_ridge_reg_neg_mse': self.kernel_ridge_reg_neg_mse(treat_f, out_f_scaled, cov_f).mean()
        }

        if log:
            self.mlflow_logger.log_metrics(results, step=self.num_train_iter)
        return results

    def cond_log_prob(self, treat_f, out_f_scaled, cov_f) -> np.array:
        """
        Conditional log-probability
        @param treat_f: Tensor with factual treatments
        @param out_f_scaled: Tensor with factual outcomes
        @param cov_f: Tensor with factual covariates
        @return: Tensor with log-probabilities
        """
        log_prob = np.zeros_like(out_f_scaled)
        for treat_option, normalized_rbf_y, K_inv in zip([0.0, 1.0], self.normalized_rbf_y, self.K_inv):
            if (treat_f == treat_option).sum() > 0:
                L = normalized_rbf_y(out_f_scaled[treat_f == treat_option], self.out_f_scaled[self.treat_f == treat_option])
                K_x = self.rbf_x(self.cov_f[self.treat_f == treat_option], cov_f[treat_f == treat_option])
                w = np.dot(K_inv, K_x)
                w = w / w.sum(0, keepdims=True)
                log_prob[treat_f == treat_option, :] = np.log((L * w.T).sum(1, keepdims=True))

        log_prob[np.isnan(log_prob)] = -1e10
        return log_prob

    def cond_cdf(self, treat_f, out_f_scaled, cov_f):
        cdfs = np.zeros_like(out_f_scaled)
        for treat_option, normalized_rbf_y, K_inv in zip([0.0, 1.0], self.normalized_rbf_y, self.K_inv):
            if (treat_f == treat_option).sum() > 0:
                diff_mat = np.subtract.outer(out_f_scaled[treat_f == treat_option], self.out_f_scaled[self.treat_f == treat_option])
                K_x = self.rbf_x(self.cov_f[self.treat_f == treat_option], cov_f[treat_f == treat_option])
                w = np.dot(K_inv, K_x)
                w = w / w.sum(0, keepdims=True)
                cdfs[treat_f == treat_option, :] = (ndtr(np.squeeze(diff_mat) / normalized_rbf_y.sd) * w.T).sum(1, keepdims=True)
        cdfs = np.clip(cdfs, 0.0, 1.0)
        return cdfs

    def get_bounds(self, cov_f, delta_scaled_or_alpha, mode='cdf', n_grid=200) -> Tuple[torch.Tensor]:
        # This ideally should be vectorized in the future
        if mode == 'cdf':
            out_grid = np.linspace(-self.scaled_out_f_bound / 2, self.scaled_out_f_bound / 2, n_grid)
        elif mode == 'icdf':
            logger.warning('ICDF bounds are not implemented for DKME plugin')
            return np.array(np.nan), np.array(np.nan)
        else:
            raise NotImplementedError()

        cond_cdfs1_out = np.zeros((delta_scaled_or_alpha.shape[0], n_grid, cov_f.shape[0], 1))
        cond_cdfs0_out_min_delta = np.zeros((delta_scaled_or_alpha.shape[0], n_grid, cov_f.shape[0], 1))

        for g in tqdm(range(n_grid)):
            cond_cdfs1_out_temp = self.cond_cdf(np.ones((cov_f.shape[0],)).astype(float), out_grid[g].repeat(cov_f.shape[0]).reshape(-1, 1), cov_f)
            for d in range(delta_scaled_or_alpha.shape[0]):
                cond_cdfs1_out[d, g] = cond_cdfs1_out_temp
                cond_cdfs0_out_min_delta[d, g] = self.cond_cdf(np.zeros((cov_f.shape[0],)).astype(float), (out_grid[g] - delta_scaled_or_alpha[d]).repeat(cov_f.shape[0]).reshape(-1, 1), cov_f)

        lb = np.squeeze(np.maximum((cond_cdfs1_out - cond_cdfs0_out_min_delta).max(1), np.zeros(1)))
        ub = np.squeeze(1.0 + np.minimum((cond_cdfs1_out - cond_cdfs0_out_min_delta).min(1), np.zeros(1)))
        return torch.tensor(lb), torch.tensor(ub)


class IPTWPluginCNFs(PluginCNFs):
    """
    Plugin CNF with IPW weights (=two step IPW-learner)
    """

    val_metric = 'val_log_prob_f_minus_bce'

    def __init__(self, args: DictConfig = None, **kwargs):
        super(IPTWPluginCNFs, self).__init__(args)

        self.has_prop_score = True
        self.prop_alpha = args.model.prop_alpha
        self.clip_prop = args.model.clip_prop
        self.num_burn_in_epochs = args.model.num_burn_in_epochs

        # Model parameters = Conditional NFs (marginalized nuisance)
        self.repr_nn = DenseNN(self.dim_cov, [self.dim_hid], param_dims=[self.dim_hid, self.dim_treat]).float()

    def bce(self, treat_f, cov_f):
        _, prop_preds = self.repr_nn(cov_f)
        bce = torch.binary_cross_entropy_with_logits(prop_preds, treat_f)
        return bce

    def get_propensity(self, cov_f) -> [torch.Tensor, torch.Tensor]:
        _, prop_preds = self.repr_nn(cov_f)
        prop1 = torch.sigmoid(prop_preds)
        prop0 = 1.0 - prop1
        return prop0, prop1

    def evaluate(self, data_dict: dict, log: bool, prefix: str) -> dict:
        cov_f, treat_f, out_f_scaled = self.prepare_eval_data(data_dict)

        self.eval()
        with torch.no_grad():
            log_prob = self.cond_log_prob(treat_f, out_f_scaled, cov_f).mean().item()
            bce = self.bce(treat_f, cov_f).mean().item()

            results = {
                f'{prefix}_log_prob_f': log_prob,
                f'{prefix}_bce': bce,
                f'{prefix}_log_prob_f_minus_bce': log_prob - self.prop_alpha * bce
            }

        if log:
            self.mlflow_logger.log_metrics(results, step=self.num_train_iter)
        return results