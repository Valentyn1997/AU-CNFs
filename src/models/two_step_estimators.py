import torch
import pyro.distributions as dist
import pyro.distributions.transforms as T
from torch.distributions import TransformedDistribution
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import DictConfig
import logging
from pyro.nn import DenseNN
from tqdm import tqdm
from typing import Tuple, List
from torch_ema import ExponentialMovingAverage
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
import numpy as np


from src.models.plugins import PluginCNFs, IPTWPluginCNFs


logger = logging.getLogger(__name__)


class CovariateAdjustedCNFs(PluginCNFs):

    def __init__(self, args: DictConfig = None, **kwargs):
        super(CovariateAdjustedCNFs, self).__init__(args)

        self.target_count_bins = args.model.target_count_bins if args.model.target_count_bins is not None else 2 * args.model.nuisance_count_bins
        self.target_noise_std_X = args.model.target_noise_std_X if args.model.target_noise_std_X is not None else args.model.noise_std_X
        self.scaled_target_bound = args.model.scaled_target_bound

        self.target_mode = args.model.target_mode
        self.target_lr = args.model.target_lr
        self.target_grid_size = args.model.n_delta

        self.target_batch_size = args.model.target_batch_size
        self.target_num_epochs = args.model.target_num_epochs
        self.target_gamma = args.model.target_gamma
        self.correction_coeff = None

        # Model parameters = NF (target)
        self.target_base_dists = [dist.Normal(torch.zeros(self.dim_out).float(), torch.ones(self.dim_out).float()) for _ in range(2)]

        self.target_loc = [torch.nn.Parameter(torch.zeros(self.dim_out).float()) for _ in range(2)]
        self.target_scale = [torch.nn.Parameter(torch.ones(self.dim_out).float()) for _ in range(2)]
        self.target_affine_transforms = [T.AffineTransform(target_loc, target_scale) for (target_loc, target_scale) in zip(self.target_loc, self.target_scale)]

        self.target_dist_nns = [DenseNN(self.dim_cov, args.model.target_hid_layers * [self.dim_hid],
                                       param_dims=[self.target_count_bins,
                                                  self.target_count_bins,
                                                  (self.target_count_bins - 1)]).float() for _ in range(2)]
        self.target_spline_transforms = [T.ConditionalSpline(target_dist_nn, self.dim_out, order='quadratic',
                                                             count_bins=self.target_count_bins, bound=self.scaled_target_bound).to(self.device)
                                         for target_dist_nn in self.target_dist_nns]

        self.target_flow_dists = [dist.ConditionalTransformedDistribution(target_base_dist,
                                                                          [target_affine_transform, target_spline_transform])
                                  for (target_base_dist, target_affine_transform, target_spline_transform) in
                                  zip(self.target_base_dists, self.target_affine_transforms, self.target_spline_transforms)]

        self.ema_target = None

    def get_target_optimizers(self) -> List[torch.optim.Optimizer]:
        """
        Init optimizer for the target flow
        """
        modules = [torch.nn.ModuleList([target_dist_nn]) for target_dist_nn in self.target_dist_nns]
        parameters = [list(module.parameters()) + [loc, scale] for (module, loc, scale) in zip(modules, self.target_loc, self.target_scale)]
        optimizers = [torch.optim.SGD(par, lr=self.target_lr, momentum=0.9) for par in parameters]
        ema_target = ExponentialMovingAverage([p for par in parameters for p in par], decay=self.target_gamma)
        return optimizers, ema_target

    def prepare_train_data(self, train_data_dict: dict) -> Tuple[torch.Tensor]:
        """
        Data pre-processing
        :param train_data_dict: Dictionary with the training data
        """
        cov_f, treat_f, out_f_scaled = super(CovariateAdjustedCNFs, self).prepare_train_data(train_data_dict)

        self.num_target_train_iter = int(self.hparams.dataset.n_samples_train / self.target_batch_size * self.target_num_epochs)

        return cov_f, treat_f, out_f_scaled

    def get_target_bounds(self, cov_f, delta_scaled_or_alpha, mode='cdf') -> Tuple[torch.Tensor]:
        delta_scaled_or_alpha = delta_scaled_or_alpha.reshape(-1, 1, 1).repeat(1, cov_f.shape[0], 1)
        cov_f = cov_f.repeat(delta_scaled_or_alpha.shape[0], 1, 1)
        with self.ema_target.average_parameters():
            if mode == 'cdf':
                lb = self.target_flow_dists[0].condition(cov_f).cdf(delta_scaled_or_alpha)
                ub = self.target_flow_dists[1].condition(cov_f).cdf(delta_scaled_or_alpha)
            elif mode == 'icdf':
                lb = self.target_flow_dists[0].condition(cov_f).icdf(delta_scaled_or_alpha)
                ub = self.target_flow_dists[1].condition(cov_f).icdf(delta_scaled_or_alpha)
            else:
                raise NotImplementedError()
        return lb.squeeze(), ub.squeeze()

    def get_bounds(self, cov_f, delta_scaled_or_alpha, mode='cdf') -> Tuple[torch.Tensor]:
        delta_scaled_or_alpha = torch.tensor(delta_scaled_or_alpha).float()
        return self.get_target_bounds(cov_f, delta_scaled_or_alpha, mode)
    
    def get_pseudo_bounds(self, delta_scaled_or_alpha):
        lb_plugin, ub_plugin = super(CovariateAdjustedCNFs, self).get_bounds(self.cov_f, delta_scaled_or_alpha, mode=self.target_mode)
        return lb_plugin, ub_plugin
        
    def fit(self, train_data_dict: dict, log: bool):
        # Preparing optimizers
        target_optimizers, self.ema_target = self.get_target_optimizers()

        super(CovariateAdjustedCNFs, self).fit(train_data_dict, log)

        logger.info('Fitting target models models')

        if self.target_mode == 'cdf':  # Delta
            delta_scaled_or_alpha = torch.linspace(self.hparams.dataset.delta_scaled_min, self.hparams.dataset.delta_scaled_max,
                                                   self.hparams.model.n_delta)
        elif self.target_mode == 'icdf':  # Alpha
            delta_scaled_or_alpha = torch.linspace(0.0, 1.0, self.hparams.model.n_delta + 2)[1:-1]
        else:
            raise NotImplementedError()
        with torch.no_grad():
            lb_pseudo, ub_pseudo = self.get_pseudo_bounds(delta_scaled_or_alpha)
            lb_pseudo, ub_pseudo = lb_pseudo.T, ub_pseudo.T  # Batch-wise splitting
            
        train_dataloader = self.get_train_dataloader(self.cov_f, self.treat_f, self.out_f_scaled,
                                                     batch_size=self.target_batch_size, bounds=(lb_pseudo, ub_pseudo))

        delta_scaled_or_alpha = delta_scaled_or_alpha.reshape(-1, 1, 1).repeat(1, self.target_batch_size, 1)

        for step in tqdm(range(self.num_target_train_iter)) if log else range(self.num_target_train_iter):

            cov_f, treat_f, out_f, lb_pseudo, ub_pseudo = next(iter(train_dataloader))
            lb_pseudo, ub_pseudo = lb_pseudo.T, ub_pseudo.T

            noised_cov_f = cov_f + self.target_noise_std_X * torch.randn_like(cov_f)
            noised_cov_f = noised_cov_f.repeat(self.hparams.model.n_delta, 1, 1)

            for i, (b, b_pseudo) in enumerate(zip(['l', 'u'], [lb_pseudo, ub_pseudo])):
                target_optimizers[i].zero_grad()

                if self.target_mode == 'cdf':
                    b_target = self.target_flow_dists[i].condition(noised_cov_f).cdf(delta_scaled_or_alpha).squeeze()
                elif self.target_mode == 'icdf':
                    b_target = self.target_flow_dists[i].condition(noised_cov_f).icdf(delta_scaled_or_alpha).squeeze()

                loss = (delta_scaled_or_alpha[1, 0, 0] - delta_scaled_or_alpha[0, 0, 0]) * ((b_pseudo - b_target) ** 2).sum(0).mean()

                loss.backward()

                target_optimizers[i].step()
                self.target_flow_dists[i].clear_cache()

                if step % 50 == 0 and log:
                    self.mlflow_logger.log_metrics({f'train_target_{b}_loss': loss.item()}, step=step)

            self.ema_target.update()


class DoublyRobustCNFs(CovariateAdjustedCNFs, IPTWPluginCNFs):
    def __init__(self, args: DictConfig = None, **kwargs):
        super(DoublyRobustCNFs, self).__init__(args)

        self.correction_coeff = args.model.correction_coeff
        self.proj_pseudo = args.model.proj_pseudo

    def _project_pseudo_bounds_on_monotone(self, bound, delta_scaled_or_alpha):

        logger.info('Projecting pseudo-CDFs on the class of monotone functions.')

        def get_lower(polygon):
            minx = np.argmin(polygon[:, 0])
            maxx = np.argmax(polygon[:, 0]) + 1
            if minx >= maxx:
                lower_curve = np.concatenate([polygon[minx:], polygon[:maxx]])
            else:
                lower_curve = polygon[minx:maxx]
            return lower_curve

        bound_proj = torch.zeros_like(bound)
        delta_scaled_or_alpha = delta_scaled_or_alpha.cpu()

        for i in tqdm(range(bound.shape[1])):
            bound_sp = UnivariateSpline(delta_scaled_or_alpha.squeeze(), bound[:, i].cpu(), s=0, k=1)
            bound_anti_der = bound_sp.antiderivative()(delta_scaled_or_alpha)
            antider_graph = np.concatenate([delta_scaled_or_alpha, bound_anti_der], axis=1)
            antider_graph = np.concatenate([np.array([[delta_scaled_or_alpha.squeeze()[0], bound_anti_der.squeeze().min()]]),
                                            antider_graph[1:]], axis=0)
            antider_hull = ConvexHull(antider_graph)
            lower_antider_hull = get_lower(antider_graph[antider_hull.vertices])
            lower_antider_hull_sp = UnivariateSpline(lower_antider_hull[:, 0], lower_antider_hull[:, 1], s=0, k=1)
            bound_proj[:, i] = torch.tensor(lower_antider_hull_sp.derivative()(delta_scaled_or_alpha.squeeze())).float()

        return bound_proj

    def get_pseudo_bounds(self, delta_scaled_or_alpha):
        prop0, prop1 = self.get_propensity(self.cov_f)

        A_by_pi = ((self.treat_f.T == 1.0) & (prop1.T >= self.clip_prop)).float() / (prop1.T + 1e-9)
        one_min_A_by_one_min_pi = ((self.treat_f.T == 0.0) & (prop0.T >= self.clip_prop)).float() / (prop0.T + 1e-9)

        bounds_dict = super(CovariateAdjustedCNFs, self).get_bounds(self.cov_f, delta_scaled_or_alpha, mode=self.target_mode,
                                                                    return_pseudo=True)
        delta_scaled_or_alpha = delta_scaled_or_alpha.unsqueeze(-1)
        if self.target_mode == 'cdf':
            lb_pseudo1 = A_by_pi * ((self.out_f_scaled.T <= bounds_dict['lb_arg']).float() - bounds_dict['lb_cdf1'])
            lb_pseudo0 = one_min_A_by_one_min_pi * ((self.out_f_scaled.T <= bounds_dict['lb_arg'] - delta_scaled_or_alpha).float()
                                                    - bounds_dict['lb_cdf0_min_delta'])
            lb_dr = self.correction_coeff * (lb_pseudo1 - lb_pseudo0).nan_to_num(0.0) + bounds_dict['lb']

            ub_pseudo1 = A_by_pi * ((self.out_f_scaled.T <= bounds_dict['ub_arg']).float() - bounds_dict['ub_cdf1'])
            ub_pseudo0 = one_min_A_by_one_min_pi * ((self.out_f_scaled.T <= bounds_dict['ub_arg'] - delta_scaled_or_alpha).float()
                                                    - bounds_dict['ub_cdf0_min_delta'])
            ub_dr = self.correction_coeff * (ub_pseudo1 - ub_pseudo0).nan_to_num(0.0) + bounds_dict['ub']

            if self.proj_pseudo:
                lb_dr, ub_dr = self._project_pseudo_bounds_on_monotone(lb_dr, delta_scaled_or_alpha), \
                               self._project_pseudo_bounds_on_monotone(ub_dr, delta_scaled_or_alpha)

            return lb_dr, ub_dr

        elif self.target_mode == 'icdf':
            lb_pseudo1 = A_by_pi * ((self.out_f_scaled.T <= bounds_dict['lb_icdf1']).float() - bounds_dict['lb_arg']) / \
                         (bounds_dict['lb_prob1'] + 1e-9)
            lb_pseudo0 = one_min_A_by_one_min_pi * ((self.out_f_scaled.T <= bounds_dict['lb_icdf0_min_alpha']).float() -
                                                    (bounds_dict['lb_arg'] - delta_scaled_or_alpha)) / \
                         (bounds_dict['lb_prob0'] + 1e-9)
            lb_dr = self.correction_coeff * (lb_pseudo1 - lb_pseudo0) + bounds_dict['lb']

            ub_pseudo1 = A_by_pi * ((self.out_f_scaled.T <= bounds_dict['ub_icdf1']).float() - bounds_dict['ub_arg']) / \
                         (bounds_dict['ub_prob1'] + 1e-9)
            ub_pseudo0 = one_min_A_by_one_min_pi * ((self.out_f_scaled.T <= bounds_dict['ub_icdf0_min_alpha']).float() -
                                                    (1 + bounds_dict['ub_arg'] - delta_scaled_or_alpha)) / \
                         (bounds_dict['ub_prob0'] + 1e-9)
            ub_dr = self.correction_coeff * (ub_pseudo1 - ub_pseudo0) + bounds_dict['ub']

            if self.proj_pseudo:
                lb_dr, ub_dr = self._project_pseudo_bounds_on_monotone(lb_dr, delta_scaled_or_alpha), \
                               self._project_pseudo_bounds_on_monotone(ub_dr, delta_scaled_or_alpha)

            return lb_dr, ub_dr
        else:
            raise NotImplementedError()


class AIPTWINFs(CovariateAdjustedCNFs, IPTWPluginCNFs):
    def __init__(self, args: DictConfig = None, **kwargs):
        super(AIPTWINFs, self).__init__(args)

        self.correction_coeff = 1.0

        self.treat_options = [0.0, 1.0]

        self.target_base_dists = [dist.Normal(torch.zeros(self.dim_out).float(), torch.ones(self.dim_out).float()) for _ in self.treat_options]
        self.target_loc = [torch.nn.Parameter(torch.zeros(self.dim_out).float()) for _ in self.treat_options]
        self.target_scale = [torch.nn.Parameter(torch.ones(self.dim_out).float()) for _ in self.treat_options]
        self.target_affine_transforms = [T.AffineTransform(loc, scale) for (loc, scale) in zip(self.target_loc, self.target_scale)]

        self.target_spline_transforms = [T.spline(self.dim_out, order='quadratic',
                                          count_bins=self.target_count_bins,
                                          bound=args.model.scaled_target_bound).float() for _ in self.treat_options]
        self.target_flow_dists = [TransformedDistribution(base_dist, [affine_transform, spline_transform])
                          for (base_dist, affine_transform, spline_transform) in
                          zip(self.target_base_dists, self.target_affine_transforms, self.target_spline_transforms)]

    def get_target_optimizers(self) -> List[torch.optim.Optimizer]:
        """
        Init optimizer for the target flow
        """
        modules = [torch.nn.ModuleList([spline_transform]) for spline_transform in self.target_spline_transforms]
        parameters = [list(module.parameters()) + [loc, scale] for (module, loc, scale) in zip(modules, self.target_loc, self.target_scale)]
        optimizers = [torch.optim.Adam(par, lr=self.target_lr) for par in parameters]
        ema_target = ExponentialMovingAverage([p for par in parameters for p in par], decay=self.target_gamma)
        return optimizers, ema_target

    def get_cond_log_prob_grid(self, out_pot_grid, cov_f):
        out_pot_grid = out_pot_grid.reshape(-1, 1, 1).repeat(1, cov_f.shape[0], 1)
        cov_f = cov_f.repeat(out_pot_grid.shape[0], 1, 1)
        treat_pot0 = torch.zeros((out_pot_grid.shape[0], cov_f.shape[1], 1)).float()
        treat_pot1 = torch.ones((out_pot_grid.shape[0], cov_f.shape[1], 1)).float()

        return self.cond_log_prob(treat_pot0, out_pot_grid, cov_f).squeeze(), \
               self.cond_log_prob(treat_pot1, out_pot_grid, cov_f).squeeze()

    def get_train_dataloader(self, cov_f, treat_f, out_f, batch_size, prop=None, cond_log_prob=None) -> DataLoader:
        if prop is None or cond_log_prob is None:
            training_data = TensorDataset(cov_f, treat_f, out_f)
        else:
            training_data = TensorDataset(cov_f, treat_f, out_f, prop[0], prop[1], cond_log_prob[0], cond_log_prob[1])
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=self.device))
        return train_dataloader


    def fit(self, train_data_dict: dict, log: bool):
        # Preparing optimizers
        target_optimizers, self.ema_target = self.get_target_optimizers()

        super(CovariateAdjustedCNFs, self).fit(train_data_dict, log)

        logger.info('Fitting target models models')

        out_pot_grid = torch.linspace(self.out_f_scaled.min(), self.out_f_scaled.max(), self.hparams.model.n_delta)

        with torch.no_grad():
            cond_log_prob0, cond_log_prob1 = self.get_cond_log_prob_grid(out_pot_grid, self.cov_f)
            prop0, prop1 = self.get_propensity(self.cov_f)

        train_dataloader = self.get_train_dataloader(self.cov_f, self.treat_f, self.out_f_scaled,
                                                     batch_size=self.target_batch_size, prop=(prop0, prop1),
                                                     cond_log_prob=(cond_log_prob0.T, cond_log_prob1.T))

        for step in tqdm(range(self.num_target_train_iter)) if log else range(self.num_target_train_iter):
            # NF (target) fitting
            cov_f, treat_f, out_f, prop0, prop1, cond_log_prob0, cond_log_prob1 = next(iter(train_dataloader))

            for i, (treat_option, prop, cond_log_prob) in enumerate(zip(self.treat_options, [prop0, prop1], [cond_log_prob0, cond_log_prob1])):
                target_optimizers[i].zero_grad()

                # Cross-entropy + bias correction
                log_prob_pot = self.target_flow_dists[i].log_prob(out_pot_grid.reshape(-1, 1)).squeeze()
                cross_entropy = - (log_prob_pot * cond_log_prob.exp().mean(0)).sum() * (out_pot_grid[1] - out_pot_grid[0])
                cond_cross_entropy = - (log_prob_pot.unsqueeze(0) * cond_log_prob.exp()).sum(1) * (out_pot_grid[1] - out_pot_grid[0])

                log_prob_f = self.target_flow_dists[i].log_prob(out_f).squeeze()
                A_by_pi = (((treat_f == treat_option) & (prop >= self.clip_prop)) / (prop + 1e-9)).squeeze()

                bias_correction = (A_by_pi * (- log_prob_f - cond_cross_entropy)).mean()

                loss = cross_entropy + bias_correction
                loss.backward()

                target_optimizers[i].step()
                self.target_flow_dists[i].clear_cache()

                if step % 50 == 0 and log:
                    self.mlflow_logger.log_metrics({
                        f'train_nce_{int(treat_option)}': cross_entropy.item(),
                        f'bias_correction_{int(treat_option)}': bias_correction.item()
                    }, step=step)

            self.ema_target.update()

    def cond_dist(self, treat_f, cov_f) -> torch.distributions.Distribution:
        # Returning potential outcome distributions
        if (treat_f == 0.0).all():
            return self.target_flow_dists[0]
        elif (treat_f == 1.0).all():
            return self.target_flow_dists[1]
        else:
            raise NotImplementedError()

    def get_bounds(self, cov_f, delta_scaled_or_alpha, mode='cdf') -> Tuple[torch.Tensor]:
        delta_scaled_or_alpha = torch.tensor(delta_scaled_or_alpha).float()
        bounds_dict = self.get_plugin_bounds(cov_f, delta_scaled_or_alpha, mode)
        return bounds_dict['lb'], bounds_dict['ub']
