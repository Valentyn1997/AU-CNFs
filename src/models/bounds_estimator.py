import torch
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger
import logging
from ray import tune
import ray
from copy import deepcopy
from typing import Tuple, Union

from src.models.utils import fit_eval_kfold

logger = logging.getLogger(__name__)


class BoundsEstimator(torch.nn.Module):
    """
    Abstract class for
    """

    tune_criterion = None

    def __init__(self, args: DictConfig = None, **kwargs):
        super(BoundsEstimator, self).__init__()

        # Dataset params
        self.dim_out, self.dim_cov, self.dim_treat = 1, args.model.dim_cov, 1
        assert self.dim_treat == 1

        # Model hyparams
        self.hparams = args

        # MlFlow Logger
        if args.exp.logging:
            experiment_name = f'{args.model.name}/{args.dataset.name}'
            self.mlflow_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri)

    def prepare_train_data(self, train_data_dict: dict):
        """
        Data pre-processing
        :param train_data_dict: Dictionary with the training data
        """
        raise NotImplementedError()

    def prepare_eval_data(self, data_dict: dict):
        """
        Data pre-processing
        :param data_dict: Dictionary with the evaluation data
        """
        raise NotImplementedError()

    def prepare_tensors(self, cov=None, treat=None, out=None, kind='torch') -> Tuple[dict]:
        """
        Conversion of tensors
        @param cov: Tensor with covariates
        @param treat: Tensor with treatments
        @param out: Tensor with outcomes
        @param kind: torch / numpy
        @return: cov, treat, out
        """
        if kind == 'torch':
            cov = torch.tensor(cov).reshape(-1, self.dim_cov).float() if cov is not None else None
            treat = torch.tensor(treat).reshape(-1, self.dim_treat).float() if treat is not None else None
            out = torch.tensor(out).reshape(-1, self.dim_out).float() if out is not None else None
        elif kind == 'numpy':
            cov = cov.reshape(-1, self.dim_cov) if cov is not None else None
            treat = treat.reshape(-1).astype(float) if treat is not None else None
            out = out.reshape(-1, self.dim_out) if out is not None else None
        else:
            raise NotImplementedError()
        return cov, treat, out

    def get_bounds(self, cov_f, delta_scaled, mode='cdf', return_arg=False) -> Tuple[torch.Tensor]:
        raise NotImplementedError()

    def fit(self, train_data_dict: dict, log: bool) -> None:
        """
        Fitting the estimator
        @param train_data_dict: Training data dictionary
        @param log: Logging to the MlFlow
        """
        raise NotImplementedError()

    def evaluate(self, data_dict: dict, log: bool, prefix: str) -> dict:
        raise NotImplementedError()

    def evaluate_bounds(self, data_dict: dict, dataset, log: bool, prefix: str) -> dict:
        delta = np.linspace(self.hparams.dataset.delta_min, self.hparams.dataset.delta_max, self.hparams.model.n_delta)
        delta_scaled = delta / data_dict['out_scaler.scale_']
        alpha = np.linspace(0.0, 1.0, delta_scaled.shape[0] + 2)[1:-1]

        lb_cdf_gt, ub_cdf_gt = dataset.get_bounds(delta.reshape(-1, 1), data_dict, mode='cdf')
        lb_icdf_gt, ub_icdf_gt = dataset.get_bounds(alpha.reshape(-1, 1), data_dict, mode='icdf')

        cov_f, treat_f, out_f_scaled = self.prepare_eval_data(data_dict)

        with torch.no_grad():
            lb_cdf, ub_cdf = self.get_bounds(cov_f, delta_scaled, mode='cdf')
            lb_icdf, ub_icdf = self.get_bounds(cov_f, alpha, mode='icdf')

            lb_cdf = lb_cdf.cpu().numpy() if isinstance(lb_cdf, torch.Tensor) else lb_cdf
            ub_cdf = ub_cdf.cpu().numpy() if isinstance(ub_cdf, torch.Tensor) else ub_cdf
            lb_icdf = lb_icdf.cpu().numpy() if isinstance(lb_icdf, torch.Tensor) else lb_icdf
            ub_icdf = ub_icdf.cpu().numpy() if isinstance(ub_icdf, torch.Tensor) else ub_icdf

            lb_icdf_unscaled, ub_icdf_unscaled = lb_icdf * data_dict['out_scaler.scale_'], ub_icdf * data_dict['out_scaler.scale_']


        results = {
            f'lb_cdf_dist_{prefix}': np.sqrt((delta[1] - delta[0]) * ((lb_cdf_gt - lb_cdf) ** 2).sum(0)).mean(),
            # d_2 CDF distance
            f'ub_cdf_dist_{prefix}': np.sqrt((delta[1] - delta[0]) * ((ub_cdf_gt - ub_cdf) ** 2).sum(0)).mean(),
            # d_2 CDF distance
            f'lb_icdf_dist_{prefix}': np.sqrt((alpha[1] - alpha[0]) * ((lb_icdf_gt - lb_icdf_unscaled) ** 2).sum(0)).mean(),
            # d_2 quantile distance
            f'ub_icdf_dist_{prefix}': np.sqrt((alpha[1] - alpha[0]) * ((ub_icdf_gt - ub_icdf_unscaled) ** 2).sum(0)).mean()
            # d_2 quantile distance
        }

        if log:
            self.mlflow_logger.log_metrics(results, step=self.num_train_iter)
        return results

    def save_train_data_to_buffer(self, cov_f, treat_f, out_f_scaled) -> None:
        """
        Save train data for non-parametric inference of two-stage training
        @param cov_f: Tensor with factual covariates
        @param treat_f: Tensor with factual treatments
        @param out_f: Tensor with factual outcomes
        """
        self.cov_f = cov_f
        self.treat_f = treat_f
        self.out_f_scaled = out_f_scaled

    @staticmethod
    def set_hparams(model_args: DictConfig, new_model_args: dict):
        for k in new_model_args.keys():
            assert k in model_args.keys()
            model_args[k] = new_model_args[k]

    # def set_norm_consts(self, add_bound=2.5) -> None:
    #     """
    #     Calculating normalization constants for improperly normalized density estimators
    #     @param add_bound: Additional bounds for normalization of truncated series estimator
    #     """
    #
    #     self.norm_const = [1.0, 1.0]
    #     logger.info('Calculating normalization constants')
    #
    #     for i, treat_option in enumerate(self.treat_options):
    #         if self.dim_out == 1:
    #             norm_bins_effect = self.norm_bins
    #             out_pot = np.linspace(self.out_f.min() - add_bound, self.out_f.max() + add_bound, norm_bins_effect)
    #             dx = (self.out_f.max() - self.out_f.min() + 2 * add_bound) / norm_bins_effect
    #         elif self.dim_out == 2:
    #             norm_bins_sqrt = int(np.sqrt(self.norm_bins)) + 1
    #             norm_bins_effect = norm_bins_sqrt ** 2
    #             out_pot_0 = np.linspace(self.out_f[:, 0].min() - add_bound, self.out_f[:, 0].max() + add_bound, norm_bins_sqrt)
    #             out_pot_1 = np.linspace(self.out_f[:, 1].min() - add_bound, self.out_f[:, 1].max() + add_bound, norm_bins_sqrt)
    #             out_pot_0, out_pot_1 = np.meshgrid(out_pot_0, out_pot_1)
    #             out_pot = np.concatenate([out_pot_0.reshape(-1, 1), out_pot_1.reshape(-1, 1)], axis=1)
    #             dx = (self.out_f[:, 0].max() - self.out_f[:, 0].min() + 2 * add_bound) *\
    #                  (self.out_f[:, 1].max() - self.out_f[:, 1].min() + 2 * add_bound) / norm_bins_effect
    #         else:
    #             raise NotImplementedError()
    #         treat_pot = treat_option * np.ones((norm_bins_effect,))
    #         log_prob = self.inter_log_prob(treat_pot, out_pot)
    #         if isinstance(log_prob, torch.Tensor):
    #             log_prob = log_prob.detach().numpy()
    #         prob = np.exp(log_prob)
    #         self.norm_const[i] = prob.sum() * dx

    def finetune(self, train_data_dict: dict, resources_per_trial: dict, val_data_dict: dict = None):
        """
        Hyperparameter tuning with ray[tune]
        @param train_data_dict: Training data dictionary
        @param resources_per_trial: CPU / GPU resources dictionary
        @return: self
        """

        logger.info(f"Running hyperparameters selection with {self.hparams.model['tune_range']} trials")
        logger.info(f'Using {self.tune_criterion} for hyperparameters selection')
        ray.init(num_gpus=0, num_cpus=2)

        hparams_grid = {k: getattr(tune, self.hparams.model['tune_type'])(list(v))
                        for k, v in self.hparams.model['hparams_grid'].items()}
        analysis = tune.run(tune.with_parameters(fit_eval_kfold,
                                                 model_cls=self.__class__,
                                                 train_data_dict=deepcopy(train_data_dict),
                                                 val_data_dict=deepcopy(val_data_dict),
                                                 orig_hparams=self.hparams),
                            resources_per_trial=resources_per_trial,
                            raise_on_failed_trial=False,
                            metric="val_metric",
                            mode="max",
                            config=hparams_grid,
                            num_samples=self.hparams.model['tune_range'],
                            name=f"{self.__class__.__name__}",
                            max_failures=1,
                            )
        ray.shutdown()

        logger.info(f"Best hyperparameters found: {analysis.best_config}.")
        logger.info("Resetting current hyperparameters to best values.")
        self.set_hparams(self.hparams.model, analysis.best_config)

        self.__init__(self.hparams)
        return self
