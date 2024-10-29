import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import numpy as np
from lightning.fabric.utilities.seed import seed_everything
from sklearn.model_selection import ShuffleSplit, KFold

from src.models.utils import subset_by_indices


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Initialisation of train_data_dict
    torch.set_default_device(args.exp.device)
    seed_everything(args.exp.seed)
    dataset = instantiate(args.dataset, _recursive_=True)
    train_data_dict = dataset.get_data()

    results = {}

    # Initialisation of model

    # Support bounds
    args.model.scaled_out_f_bound = float(train_data_dict['out_f_scaled'].max() - train_data_dict['out_f_scaled'].min()) + 5.0

    args.dataset.delta_min = float(train_data_dict['out_f'][train_data_dict['treat_f'] == 1.0].min() - train_data_dict['out_f'][train_data_dict['treat_f'] == 0.0].max())
    args.dataset.delta_max = float(train_data_dict['out_f'][train_data_dict['treat_f'] == 1.0].max() - train_data_dict['out_f_scaled'][train_data_dict['treat_f'] == 0.0].min())

    args.dataset.delta_scaled_min = float(train_data_dict['out_f_scaled'][train_data_dict['treat_f'] == 1.0].min() - train_data_dict['out_f_scaled'][train_data_dict['treat_f'] == 0.0].max())
    args.dataset.delta_scaled_max = float(train_data_dict['out_f_scaled'][train_data_dict['treat_f'] == 1.0].max() - train_data_dict['out_f_scaled'][train_data_dict['treat_f'] == 0.0].min())
    args.model.scaled_target_bound = args.dataset.delta_scaled_max - args.dataset.delta_scaled_min + 5.0

    delta = np.linspace(args.dataset.delta_min, args.dataset.delta_max, args.model.n_delta)
    # delta_scaled = delta / train_data_dict['out_scaler.scale_']
    # lbs_cdf, ubs_cdf = [], []
    lbs_zero_cdf, ubs_zero_cdf = [], []

    for i in range(args.dataset.n_runs):

        model = instantiate(args.model, args, _recursive_=True)

        # Finetuning for the first split
        if args.model.tune_hparams:
            model.finetune(train_data_dict, {'cpu': 0.5, 'gpu': 0.0})

        # Fitting the model
        model.fit(train_data_dict=train_data_dict, log=args.exp.logging)

        # Evaluation log-prob
        results_in = model.evaluate(data_dict=train_data_dict, log=args.exp.logging, prefix='in')

        logger.info(f'Run: {i}; In-sample log-prob: {results_in}')

        # Evaluation bounds
        cov_f, _, _ = model.prepare_eval_data(train_data_dict)
        with torch.no_grad():
            # lb_cdf, ub_cdf = model.get_bounds(cov_f, delta_scaled, mode='cdf')
            lb_zero_cdf, ub_zero_cdf = model.get_bounds(cov_f, np.zeros(1), mode='cdf')
            # lbs_cdf.append(lb_cdf.cpu().numpy())
            # ubs_cdf.append(ub_cdf.cpu().numpy())
            lbs_zero_cdf.append(lb_zero_cdf.cpu().numpy())
            ubs_zero_cdf.append(ub_zero_cdf.cpu().numpy())

        run_id = model.mlflow_logger.run_id
        model.mlflow_logger.experiment.set_terminated(run_id) if args.exp.logging else None

    # lbs_cdf, ubs_cdf = np.array(lbs_cdf), np.array(ubs_cdf)
    lbs_zero_cdf, ubs_zero_cdf = np.array(lbs_zero_cdf), np.array(ubs_zero_cdf)
    dataset.save_results(lbs_zero_cdf, ubs_zero_cdf, run_id=run_id)

    mean_bound_width = (ubs_zero_cdf.mean(0) - lbs_zero_cdf.mean(0)).mean()
    logger.info(f'Mean bound width: {mean_bound_width}')

    return results


if __name__ == "__main__":
    main()