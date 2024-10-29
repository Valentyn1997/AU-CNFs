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
    data_dicts = dataset.get_data() if args.dataset.collection else [dataset.get_data()]
    if args.dataset.dataset_ix is not None:
        data_dicts = [data_dicts[args.dataset.dataset_ix]]
        specific_ix = True
    else:
        specific_ix = False

    for ix, data_dict in enumerate(data_dicts):

        # Train-test split
        if args.dataset.train_test_splitted:
            train_data_dict, test_data_dict = data_dict[0], data_dict[1]
        else:
            if hasattr(args.dataset, 'k_fold'):
                rs = KFold(n_splits=args.dataset.k_fold, random_state=args.exp.seed, shuffle=True)
            else:
                rs = ShuffleSplit(n_splits=args.dataset.n_shuffle_splits, random_state=args.exp.seed,
                                  test_size=args.dataset.test_size)
            for split_ix, (train_index, test_index) in enumerate(rs.split(data_dict['cov_f'])):

                train_data_dict, test_data_dict = subset_by_indices(data_dict, train_index), subset_by_indices(data_dict, test_index)

        results = {}

        # Initialisation of model
        args.dataset.dataset_ix = ix if not specific_ix else args.dataset.dataset_ix
        # args.dataset.split_ix = split_ix

        # Support bounds
        args.model.scaled_out_f_bound = float(train_data_dict['out_f_scaled'].max() - train_data_dict['out_f_scaled'].min()) + 5.0

        args.dataset.delta_min = float(train_data_dict['out_pot1'].min() - train_data_dict['out_pot0'].max())
        args.dataset.delta_max = float(train_data_dict['out_pot1'].max() - train_data_dict['out_pot0'].min())

        args.dataset.delta_scaled_min = float(train_data_dict['out_f_scaled'][train_data_dict['treat_f'] == 1.0].min() - train_data_dict['out_f_scaled'][train_data_dict['treat_f'] == 0.0].max())
        args.dataset.delta_scaled_max = float(train_data_dict['out_f_scaled'][train_data_dict['treat_f'] == 1.0].max() - train_data_dict['out_f_scaled'][train_data_dict['treat_f'] == 0.0].min())
        args.model.scaled_target_bound = args.dataset.delta_scaled_max - args.dataset.delta_scaled_min + 5.0

        model = instantiate(args.model, args, _recursive_=True)

        # Finetuning for the first split
        if args.model.tune_hparams:
            model.finetune(train_data_dict, {'cpu': 0.5, 'gpu': 0.0})

        # Fitting the model
        model.fit(train_data_dict=train_data_dict, log=args.exp.logging)

        # Evaluation log-prob
        results_in = model.evaluate(data_dict=train_data_dict, log=args.exp.logging, prefix='in')
        results_out = model.evaluate(data_dict=test_data_dict, log=args.exp.logging, prefix='out')

        logger.info(f'Dataset ix: {ix}; In-sample log-prob: {results_in}; Out-sample log-prob: {results_out}')

        # Evaluation bounds
        results_in = model.evaluate_bounds(data_dict=train_data_dict, dataset=dataset, log=args.exp.logging, prefix='in')
        results_out = model.evaluate_bounds(data_dict=test_data_dict, dataset=dataset, log=args.exp.logging, prefix='out')
        logger.info(f'Dataset ix: {ix}; In-sample bounds rmse: {results_in}; Out-sample bounds rmse: {results_out}')


        model.mlflow_logger.experiment.set_terminated(model.mlflow_logger.run_id) if args.exp.logging else None

    return results


if __name__ == "__main__":
    main()