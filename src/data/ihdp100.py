import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, expon

from src import ROOT_PATH
from src.models.utils import lb_cdf_norm, ub_cdf_norm, lb_icdf_norm, ub_icdf_norm


class IHDP100:

    def __init__(self, **kwargs):
        self.train_data_path = f"{ROOT_PATH}/data/ihdp100/ihdp_npci_1-100.train.npz"
        self.test_data_path = f"{ROOT_PATH}/data/ihdp100/ihdp_npci_1-100.test.npz"

    def get_data(self):
        train_data = np.load(self.train_data_path, 'r')
        test_data = np.load(self.test_data_path, 'r')

        datasets = []

        for i in range(train_data['x'].shape[-1]):

            data_dicts = []
            out_scaler = None

            for kind, data in zip(['train', 'test'], [train_data, test_data]):
                if kind == 'train':
                    out_scaler = StandardScaler()
                    out_scaler.fit(data['yf'][:, i].reshape(-1, 1))

                data_dicts.append({
                    'cov_f': data['x'][:, :, i],
                    'treat_f': data['t'][:, i],
                    'out_f': data['yf'][:, i],
                    'out_f_scaled': out_scaler.transform(data['yf'][:, i].reshape(-1, 1)).reshape(-1),
                    'out_pot0': np.where(1.0 - data['t'][:, i], data['yf'][:, i], data['ycf'][:, i]),
                    'out_pot1': np.where(data['t'][:, i], data['yf'][:, i], data['ycf'][:, i]),
                    'out_scaler.scale_': float(out_scaler.scale_),
                    'mu0': data['mu0'][:, i],
                    'mu1': data['mu1'][:, i]
                })

            datasets.append(data_dicts)

        return datasets

    def get_bounds(self, delta_or_alpha, data_dict, mode='cdf'):
        if mode == 'cdf':
            return lb_cdf_norm(delta_or_alpha, loc0=data_dict['mu0'], scale0=1.0, loc1=data_dict['mu1'], scale1=1.0), \
                   ub_cdf_norm(delta_or_alpha, loc0=data_dict['mu0'], scale0=1.0, loc1=data_dict['mu1'], scale1=1.0)
        elif mode == 'icdf':
            return lb_icdf_norm(delta_or_alpha, loc0=data_dict['mu0'], scale0=1.0, loc1=data_dict['mu1'], scale1=1.0), \
                   ub_icdf_norm(delta_or_alpha, loc0=data_dict['mu0'], scale0=1.0, loc1=data_dict['mu1'], scale1=1.0)
        else:
            raise NotImplementedError()
