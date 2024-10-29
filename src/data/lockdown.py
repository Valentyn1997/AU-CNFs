import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, expon

from src import ROOT_PATH


class Lockdown:

    def __init__(self, dim_cov, **kwargs):
        self.train_data_path = f"{ROOT_PATH}/data/lockdown/weekly_data_preprocessed.csv"
        self.train_df = None
        self.dim_cov = dim_cov

    def get_data(self):
        self.train_df = pd.read_csv(self.train_data_path)

        out_f = self.train_df['log_incidence_rate'].values
        if self.dim_cov == 2:
            cov_f = self.train_df[['log_incidence_rate_prev_1', 'lockdown_prev_2']].values
        elif self.dim_cov == 3:
            cov_f = self.train_df[['log_incidence_rate_prev_1', 'log_incidence_rate_prev_2', 'lockdown_prev_2']].values
        elif self.dim_cov == 4:
            cov_f = self.train_df[['log_incidence_rate_prev_1', 'log_incidence_rate_prev_2', 'log_incidence_rate_prev_3', 'lockdown_prev_2']].values
        else:
            raise NotImplementedError()
        # treat_f = train_df['lockdown_bin'].values
        treat_f = (self.train_df['lockdown_prev_1'].values > 0.5).astype(float)

        out_scaler = StandardScaler()
        out_f_scaled = out_scaler.fit_transform(out_f.reshape(-1, 1)).reshape(-1)

        train_data_dict = {
            'cov_f': cov_f,
            'treat_f': treat_f,
            'out_f': out_f,
            'out_f_scaled': out_f_scaled,
            'out_scaler.scale_': float(out_scaler.scale_)
        }
        return train_data_dict

    def save_results(self, lbs, ubs, run_id):
        lb_df = pd.DataFrame(lbs.T, columns=[f'lb_{i}' for i in range(lbs.shape[0])])
        ub_df = pd.DataFrame(ubs.T, columns=[f'ub_{i}' for i in range(ubs.shape[0])])
        df = pd.concat([self.train_df, lb_df, ub_df], axis=1)
        save_path = f"{ROOT_PATH}/data/lockdown/weekly_data_results_{run_id}.csv"
        df.to_csv(save_path)
