#!/usr/bin/env python3
"""
Standard Kalman filter for state estimation.
"""

import numpy as np
from .base_filter import BaseFilter

class KF(BaseFilter):
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_Sigma_v,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_Sigma_v,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                 x0_scale=None, w_scale=None, v_scale=None,
                 input_lower_bound=None, input_upper_bound=None):
        super().__init__(T, dist, noise_dist, system_data, B,
                        true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                        nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v,
                        x0_max, x0_min, w_max, w_min, v_max, v_min,
                        x0_scale, w_scale, v_scale, None,
                        input_lower_bound, input_upper_bound)
        
        self.nominal_Sigma_v = nominal_Sigma_v


    def _initial_update(self, x_est_init, y0):
        P0 = self.nominal_x0_cov.copy()
        S0 = self.C @ P0 @ self.C.T + self.nominal_Sigma_v
        K0 = P0 @ self.C.T @ np.linalg.inv(S0)
        innovation0 = y0 - (self.C @ x_est_init + self.nominal_mu_v)
        self._P = np.zeros((self.T+1, self.nx, self.nx))
        self._P[0] = (np.eye(self.nx) - K0 @ self.C) @ P0
        return x_est_init + K0 @ innovation0
    
    def _kalman_update(self, x_pred, y, t):
        P_pred = self.A @ self._P[t-1] @ self.A.T + self.nominal_Sigma_w
        S_t = self.C @ P_pred @ self.C.T + self.nominal_Sigma_v
        K_t = P_pred @ self.C.T @ np.linalg.inv(S_t)
        innovation = y - (self.C @ x_pred + self.nominal_mu_v)
        x_new = x_pred + K_t @ innovation
        self._P[t] = (np.eye(self.nx) - K_t @ self.C) @ P_pred
        return x_new
    
    def forward(self):
        return self._run_simulation_loop(self._kalman_update)
    
    def forward_track(self, desired_trajectory):
        return self._run_simulation_loop(self._kalman_update, desired_trajectory)
    
    def forward_track_MPC(self, desired_trajectory):
        return self._run_simulation_loop_MPC(self._kalman_update, desired_trajectory)