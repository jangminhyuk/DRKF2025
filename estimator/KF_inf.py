#!/usr/bin/env python3
"""
Steady-state infinite-horizon Kalman filter.
"""

import numpy as np
from .base_filter import BaseFilter

class KF_inf(BaseFilter):
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
        self.compute_steady_state()


    def compute_steady_state(self):
        tol = 1e-6
        max_iter = 1000
        P = self.nominal_x0_cov.copy()
        for i in range(max_iter):
            S = self.C @ P @ self.C.T + self.nominal_Sigma_v
            K = P @ self.C.T @ np.linalg.inv(S)
            P_next = self.A @ P @ self.A.T + self.nominal_Sigma_w - self.A @ P @ self.C.T @ np.linalg.inv(S) @ self.C @ P @ self.A.T
            if np.linalg.norm(P_next - P, ord='fro') < tol:
                P = P_next
                break
            P = P_next
        self.P_inf = P
        self.K_inf = P @ self.C.T @ np.linalg.inv(self.C @ P @ self.C.T + self.nominal_Sigma_v)
    
    def _initial_update(self, x_est_init, y0):
        innovation0 = y0 - (self.C @ x_est_init + self.nominal_mu_v)
        return x_est_init + self.K_inf @ innovation0
    
    def _steady_state_update(self, x_pred, y, t):
        innovation = y - (self.C @ x_pred + self.nominal_mu_v)
        return x_pred + self.K_inf @ innovation
    
    def forward(self):
        return self._run_simulation_loop(self._steady_state_update)

    def forward_track(self, desired_trajectory):
        return self._run_simulation_loop(self._steady_state_update, desired_trajectory)
    
    def forward_track_MPC(self, desired_trajectory):
        return self._run_simulation_loop_MPC(self._steady_state_update, desired_trajectory)
