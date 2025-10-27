#!/usr/bin/env python3
"""
Distributionally robust Kalman filter (DRKF) infinite horizon.
"""

import numpy as np
import cvxpy as cp
from LQR_with_estimator.base_filter import BaseFilter

class DRKF_ours_inf(BaseFilter):
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_Sigma_v,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_Sigma_v,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                 x0_scale=None, w_scale=None, v_scale=None,
                 theta_w=None, theta_v=None,
                 input_lower_bound=None, input_upper_bound=None):
"""Initialize infinite horizon DRKF with robust parameters.
        """
        super().__init__(T, dist, noise_dist, system_data, B,
                        true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                        nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v,
                        x0_max, x0_min, w_max, w_min, v_max, v_min,
                        x0_scale, w_scale, v_scale, None,
                        input_lower_bound, input_upper_bound)
        self.theta_w = theta_w
        self.theta_v = theta_v
        
        worst_case_Sigma_v, worst_case_Xprior, status = self.solve_sdp()
        self.wc_Sigma_v = worst_case_Sigma_v
        self.wc_Xprior = worst_case_Xprior


    # --- SDP Formulation and Solver for Worst-Case Measurement Covariance ---
    def create_DR_sdp(self):
        
        # Construct the SDP problem.
        # Variables
        X = cp.Variable((self.nx, self.nx), symmetric=True, name='X')
        X_pred = cp.Variable((self.nx, self.nx), symmetric=True, name='X_pred')
        Sigma_v = cp.Variable((self.ny, self.ny), symmetric=True, name='Sigma_v')
        Sigma_w = cp.Variable((self.nx, self.nx), symmetric=True, name='Sigma_w')
        Y = cp.Variable((self.nx, self.nx), name='Y')
        Z = cp.Variable((self.ny, self.ny), name='Z')
        
        # Parameters
        Sigma_w_hat = cp.Parameter((self.nx, self.nx), name='Sigma_w_hat')  # nominal process noise covariance
        theta_w = cp.Parameter(nonneg=True, name='theta_w')
        Sigma_v_hat = cp.Parameter((self.ny, self.ny), name='Sigma_v_hat')  # nominal measurement noise covariance
        theta_v = cp.Parameter(nonneg=True, name='theta_v')
        
        
        # Objective: maximize trace(X)
        obj = cp.Maximize(cp.trace(X))
        
        # Constraints using Schur complements and the additional constraint on Sigma_v.
        constraints = [
            cp.bmat([[X_pred - X, X_pred @ self.C.T],
                     [self.C @ X_pred, self.C @ X_pred @ self.C.T + Sigma_v]
                    ]) >> 0,
            cp.trace(Sigma_w_hat + Sigma_w - 2*Y) <= theta_w**2,
            cp.bmat([[Sigma_w_hat, Y],
                     [Y.T, Sigma_w]
                    ]) >> 0,
            cp.trace(Sigma_v_hat + Sigma_v - 2*Z) <= theta_v**2,
            cp.bmat([[Sigma_v_hat, Z],
                     [Z.T, Sigma_v]
                    ]) >> 0,
            X_pred == self.A @ X @ self.A.T + Sigma_w,             
            X >> 0,
            X_pred >> 0,
            Sigma_v >> 0,
            Sigma_w >> 0
        ]
        
        prob = cp.Problem(obj, constraints)
        return prob
    
    def solve_sdp(self):
        prob = self.create_DR_sdp()
        params = prob.parameters()
        params[0].value = self.nominal_Sigma_w
        params[1].value = self.theta_w
        params[2].value = self.nominal_Sigma_v
        params[3].value = self.theta_v
        
        prob.solve(solver=cp.MOSEK)
        
        if prob.status in ["infeasible", "unbounded"]:
            print(prob.status, 'inf DRKF SDP problem')
            
        sol = prob.variables()
        
        
        worst_case_Xprior = sol[1].value
        worst_case_Sigma_v = sol[2].value 
        return worst_case_Sigma_v, worst_case_Xprior, prob.status

    # --- DR-KF Update Step ---
    def DR_kalman_filter(self, v_mean_hat, x_pred, y):
        y_pred = self.C @ x_pred + v_mean_hat
        innovation = y - y_pred
        S = self.C @ self.wc_Xprior @ self.C.T + self.wc_Sigma_v
        K = self.wc_Xprior @ self.C.T @ np.linalg.inv(S)
        x_new = x_pred + K @ innovation
        return x_new

    def _initial_update(self, x_est_init, y0):
        return self.DR_kalman_filter(self.nominal_mu_v, x_est_init, y0)
    
    def _drkf_update(self, x_pred, y, t):
        return self.DR_kalman_filter(self.nominal_mu_v, x_pred, y)
    
    def forward(self):
        return self._run_simulation_loop(self._drkf_update)
    def forward_track(self, desired_trajectory):
        return self._run_simulation_loop(self._drkf_update, desired_trajectory)
    
    def forward_track_MPC(self, desired_trajectory):
        return self._run_simulation_loop_MPC(self._drkf_update, desired_trajectory)
