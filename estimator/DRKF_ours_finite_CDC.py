#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRKF_ours_finite_CDC.py implements a distributionally robust Kalman filter (DRKF) for state estimation
in a closed-loop LQR experiment. This is the CDC version with simplified SDP formulation.
"""

import numpy as np
import cvxpy as cp
from .base_filter import BaseFilter

class DRKF_ours_finite_CDC(BaseFilter):
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_Sigma_v,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_Sigma_v,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                 x0_scale=None, w_scale=None, v_scale=None,
                 theta_x=None, theta_v=None, shared_noise_sequences=None,
                 input_lower_bound=None, input_upper_bound=None):
        """
        Parameters:
          T             : Horizon length.
          dist, noise_dist : Distribution types ('normal' or 'quadratic').
          system_data   : Tuple (A, C).
          B             : Control input matrix.
          
          The following parameters are provided in two sets:
             (i) True parameters (used to simulate the system):
                 - true_x0_mean, true_x0_cov: initial state distribution.
                 - true_mu_w, true_Sigma_w: process noise.
                 - true_mu_v, true_Sigma_v: measurement noise.
             (ii) Nominal parameters (obtained via EM, used in filtering):
                 - Use known means (nominal_x0_mean, nominal_mu_w, nominal_mu_v) and
                   EM–estimated covariances (nominal_x0_cov, nominal_Sigma_w, nominal_Sigma_v).
          x0_max, x0_min, etc.: Bounds for non–normal distributions.
          theta_x, theta_v: DRKF parameters.
          shared_noise_sequences: Pre-generated noise sequences for consistent experiments.
        """
        super().__init__(T, dist, noise_dist, system_data, B,
                        true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                        nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v,
                        x0_max, x0_min, w_max, w_min, v_max, v_min,
                        x0_scale, w_scale, v_scale, shared_noise_sequences,
                        input_lower_bound, input_upper_bound)
        
        self.theta_x = theta_x
        self.theta_v = theta_v
        
        # Allocate arrays for worst-case covariance matrices
        self.wc_Sigma_v = np.zeros((T+1, self.ny, self.ny))
        self.wc_Xprior = np.zeros((T+1, self.nx, self.nx))
        self.wc_Xpost = np.zeros((T+1, self.nx, self.nx))
        
        self.solve_DRSDP_offline()

    # --- Solve DR-SDP to obtain worst-case covariance matrices and DR Kalman gain for each time step (offline)---
    def solve_DRSDP_offline(self):
        X_pred_hat = self.nominal_x0_cov
        
        for t in range(self.T + 1):
            self.wc_Sigma_v[t], self.wc_Xprior[t], self.wc_Xpost[t] = self.solve_sdp(X_pred_hat)
            X_pred_hat = self.A @ self.wc_Xpost[t] @ self.A.T + self.nominal_Sigma_w
            
        

    # --- SDP Formulation and Solver for Worst-Case Measurement Covariance ---
    def create_DR_sdp(self):
        # Compute lambda_min for nominal measurement noise covariance (Sigma_v_hat)
        lambda_min_val = np.linalg.eigvalsh(self.nominal_Sigma_v).min()
        
        # Construct the SDP problem.
        # Variables
        X = cp.Variable((self.nx, self.nx), symmetric=True, name='X')
        X_pred = cp.Variable((self.nx, self.nx), symmetric=True, name='X_pred')
        Sigma_v = cp.Variable((self.ny, self.ny), symmetric=True, name='Sigma_v')
        Y = cp.Variable((self.nx, self.nx), name='Y')
        Z = cp.Variable((self.ny, self.ny), name='Z')
        
        # Parameters
        X_pred_hat = cp.Parameter((self.nx, self.nx), name='X_pred_hat')
        theta_x = cp.Parameter(nonneg=True, name='theta_x')
        Sigma_v_hat = cp.Parameter((self.ny, self.ny), name='Sigma_v_hat')  # nominal measurement noise covariance
        theta_v = cp.Parameter(nonneg=True, name='theta_v')
        
        
        # Objective: maximize trace(X)
        obj = cp.Maximize(cp.trace(X))
        
        # Constraints using Schur complements and the additional constraint on Sigma_v.
        constraints = [
            cp.bmat([[X_pred - X, X_pred @ self.C.T],
                     [self.C @ X_pred, self.C @ X_pred @ self.C.T + Sigma_v]
                    ]) >> 0,
            cp.trace(X_pred_hat + X_pred - 2*Y) <= theta_x**2,
            cp.bmat([[X_pred_hat, Y],
                     [Y.T, X_pred]
                    ]) >> 0,
            cp.trace(Sigma_v_hat + Sigma_v - 2*Z) <= theta_v**2,
            cp.bmat([[Sigma_v_hat, Z],
                     [Z.T, Sigma_v]
                    ]) >> 0,
            X >> 0,
            X_pred >> 0,
            Sigma_v >> 0,
            # Sigma_v is larger than lambda_min(Sigma_v_hat)*I
            Sigma_v >> lambda_min_val * np.eye(self.ny)
        ]
        
        prob = cp.Problem(obj, constraints)
        return prob
    
    def solve_sdp(self, X_pred_hat):
        prob = self.create_DR_sdp()
        params = prob.parameters()
        
        
        params[0].value = X_pred_hat
        params[1].value = self.theta_x
        params[2].value = self.nominal_Sigma_v
        params[3].value = self.theta_v
        
        prob.solve(solver=cp.MOSEK)
        
        if prob.status in ["infeasible", "unbounded"]:
            print(prob.status, 'finite DRKF SDP CDC formulation')
            
        sol = prob.variables()
        
        worst_case_Xpost = sol[0].value
        worst_case_Xprior = sol[1].value
        worst_case_Sigma_v = sol[2].value
        
        
        return worst_case_Sigma_v, worst_case_Xprior, worst_case_Xpost

    # --- DR-KF Update Step ---
    def DR_kalman_filter(self, v_mean_hat, x_pred, y, t):
        y_pred = self.C @ x_pred + v_mean_hat
        innovation = y - y_pred
        S = self.C @ self.wc_Xprior[t] @ self.C.T + self.wc_Sigma_v[t]
        K = self.wc_Xprior[t] @ self.C.T @ np.linalg.inv(S)
        x_new = x_pred + K @ innovation
        return x_new

    def _initial_update(self, x_est_init, y0):
        return self.DR_kalman_filter(self.nominal_mu_v, x_est_init, y0, 0)
    
    def _drkf_finite_cdc_update(self, x_pred, y, t):
        return self.DR_kalman_filter(self.nominal_mu_v, x_pred, y, t)
    
    def forward(self):
        return self._run_simulation_loop(self._drkf_finite_cdc_update)
    
    def forward_track(self, desired_trajectory):
        return self._run_simulation_loop(self._drkf_finite_cdc_update, desired_trajectory)
    
    def forward_track_MPC(self, desired_trajectory):
        return self._run_simulation_loop_MPC(self._drkf_finite_cdc_update, desired_trajectory)
