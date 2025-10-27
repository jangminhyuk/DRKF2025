#!/usr/bin/env python3
"""
Risk-sensitive filter for state estimation.
"""

import numpy as np
from .base_filter import BaseFilter

class RiskSensitive(BaseFilter):
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_Sigma_v,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_Sigma_v,
                 theta_rs,   # risk sensitivity parameter
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                 x0_scale=None, w_scale=None, v_scale=None,
                 input_lower_bound=None, input_upper_bound=None):
        """
        Parameters:
          T: Time horizon.
          dist, noise_dist: 'normal' or 'quadratic'
          system_data: Tuple (A, C)
          B: Control input matrix.
          true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v:
              True parameters for simulation.
          nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v:
          
          theta_rs: Risk sensitivity parameter.
          x0_max, x0_min, etc.: Bounds for non–normal distributions.
        """
        super().__init__(T, dist, noise_dist, system_data, B,
                        true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                        nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v,
                        x0_max, x0_min, w_max, w_min, v_max, v_min,
                        x0_scale, w_scale, v_scale, None,
                        input_lower_bound, input_upper_bound)
        
        self.D = 1.0*np.eye(self.nx)
        self.theta_rs = theta_rs
        self.theta_too_large = False  # Flag to track if theta is too large

    def _initial_update(self, x_est_init, y0):
        P0 = self.nominal_x0_cov.copy()

        # Risk-deformed prior: (P0^{-1} - θ D^T D)^{-1}
        P0_inv = np.linalg.inv(P0)
        M0 = P0_inv - self.theta_rs * (self.D.T @ self.D)
        # Guard: ensure SPD
        if not np.all(np.linalg.eigvalsh(M0) > 0):
            #print(f"Warning: θ={self.theta_rs} too large at initialization: P0^{{-1}} - θ D^T D not SPD.")
            self.theta_too_large = True
            #raise ValueError(f"θ={self.theta_rs} too large: P0^{{-1}} - θ D^T D not SPD.")
        P0_tilde = np.linalg.inv(M0)

        S0 = self.C @ P0_tilde @ self.C.T + self.nominal_Sigma_v
        K0 = P0_tilde @ self.C.T @ np.linalg.inv(S0)
        innovation0 = y0 - (self.C @ x_est_init + self.nominal_mu_v)
        x_post0 = x_est_init + K0 @ innovation0

        # Store the a‑posteriori covariance for step 0 (standard form)
        self._P = np.zeros((self.T+1, self.nx, self.nx))
        self._P[0] = (np.eye(self.nx) - K0 @ self.C) @ P0_tilde
        return x_post0
    
    def _risk_sensitive_update(self, x_pred, y, t):
        # If theta was too large during initialization, this method shouldn't be called
        if self.theta_too_large:
            raise ValueError(f"Risk-sensitive update called but θ={self.theta_rs} is too large.")
            
        # Predict covariance from the previous a‑posteriori
        P_pred = self.A @ self._P[t-1] @ self.A.T + self.nominal_Sigma_w

        # Risk-deformed prior
        P_inv = np.linalg.inv(P_pred)
        M = P_inv - self.theta_rs * (self.D.T @ self.D)
        if not np.all(np.linalg.eigvalsh(M) > 0):
            #print(f"Warning: θ={self.theta_rs} too large at t={t}: P^{{-1}} - θ D^T D not SPD.")
            self.theta_too_large = True
            #raise ValueError(f"θ={self.theta_rs} too large at t={t}: P^{{-1}} - θ D^T D not SPD.")
            
        P_tilde = np.linalg.inv(M)

        # Gain and update
        S_t = self.C @ P_tilde @ self.C.T + self.nominal_Sigma_v
        K_t = P_tilde @ self.C.T @ np.linalg.inv(S_t)
        innovation = y - (self.C @ x_pred + self.nominal_mu_v)
        x_post = x_pred + K_t @ innovation

        # Standard form covariance update (same as KF)
        self._P[t] = (np.eye(self.nx) - K_t @ self.C) @ P_tilde
        return x_post
    
    
    def forward(self):
        """Forward simulation using risk-sensitive filter."""
        try:
            result = self._run_simulation_loop(self._risk_sensitive_update)
            return result
        except ValueError as e:
            if "too large" in str(e):
                print(f"Skipping experiment: θ={self.theta_rs} is too large for risk-sensitive filtering.")
                return None
            else:
                raise e
        except Exception as e:
            print(f"Unexpected error in risk-sensitive filter with θ={self.theta_rs}: {e}")
            return None
    
    def forward_track(self, desired_trajectory):
        """Forward trajectory tracking using risk-sensitive filter."""
        try:
            result = self._run_simulation_loop(self._risk_sensitive_update, desired_trajectory)
            return result
        except ValueError as e:
            if "too large" in str(e):
                print(f"Skipping experiment: θ={self.theta_rs} is too large for risk-sensitive filtering.")
                return None
            else:
                raise e
        except Exception as e:
            print(f"Unexpected error in risk-sensitive filter with θ={self.theta_rs}: {e}")
            return None
    
    def forward_track_MPC(self, desired_trajectory):
        """Forward MPC trajectory tracking using risk-sensitive filter."""
        try:
            result = self._run_simulation_loop_MPC(self._risk_sensitive_update, desired_trajectory)
            return result
        except ValueError as e:
            if "too large" in str(e):
                print(f"Skipping experiment: θ={self.theta_rs} is too large for risk-sensitive filtering.")
                return None
            else:
                raise e
        except Exception as e:
            print(f"Unexpected error in risk-sensitive filter with θ={self.theta_rs}: {e}")
            return None