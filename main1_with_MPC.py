#!/usr/bin/env python3
"""
Trajectory tracking with MPC controller and state estimation comparison.
"""

import numpy as np
import argparse
import os
import pickle
from joblib import Parallel, delayed
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

from estimator.KF import KF
from estimator.KF_inf import KF_inf
from estimator.DRKF_ours_inf import DRKF_ours_inf
from estimator.DRKF_ours_finite import DRKF_ours_finite
from estimator.DRKF_ours_inf_CDC import DRKF_ours_inf_CDC
from estimator.DRKF_ours_finite_CDC import DRKF_ours_finite_CDC
from estimator.BCOT import BCOT 
from estimator.risk_sensitive import RiskSensitive
from estimator.risk_seek import RiskSeek
from estimator.DRKF_neurips import DRKF_neurips
from common_utils import (save_data, is_stabilizable, is_detectable, is_positive_definite,
                         enforce_positive_definiteness, compute_mpc_cost, generate_desired_trajectory)
from scipy.linalg import solve_discrete_are

from estimator.base_filter import BaseFilter

_temp_A, _temp_C = np.eye(2), np.eye(2)
_temp_params = np.zeros((2, 1)), np.eye(2)
_sampler = BaseFilter(1, 'normal', 'normal', (_temp_A, _temp_C), np.eye(2, 1),
                     *_temp_params, *_temp_params, *_temp_params,
                     *_temp_params, *_temp_params, *_temp_params)

def normal(mu, Sigma, N=1):
    return _sampler.normal(mu, Sigma, N)

def uniform(a, b, N=1):
    return _sampler.uniform(a, b, N).T if N > 1 else _sampler.uniform(a, b, N)

def quadratic(w_max, w_min, N=1):
    result = _sampler.quadratic(w_max, w_min, N)
    return result.T if N > 1 else result


# --- LQR Controller Computation (for MPC fallback) ---
def compute_lqr_gain(A, B, Q_lqr, R_lqr):
    """Compute LQR gain using discrete-time ARE."""
    P = solve_discrete_are(A, B, Q_lqr, R_lqr)
    K = np.linalg.inv(B.T @ P @ B + R_lqr) @ (B.T @ P @ A)
    return K

def generate_data(T, nx, ny, A, C, mu_w, Sigma_w, mu_v, M,
                  x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist,
                  x0_scale=None, w_scale=None, v_scale=None):
    x_true_all = np.zeros((T + 1, nx, 1))
    y_all = np.zeros((T, ny, 1))
    
    if dist == "normal":
        x_true = normal(x0_mean, x0_cov)
    elif dist == "quadratic":
        x_true = quadratic(x0_max, x0_min)
    x_true_all[0] = x_true
    
    for t in range(T):
        if dist == "normal":
            true_w = normal(mu_w, Sigma_w)
            true_v = normal(mu_v, M)
        elif dist == "quadratic":
            true_w = quadratic(w_max, w_min)
            true_v = quadratic(v_max, v_min)
        y_t = C @ x_true + true_v
        y_all[t] = y_t
        x_true = A @ x_true + true_w
        x_true_all[t+1] = x_true
    return x_true_all, y_all


def run_experiment(exp_idx, dist, num_sim, seed_base, robust_val, filters_to_execute, desired_traj):
    experiment_seed = seed_base + exp_idx * 12345
    np.random.seed(experiment_seed)
    T = desired_traj.shape[1] - 1
    dt = 0.2
    nx = 4
    ny = 2
    
    # Discrete-time double integrator system (4D position/velocity)
    A = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    B = np.array([[0.5 * dt**2, 0],
                  [dt, 0],
                  [0, 0.5 * dt**2],
                  [0, dt]])
    C = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    
    # Cost matrices for MPC trajectory tracking (also used for LQR fallback)
    Q_lqr = np.diag([10.0, 1.0, 10.0, 1.0])  # State cost
    R_lqr = 0.1 * np.eye(2)  # Control cost
    
    # Compute LQR gain for MPC fallback in trajectory tracking
    K_lqr = compute_lqr_gain(A, B, Q_lqr, R_lqr)
    
    # Input saturation bounds for MPC trajectory tracking
    input_lower_bound = -1000.0 
    input_upper_bound = 1000.0
    
    # Verify system properties
    if not is_stabilizable(A, B):
        print("Warning: (A, B) is not stabilizable!")
        return None, None
    
    system_data = (A, C)
    
    # Use the initial desired state as the system's initial state
    x0_mean = desired_traj[:, 0].reshape(-1, 1)
    
    nx = 4; nw = 4; nu = 2; ny = 2
    if dist == "normal":
        mu_w = 0.0 * np.ones((nw, 1))
        Sigma_w = 0.02 * np.eye(nw)
        x0_cov = 0.02 * np.eye(nx)
        mu_v = 0.0 * np.ones((ny, 1))
        Sigma_v = 0.02 * np.eye(ny)
        v_max = v_min = None
        w_max = w_min = x0_max = x0_min = None
        x0_scale = w_scale = v_scale = None
    elif dist == "quadratic":
        w_max = 0.1 * np.ones(nx)
        w_min = -0.1 * np.ones(nx)
        mu_w = (0.5 * (w_max + w_min))[:, None]
        Sigma_w = 3.0/20.0 * np.diag((w_max - w_min)**2)
        x0_max = 0.1 * np.ones(nx)
        x0_min = -0.1 * np.ones(nx)
        x0_mean = (0.5 * (x0_max + x0_min))[:, None]
        x0_cov = 3.0/20.0 * np.diag((x0_max - x0_min)**2)
        v_min = -0.1 * np.ones(ny)
        v_max = 0.1 * np.ones(ny)
        mu_v = (0.5 * (v_max + v_min))[:, None]
        Sigma_v = 3.0/20.0 * np.diag((v_max - v_min)**2)
        x0_scale = w_scale = v_scale = None
    else:
        raise ValueError("Unsupported noise distribution.")
    
    # --- Generate Data for EM ---
    N_data = 5
    _, y_all_em = generate_data(N_data, nx, ny, A, C,
                                mu_w, Sigma_w, mu_v, Sigma_v,
                                x0_mean, x0_cov, x0_max, x0_min,
                                w_max, w_min, v_max, v_min, dist,
                                x0_scale, w_scale, v_scale)
    y_all_em = y_all_em.squeeze()
    
    # --- EM Estimation of Nominal Covariances ---
    # Ensure EM algorithm is reproducible within this experiment
    np.random.seed(experiment_seed + 1000)  # Different seed offset for EM
    
    mu_w_hat = np.zeros((nx, 1))
    mu_v_hat = np.zeros((ny, 1))
    mu_x0_hat = x0_mean.copy()
    Sigma_w_hat = np.eye(nx)
    Sigma_v_hat = np.eye(ny)
    Sigma_x0_hat = x0_cov.copy()
    
    kf_em = KalmanFilter(transition_matrices=A,
                           observation_matrices=C,
                           transition_covariance=Sigma_w_hat,
                           observation_covariance=Sigma_v_hat,
                           transition_offsets=mu_w_hat.squeeze(),
                           observation_offsets=mu_v_hat.squeeze(),
                           initial_state_mean=mu_x0_hat.squeeze(),
                           initial_state_covariance=Sigma_x0_hat,
                           em_vars=['transition_covariance', 'observation_covariance',
                                    'transition_offsets', 'observation_offsets'])
    max_iter = 100
    eps_log = 1e-4
    loglikelihoods = np.zeros(max_iter)
    for i in range(max_iter):
        kf_em = kf_em.em(X=y_all_em, n_iter=1)
        loglikelihoods[i] = kf_em.loglikelihood(y_all_em)
        Sigma_w_hat = kf_em.transition_covariance
        Sigma_v_hat = kf_em.observation_covariance
        mu_x0_hat = kf_em.initial_state_mean
        Sigma_x0_hat = kf_em.initial_state_covariance
        if i > 0 and (loglikelihoods[i] - loglikelihoods[i-1] <= eps_log):
            break
    
    Sigma_w_hat = enforce_positive_definiteness(Sigma_w_hat)
    Sigma_v_hat = enforce_positive_definiteness(Sigma_v_hat)
    Sigma_x0_hat = enforce_positive_definiteness(Sigma_x0_hat)
    
    nominal_mu_w    = mu_w
    nominal_Sigma_w = Sigma_w_hat.copy()
    nominal_mu_v    = mu_v
    nominal_Sigma_v = Sigma_v_hat.copy()
    nominal_x0_mean = x0_mean
    nominal_x0_cov  = Sigma_x0_hat.copy()
    
    if not is_detectable(A, C):
        print("Warning: (A, C) is not detectable!")
        exit()
    if not is_positive_definite(nominal_Sigma_w):
        print("Warning: nominal_Sigma_w is not positive definite!")
        exit()
    if not is_positive_definite(nominal_Sigma_v):
        print("Warning: nominal_Sigma_v (noise covariance) is not positive definite!")
        exit()
    
    
    # --- Generate Shared Noise Sequences for Fair Comparison ---
    # Generate noise sequences once per experiment, reuse across all filters
    shared_noise_sequences = []
    for sim_idx in range(num_sim):
        np.random.seed(experiment_seed + 2000 + sim_idx)
        
        # Generate noise for this simulation
        if dist == "normal":
            x0_noise = normal(x0_mean, x0_cov)
            w_noise = [normal(mu_w, Sigma_w) for _ in range(T+1)]
            v_noise = [normal(mu_v, Sigma_v) for _ in range(T+1)]
        elif dist == "quadratic":
            x0_noise = quadratic(x0_max, x0_min)
            w_noise = [quadratic(w_max, w_min) for _ in range(T+1)]
            v_noise = [quadratic(v_max, v_min) for _ in range(T+1)]
            
        shared_noise_sequences.append({
            'x0': x0_noise,
            'w': w_noise,
            'v': v_noise
        })
    
    # --- Simulation Functions for Each Filter (No Control) ---
    def run_simulation_finite(sim_idx_local):
        estimator = KF(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[sim_idx_local]
        estimator._noise_index = 0
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[sim_idx_local]
        estimator._noise_index = 0
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
        if res is None:
            return None
        cost = compute_mpc_cost(res, Q_lqr, R_lqr, desired_traj)
        return res, cost

    def run_simulation_inf_kf(sim_idx_local):
        estimator = KF_inf(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[sim_idx_local]
        estimator._noise_index = 0
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[sim_idx_local]
        estimator._noise_index = 0
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
        if res is None:
            return None
        cost = compute_mpc_cost(res, Q_lqr, R_lqr, desired_traj)
        return res, cost

    def run_simulation_inf_drkf(sim_idx_local):
        estimator = DRKF_ours_inf(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            theta_w=robust_val, theta_v=robust_val,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[sim_idx_local]
        estimator._noise_index = 0
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
        if res is None:
            return None
        cost = compute_mpc_cost(res, Q_lqr, R_lqr, desired_traj)
        return res, cost
    
    def run_simulation_finite_drkf(sim_idx_local):
        estimator = DRKF_ours_finite(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            theta_x=robust_val, theta_v=robust_val, theta_w=robust_val,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[sim_idx_local]
        estimator._noise_index = 0
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
        if res is None:
            return None
        cost = compute_mpc_cost(res, Q_lqr, R_lqr, desired_traj)
        return res, cost

    def run_simulation_bcot(sim_idx_local):
        estimator = BCOT(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            radius=robust_val, maxit=20,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[sim_idx_local]
        estimator._noise_index = 0
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
        if res is None:
            return None
        cost = compute_mpc_cost(res, Q_lqr, R_lqr, desired_traj)
        return res, cost

    def run_simulation_inf_drkf_cdc(sim_idx_local):
        estimator = DRKF_ours_inf_CDC(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            theta_x=robust_val, theta_v=robust_val,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[sim_idx_local]
        estimator._noise_index = 0
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
        if res is None:
            return None
        cost = compute_mpc_cost(res, Q_lqr, R_lqr, desired_traj)
        return res, cost
    
    def run_simulation_finite_drkf_cdc(sim_idx_local):
        estimator = DRKF_ours_finite_CDC(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            theta_x=robust_val, theta_v=robust_val,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[sim_idx_local]
        estimator._noise_index = 0
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
        if res is None:
            return None
        cost = compute_mpc_cost(res, Q_lqr, R_lqr, desired_traj)
        return res, cost

    def run_simulation_risk(sim_idx_local):
        estimator = RiskSensitive(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=x0_mean,  # known initial state mean
            nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=mu_w,        # known process noise mean
            nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=mu_v,        # known measurement noise mean
            nominal_Sigma_v=nominal_Sigma_v,
            theta_rs=robust_val,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[sim_idx_local]
        estimator._noise_index = 0
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
        if res is None:
            return None
        cost = compute_mpc_cost(res, Q_lqr, R_lqr, desired_traj)
        return res, cost

    def run_simulation_drkf_neurips(sim_idx_local):
        estimator = DRKF_neurips(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            theta=robust_val,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[sim_idx_local]
        estimator._noise_index = 0
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
        if res is None:
            return None
        cost = compute_mpc_cost(res, Q_lqr, R_lqr, desired_traj)
        return res, cost

    def run_simulation_risk_seek(sim_idx_local):
        estimator = RiskSeek(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=x0_mean,  # known initial state mean
            nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=mu_w,        # known process noise mean
            nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=mu_v,        # known measurement noise mean
            nominal_Sigma_v=nominal_Sigma_v,
            theta_rs=robust_val,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[sim_idx_local]
        estimator._noise_index = 0
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
        if res is None:
            return None
        cost = compute_mpc_cost(res, Q_lqr, R_lqr, desired_traj)
        return res, cost

    def run_simulation_mmse_baseline(sim_idx_local):
        # MMSE baseline using true distribution parameters (best possible estimator)
        estimator = KF(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            # Use TRUE parameters as nominal (perfect model knowledge)
            nominal_x0_mean=x0_mean, nominal_x0_cov=x0_cov,
            nominal_mu_w=mu_w, nominal_Sigma_w=Sigma_w,
            nominal_mu_v=mu_v, nominal_Sigma_v=Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[sim_idx_local]
        estimator._noise_index = 0
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
        if res is None:
            return None
        cost = compute_mpc_cost(res, Q_lqr, R_lqr, desired_traj)
        return res, cost

    
    # Filter execution mapping
    filter_functions = {
        'finite': run_simulation_finite,
        'inf': run_simulation_inf_kf,
        'drkf_inf': run_simulation_inf_drkf,
        'bcot': run_simulation_bcot,
        'drkf_finite': run_simulation_finite_drkf,
        'drkf_inf_cdc': run_simulation_inf_drkf_cdc,
        'drkf_finite_cdc': run_simulation_finite_drkf_cdc,
        'risk': run_simulation_risk,
        'risk_seek': run_simulation_risk_seek,
        'drkf_neurips': run_simulation_drkf_neurips
    }
    
    # Execute only selected filters
    filter_results = {}
    for filter_name in filters_to_execute:
        if filter_name in filter_functions:
            results = [filter_functions[filter_name](i) for i in range(num_sim)]
            # Filter out None results (failed experiments)
            valid_results = [r for r in results if r is not None and isinstance(r, tuple) and len(r) == 2]
            
            if len(valid_results) == 0:
                print(f"Warning: All experiments failed for filter '{filter_name}'. Skipping this filter.")
                continue
            elif len(valid_results) < len(results):
                print(f"Warning: {len(results) - len(valid_results)} out of {len(results)} experiments failed for filter '{filter_name}'.")
            
            filter_results[filter_name] = {
                'results': [r[0] for r in valid_results],  # Extract result part
                'costs': [r[1] for r in valid_results],    # Extract cost part
                'mse_mean': np.mean([np.mean(r[0]['mse']) for r in valid_results]),
                'cost_mean': np.mean([r[1] for r in valid_results]),
                'rep_state': valid_results[0][0]['state_traj'],
                'num_valid': len(valid_results),
                'num_total': len(results)
            }
    
    # MMSE baseline (always executed)
    results_mmse_baseline = [run_simulation_mmse_baseline(i) for i in range(num_sim)]
    valid_mmse_results = [r for r in results_mmse_baseline if r is not None and isinstance(r, tuple) and len(r) == 2]
    mse_mean_mmse_baseline = np.mean([np.mean(r[0]['mse']) for r in valid_mmse_results])
    cost_mean_mmse_baseline = np.mean([r[1] for r in valid_mmse_results])
    rep_state_mmse_baseline = valid_mmse_results[0][0]['state_traj']
    
    # Calculate regret (difference from MMSE baseline) for each executed filter
    overall_results = {
        'mmse_baseline': mse_mean_mmse_baseline,
        'mmse_baseline_cost': cost_mean_mmse_baseline,
        'mmse_baseline_state': rep_state_mmse_baseline,
    }
    
    for filter_name in filters_to_execute:
        if filter_name in filter_results:
            mse_mean = filter_results[filter_name]['mse_mean']
            cost_mean = filter_results[filter_name]['cost_mean']
            rep_state = filter_results[filter_name]['rep_state']
            regret = mse_mean - mse_mean_mmse_baseline
            cost_regret = cost_mean - cost_mean_mmse_baseline
            
            overall_results[filter_name] = mse_mean
            overall_results[f'{filter_name}_cost'] = cost_mean
            overall_results[f'{filter_name}_state'] = rep_state
            overall_results[f'{filter_name}_regret'] = regret
            overall_results[f'{filter_name}_cost_regret'] = cost_regret
    # Return raw results for executed filters
    raw_results = {'mmse_baseline': [r[0] for r in valid_mmse_results]}
    for filter_name in filters_to_execute:
        if filter_name in filter_results:
            raw_results[filter_name] = filter_results[filter_name]['results']
            raw_results[f'{filter_name}_costs'] = filter_results[filter_name]['costs']
            # Store metadata about failed experiments
            raw_results[f'{filter_name}_metadata'] = {
                'num_valid': filter_results[filter_name]['num_valid'],
                'num_total': filter_results[filter_name]['num_total'],
                'success_rate': filter_results[filter_name]['num_valid'] / filter_results[filter_name]['num_total']
            }
    
    return overall_results, raw_results

# --- Main Routine ---
def main(dist, num_sim, num_exp, T_total=10.0):
    seed_base = 2024
    
    # Generate desired trajectory
    desired_traj = generate_desired_trajectory(T_total)
    
    # Ensure global reproducibility
    np.random.seed(seed_base)
    if dist=='normal':
        robust_vals = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0, 10.0]
    elif dist=='quadratic':
        robust_vals = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Configurable filter execution list - modify this to enable/disable filters
    # Available filters: 'finite', 'inf', 'risk', 'risk_seek', 'drkf_neurips', 'bcot', 'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf'
    filters_to_execute = ['finite', 'inf', 'risk', 'risk_seek', 'drkf_neurips', 'bcot', 'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf']
    filters = filters_to_execute
    filter_labels = {
        'finite': "Time-varying KF",
        'inf': "Time-invariant KF",
        'risk': "Risk-Sensitive Filter",
        'risk_seek': "Risk-Seeking Filter",
        'drkf_neurips': "DRKF (NeurIPS)",
        'bcot': "DRKF (BCOT)",
        'drkf_finite_cdc': "DRKF (ours, finite, CDC)",
        'drkf_inf_cdc': "DRKF (ours, inf, CDC)",
        'drkf_finite': "DRKF (ours, finite)",
        'drkf_inf': "DRKF (ours, inf)"
    }
    
    all_results = {}
    raw_experiments_data = {}   # Store raw experiments for each robust candidate.
    failed_theta_filters = {}  # Track filters that failed for specific theta values
    
    for robust_val in robust_vals:
        # Check if any filters should be skipped for this theta value
        current_filters = filters_to_execute.copy()
        
        # Skip filters that failed for smaller theta values (only for risk-sensitive filters)
        if 'risk' in failed_theta_filters:
            min_failed_theta = min(failed_theta_filters['risk'])
            if robust_val >= min_failed_theta:
                print(f"Skipping risk-sensitive filter for θ={robust_val} (failed at θ={min_failed_theta})")
                current_filters = [f for f in current_filters if f != 'risk']
        
        if 'risk_seek' in failed_theta_filters:
            min_failed_theta = min(failed_theta_filters['risk_seek'])
            if robust_val >= min_failed_theta:
                print(f"Skipping risk-seeking filter for θ={robust_val} (failed at θ={min_failed_theta})")
                current_filters = [f for f in current_filters if f != 'risk_seek']
        
        if not current_filters:
            print(f"No filters to run for robust parameter = {robust_val}, skipping...")
            continue
            
        print(f"Running MPC experiments for robust parameter = {robust_val}")
        # Use deterministic parallel execution - each experiment has its own unique seed
        experiments = Parallel(n_jobs=-1, backend='loky')(
            delayed(run_experiment)(exp_idx, dist, num_sim, seed_base, robust_val, current_filters, desired_traj)
            for exp_idx in range(num_exp)
        )
        # Unpack overall results from the tuple returned by run_experiment.
        overall_experiments = [exp[0] for exp in experiments]
        all_mse = {key: [] for key in current_filters}
        all_regret = {key: [] for key in current_filters}
        failed_filters_this_theta = set()  # Track filters that failed for this theta
        mmse_baseline_values = []
        
        for exp in overall_experiments:
            # Collect MSE values for each filter
            for key in current_filters:
                if key in exp:  # Only process if filter was executed
                    all_mse[key].append(np.mean(exp[key]))
                    all_regret[key].append(exp[f'{key}_regret'])
                else:
                    # Filter was supposed to run but failed - record this failure
                    failed_filters_this_theta.add(key)
                    if key not in failed_theta_filters:
                        failed_theta_filters[key] = []
                    if robust_val not in failed_theta_filters[key]:
                        failed_theta_filters[key].append(robust_val)
                        print(f"Filter '{key}' failed at θ={robust_val}")
            # Collect MMSE baseline values
            mmse_baseline_values.append(exp['mmse_baseline'])
        
        # Exclude filters that had ANY failures for this theta value
        valid_filters = [key for key in current_filters if key not in failed_filters_this_theta and all_mse[key]]
        
        final_mse = {key: np.mean(all_mse[key]) for key in valid_filters}
        final_mse_std = {key: np.std(all_mse[key]) for key in valid_filters}
        final_regret = {key: np.mean(all_regret[key]) for key in valid_filters}
        final_regret_std = {key: np.std(all_regret[key]) for key in valid_filters}
        mmse_baseline_mean = np.mean(mmse_baseline_values)
        mmse_baseline_std = np.std(mmse_baseline_values)
        
        # Store representative state trajectories from the first experiment run for this robust value
        rep_state = {filt: overall_experiments[0][f"{filt}_state"] for filt in valid_filters if f"{filt}_state" in overall_experiments[0]}
        all_results[robust_val] = {
            'mse': final_mse,
            'mse_std': final_mse_std,
            'regret': final_regret,
            'regret_std': final_regret_std,
            'mmse_baseline': mmse_baseline_mean,
            'mmse_baseline_std': mmse_baseline_std,
            'state': rep_state
        }
        # Save the raw experiments for this candidate robust parameter.
        raw_experiments_data[robust_val] = [exp[1] for exp in experiments]
        print(f"Candidate robust parameter {robust_val}: Average MSE = {final_mse}")
    
    optimal_results = {}
    optimal_regret_results = {}
    
    for f in filters_to_execute:
        if f in ['finite', 'inf']:
            candidate = list(all_results.values())[0]
            optimal_results[f] = {
                'robust_val': "N/A",
                'mse': candidate['mse'][f],
                'mse_std': candidate['mse_std'][f],
                'regret': candidate['regret'][f],
                'regret_std': candidate['regret_std'][f]
            }
            optimal_regret_results[f] = optimal_results[f].copy()
        else:
            # Find best MSE
            best_val_mse = None
            best_mse = np.inf
            for robust_val, res in all_results.items():
                if f in res['mse']:  # Check if filter exists in results
                    current_mse = res['mse'][f]
                    if current_mse < best_mse:
                        best_mse = current_mse
                        best_val_mse = robust_val
            
            # Find best regret
            best_val_regret = None
            best_regret = np.inf
            for robust_val, res in all_results.items():
                if f in res['regret']:  # Check if filter exists in results
                    current_regret = res['regret'][f]
                    if current_regret < best_regret:
                        best_regret = current_regret
                        best_val_regret = robust_val
            
            # Only store results if we found valid data for this filter
            if best_val_mse is not None and best_val_regret is not None:
                optimal_results[f] = {
                    'robust_val': best_val_mse,
                    'mse': all_results[best_val_mse]['mse'][f],
                    'mse_std': all_results[best_val_mse]['mse_std'][f],
                    'regret': all_results[best_val_mse]['regret'][f],
                    'regret_std': all_results[best_val_mse]['regret_std'][f]
                }
                
                optimal_regret_results[f] = {
                    'robust_val': best_val_regret,
                    'mse': all_results[best_val_regret]['mse'][f],
                    'mse_std': all_results[best_val_regret]['mse_std'][f],
                    'regret': all_results[best_val_regret]['regret'][f],
                    'regret_std': all_results[best_val_regret]['regret_std'][f]
                }
            else:
                print(f"Warning: Filter '{f}' has no valid results across all θ values - skipping from optimal results.")
            
        if f in optimal_results:
            print(f"Optimal robust parameter for {f} (MSE): {optimal_results[f]['robust_val']}")
        else:
            print(f"Optimal robust parameter for {f} (MSE): N/A (no valid results)")
            
        if f in optimal_regret_results:
            print(f"Optimal robust parameter for {f} (Regret): {optimal_regret_results[f]['robust_val']}")
        else:
            print(f"Optimal robust parameter for {f} (Regret): N/A (no valid results)")
    
    sorted_optimal = sorted(optimal_results.items(), key=lambda item: item[1]['mse'])
    sorted_optimal_regret = sorted(optimal_regret_results.items(), key=lambda item: item[1]['regret'])
    
    print("\nSummary of Optimal Results (sorted by MSE):")
    for filt, info in sorted_optimal:
        print(f"{filt}: Optimal robust parameter = {info['robust_val']}, Average MSE = {info['mse']:.4f} ({info['mse_std']:.4f}), Regret = {info['regret']:.4f} ({info['regret_std']:.4f})")
    
    print("\nSummary of Optimal Results (sorted by Regret):")
    for filt, info in sorted_optimal_regret:
        print(f"{filt}: Optimal robust parameter = {info['robust_val']}, Average Regret = {info['regret']:.4f} ({info['regret_std']:.4f}), MSE = {info['mse']:.4f} ({info['mse_std']:.4f})")
    
    results_path = "./results/trajectory_tracking_MPC/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    save_data(os.path.join(results_path, f'overall_results_{dist}_trajectory_tracking.pkl'), all_results)
    save_data(os.path.join(results_path, f'optimal_results_{dist}_trajectory_tracking.pkl'), optimal_results)
    save_data(os.path.join(results_path, f'optimal_regret_results_{dist}_trajectory_tracking.pkl'), optimal_regret_results)
    # Save raw experiments data.
    save_data(os.path.join(results_path, f'raw_experiments_{dist}_trajectory_tracking.pkl'), raw_experiments_data)
    print("Trajectory tracking experiments with MPC controller completed for all robust parameters.")
    
    # --- Print Readable Data for the User ---
    print("\nDetailed Experiment Results (MSE-optimized):")
    header = "{:<50} {:<35} {:<35} {:<15}".format("Method", "Average MSE", "Average Regret", "Best theta")
    print(header)
    print("-"*135)
    for filt in filters_to_execute:
        if filt in optimal_results:
            best_theta = optimal_results[filt]['robust_val']
            mse = optimal_results[filt]['mse']
            mse_std = optimal_results[filt]['mse_std']
            regret = optimal_results[filt]['regret']
            regret_std = optimal_results[filt]['regret_std']
            mse_str = f"{mse:.3f} ({mse_std:.3f})"
            regret_str = f"{regret:.3f} ({regret_std:.3f})"
            print("{:<50} {:<35} {:<35} {:<15}".format(filter_labels[filt], mse_str, regret_str, best_theta))
        else:
            print("{:<50} {:<35} {:<35} {:<15}".format(filter_labels[filt], "N/A (no valid results)", "N/A (no valid results)", "N/A"))
    
    print("\nDetailed Experiment Results (Regret-optimized):")
    header = "{:<50} {:<35} {:<35} {:<15}".format("Method", "Average MSE", "Average Regret", "Best theta")
    print(header)
    print("-"*135)
    for filt in filters_to_execute:
        if filt in optimal_regret_results:
            best_theta_regret = optimal_regret_results[filt]['robust_val']
            mse = optimal_regret_results[filt]['mse']
            mse_std = optimal_regret_results[filt]['mse_std']
            regret = optimal_regret_results[filt]['regret']
            regret_std = optimal_regret_results[filt]['regret_std']
            mse_str = f"{mse:.3f} ({mse_std:.3f})"
            regret_str = f"{regret:.3f} ({regret_std:.3f})"
            print("{:<50} {:<35} {:<35} {:<15}".format(filter_labels[filt], mse_str, regret_str, best_theta_regret))
        else:
            print("{:<50} {:<35} {:<35} {:<15}".format(filter_labels[filt], "N/A (no valid results)", "N/A (no valid results)", "N/A"))
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default="normal", type=str,
                        help="Uncertainty distribution (normal or quadratic)")
    parser.add_argument('--num_sim', default=1, type=int,
                        help="Number of simulation runs per experiment")
    parser.add_argument('--num_exp', default=20, type=int,
                        help="Number of independent experiments")
    parser.add_argument('--T_total', default=10.0, type=float,
                        help="Total time horizon for trajectory tracking")
    args = parser.parse_args()
    main(args.dist, args.num_sim, args.num_exp, args.T_total)