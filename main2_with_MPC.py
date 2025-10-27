#!/usr/bin/env python3
"""
Trajectory tracking experiments with MPC controller.
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

# Create a helper instance for distribution sampling - temporary instance just for sampling functions
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


# --- LQR Controller Computation ---
def compute_lqr_gain(A, B, Q_lqr, R_lqr):
    """Compute LQR gain using discrete-time algebraic Riccati equation"""
    P = solve_discrete_are(A, B, Q_lqr, R_lqr)
    K = np.linalg.inv(B.T @ P @ B + R_lqr) @ (B.T @ P @ A)
    return K

# --- MPC Cost Computation (using actual MPC inputs) ---
# Note: compute_mpc_cost is imported from common_utils

def load_optimal_parameters(results_path, dist):
    """Load optimal parameters from main5_1_with_MPC.py results"""
    optimal_file = os.path.join(results_path, f'optimal_results_{dist}_trajectory_tracking.pkl')
    if not os.path.exists(optimal_file):
        raise FileNotFoundError(f"Optimal results file not found: {optimal_file}\n"
                              f"Please run main5_1_with_MPC.py first to generate optimal parameters.")
    with open(optimal_file, 'rb') as f:
        optimal_results = pickle.load(f)
    return optimal_results

# --- True Data Generation ---
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
        x_true = A @ x_true + true_w  # No control input
        x_true_all[t+1] = x_true
    return x_true_all, y_all


# --- Single Filter Experiment Function for MPC Controller ---
def run_experiment(exp_idx, filter_name, optimal_theta, dist, num_sim, seed_base, desired_traj):
    """Run a single experiment with MPC controller for a specific filter"""
    np.random.seed(seed_base + exp_idx)
    
    dt = 0.2
    time_steps = int(desired_traj.shape[1])
    T = time_steps - 1  # simulation horizon
    
    # Use the initial desired state as the system's initial state.
    initial_trajectory = desired_traj[:, 0].reshape(-1, 1)
    
    # Discrete-time system dynamics (4D double integrator for MPC tracking)
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
    
    system_data = (A, C)
    
    nx = 4; nw = 4; nu = 2; ny = 2
    if dist == "normal":
        mu_w = 0.0 * np.ones((nw, 1))
        Sigma_w = 0.02 * np.eye(nw)
        x0_mean = 0.0 * np.ones((nx, 1))
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
    
    # --- Generate Data for EM Algorithm ---
    N_data = 5
    _, y_all_em = generate_data(N_data, nx, ny, A, C,
                                mu_w, Sigma_w, mu_v, Sigma_v,
                                x0_mean, x0_cov, x0_max, x0_min,
                                w_max, w_min, v_max, v_min, dist,
                                x0_scale, w_scale, v_scale)
    y_all_em = y_all_em.squeeze()
    
    # --- EM Estimation of Nominal Covariances ---
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
    
    # --- Compute LQR Gain (used as fallback in MPC implementation) ---
    Q_lqr = np.diag([10, 1, 10, 1])
    R_lqr = 0.1 * np.eye(2)
    K_lqr = compute_lqr_gain(A, B, Q_lqr, R_lqr)
    
    # Input saturation bounds for MPC trajectory tracking
    input_lower_bound = -1000.0
    input_upper_bound = 1000.0
    
    # --- Simulation Functions for the Specific Filter with MPC Controller ---
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
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
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
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
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
            theta_w=optimal_theta, theta_v=optimal_theta,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
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
            theta_x=optimal_theta, theta_v=optimal_theta, theta_w=optimal_theta,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
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
            radius=optimal_theta, maxit=20,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        estimator.K_lqr = K_lqr
        res = estimator.forward_track_MPC(desired_traj)
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
            theta_rs=optimal_theta,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        estimator.K_lqr = K_lqr
        res = estimator.forward_track(desired_traj)
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
            theta_x=optimal_theta, theta_v=optimal_theta,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        estimator.K_lqr = K_lqr
        res = estimator.forward_track(desired_traj)
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
            theta_x=optimal_theta, theta_v=optimal_theta,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        estimator.K_lqr = K_lqr
        res = estimator.forward_track(desired_traj)
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
            theta=optimal_theta,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        estimator.K_lqr = K_lqr
        res = estimator.forward_track(desired_traj)
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
            theta_rs=optimal_theta,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            input_lower_bound=input_lower_bound, input_upper_bound=input_upper_bound)
        estimator.K_lqr = K_lqr
        res = estimator.forward_track(desired_traj)
        if res is None:
            return None
        cost = compute_mpc_cost(res, Q_lqr, R_lqr, desired_traj)
        return res, cost

    # Run simulations for the specific filter (improved version with all filters)
    if filter_name == 'finite':
        results = [run_simulation_finite(i) for i in range(num_sim)]
    elif filter_name == 'inf':
        results = [run_simulation_inf_kf(i) for i in range(num_sim)]
    elif filter_name == 'drkf_inf':
        results = [run_simulation_inf_drkf(i) for i in range(num_sim)]
    elif filter_name == 'drkf_finite':
        results = [run_simulation_finite_drkf(i) for i in range(num_sim)]
    elif filter_name == 'bcot':
        results = [run_simulation_bcot(i) for i in range(num_sim)]
    elif filter_name == 'risk':
        results = [run_simulation_risk(i) for i in range(num_sim)]
    elif filter_name == 'risk_seek':
        results = [run_simulation_risk_seek(i) for i in range(num_sim)]
    elif filter_name == 'drkf_neurips':
        results = [run_simulation_drkf_neurips(i) for i in range(num_sim)]
    elif filter_name == 'drkf_inf_cdc':
        results = [run_simulation_inf_drkf_cdc(i) for i in range(num_sim)]
    elif filter_name == 'drkf_finite_cdc':
        results = [run_simulation_finite_drkf_cdc(i) for i in range(num_sim)]
    else:
        raise ValueError(f"Unknown filter name: {filter_name}")
    
    return results

# --- Main Routine ---
def main(dist, num_sim, num_exp, T_total=10.0):
    print(f"Running {num_exp} experiments with {num_sim} simulations each for {dist} distribution using MPC controller...")
    seed_base = 2024
    
    # Load optimal parameters from main5_1_with_MPC.py results
    mpc_results_path = "./results/trajectory_tracking_MPC/"
    optimal_results = load_optimal_parameters(mpc_results_path, dist)
    
    # Generate desired trajectory
    desired_traj = generate_desired_trajectory(T_total)
    
    # Define filters for MPC experiments
    filters = ['finite', 'inf', 'risk', 'risk_seek', 'drkf_neurips', 'bcot', 'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf']
    
    all_results = {}
    
    # Run experiments for each filter using MPC controller
    for filt in filters:
        print(f"Running {num_exp} experiments for {filt} filter...")
        
        # Get optimal parameter (N/A for non-robust filters)
        if filt in ['finite', 'inf']:
            optimal_theta = None  # These filters don't use robust parameters
        else:
            optimal_theta = optimal_results[filt]['robust_val']
            print(f"  Using optimal θ = {optimal_theta}")
        
        # Run experiments in parallel (each experiment contains num_sim simulations)
        experiments = Parallel(n_jobs=-1)(
            delayed(run_experiment)(exp_idx, filt, optimal_theta, dist, num_sim, seed_base, desired_traj)
            for exp_idx in range(num_exp)
        )
        
        # Flatten results from all experiments
        all_costs = []
        all_mse = []
        all_trajectories = []
        
        for exp_results in experiments:
            for result, cost in exp_results:
                all_costs.append(cost)
                all_mse.append(np.mean(result['mse']))
                all_trajectories.append(result)
        
        # Store results
        all_results[filt] = {
            'optimal_theta': optimal_theta,
            'results': all_trajectories,
            'costs': all_costs,
            'mse': all_mse
        }
        
        # Print summary
        mean_cost = np.mean(all_costs)
        std_cost = np.std(all_costs)
        mean_mse = np.mean(all_mse)
        std_mse = np.std(all_mse)
        
        if optimal_theta is not None:
            print(f"  Results: Cost = {mean_cost:.1f} ± {std_cost:.1f}, MSE = {mean_mse:.4f} ± {std_mse:.4f} (θ={optimal_theta})")
        else:
            print(f"  Results: Cost = {mean_cost:.1f} ± {std_cost:.1f}, MSE = {mean_mse:.4f} ± {std_mse:.4f}")
    
    # Save results
    if not os.path.exists(mpc_results_path):
        os.makedirs(mpc_results_path)
    
    save_file = os.path.join(mpc_results_path, f'main5_2_MPC_results_{dist}.pkl')
    save_data(save_file, all_results)
    
    print(f"\nResults saved to: {save_file}")
    print(f"Total simulations per filter: {num_exp * num_sim}")
    print("Use plot scripts to visualize the results.")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default="normal", type=str,
                        help="Uncertainty distribution (normal or quadratic)")
    parser.add_argument('--num_sim', default=1, type=int,
                        help="Number of simulation runs per experiment")
    parser.add_argument('--num_exp', default=200, type=int,
                        help="Number of independent experiments")
    parser.add_argument('--time', default=10.0, type=float,
                        help="Total simulation time")
    args = parser.parse_args()
    
    main(args.dist, args.num_sim, args.num_exp, args.time)