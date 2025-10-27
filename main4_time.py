#!/usr/bin/env python3
"""
Computation time comparison experiment for DRKF methods.
"""

import numpy as np
import os
import pickle
import time
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

from LQR_with_estimator.DRKF_ours_inf import DRKF_ours_inf
from LQR_with_estimator.DRKF_ours_finite import DRKF_ours_finite
from LQR_with_estimator.DRKF_ours_inf_CDC import DRKF_ours_inf_CDC
from LQR_with_estimator.DRKF_ours_finite_CDC import DRKF_ours_finite_CDC
from LQR_with_estimator.BCOT import BCOT 
from LQR_with_estimator.DRKF_neurips import DRKF_neurips
from common_utils import (save_data, is_stabilizable, is_detectable, is_positive_definite,
                         enforce_positive_definiteness)

from LQR_with_estimator.base_filter import BaseFilter

_temp_A, _temp_C = np.eye(2), np.eye(2)
_temp_params = np.zeros((2, 1)), np.eye(2)
_sampler = BaseFilter(1, 'normal', 'normal', (_temp_A, _temp_C), np.eye(2, 1),
                     *_temp_params, *_temp_params, *_temp_params,
                     *_temp_params, *_temp_params, *_temp_params)

def normal(mu, Sigma, N=1):
    return _sampler.normal(mu, Sigma, N)

def generate_data(T, nx, ny, A, C, mu_w, Sigma_w, mu_v, M,
                  x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist,
                  x0_scale=None, w_scale=None, v_scale=None):
    x_true_all = np.zeros((T + 1, nx, 1))
    y_all = np.zeros((T, ny, 1))
    
    x_true = normal(x0_mean, x0_cov)
    x_true_all[0] = x_true
    
    for t in range(T):
        true_w = normal(mu_w, Sigma_w)
        true_v = normal(mu_v, M)
        y_t = C @ x_true + true_v
        y_all[t] = y_t
        x_true = A @ x_true + true_w
        x_true_all[t+1] = x_true
    return x_true_all, y_all

def setup_system_parameters():
    """Setup system parameters and matrices"""
    nx = 4
    ny = 2
    
    # System matrix A (4x4)
    A = np.array([
        [1.0,   0.2,  0.0,   0.0],
        [0.0,   0.2, 0.2,  0.0],
        [0.0,   0.0,  0.2,  0.2],
        [0.0,   0.0,  0.0,  -1.0]
    ])
    
    # Output matrix C (2x4)
    C = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    B = np.zeros((nx, 2))  # No control input
    
    # Noise parameters for normal distribution
    nw = 4; nu = 2; ny = 2
    mu_w = 0.0 * np.ones((nw, 1))
    Sigma_w = 0.01 * np.eye(nw)
    x0_mean = 0.0 * np.ones((nx, 1))
    x0_cov = 0.01 * np.eye(nx)
    mu_v = 0.0 * np.ones((ny, 1))
    Sigma_v = 0.01 * np.eye(ny)
    
    return {
        'A': A, 'C': C, 'B': B, 'nx': nx, 'ny': ny, 'nu': nu,
        'mu_w': mu_w, 'Sigma_w': Sigma_w, 'mu_v': mu_v, 'Sigma_v': Sigma_v,
        'x0_mean': x0_mean, 'x0_cov': x0_cov
    }

def estimate_nominal_covariances(params):
    """Estimate nominal covariances using EM algorithm"""
    A, C = params['A'], params['C']
    nx, ny = params['nx'], params['ny']
    mu_w, Sigma_w = params['mu_w'], params['Sigma_w']
    mu_v, Sigma_v = params['mu_v'], params['Sigma_v']
    x0_mean, x0_cov = params['x0_mean'], params['x0_cov']
    
    # Generate data for EM
    N_data = 10
    _, y_all_em = generate_data(N_data, nx, ny, A, C,
                                mu_w, Sigma_w, mu_v, Sigma_v,
                                x0_mean, x0_cov, None, None,
                                None, None, None, None, 'normal')
    y_all_em = y_all_em.squeeze()
    
    # EM algorithm
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
    
    return {
        'nominal_mu_w': mu_w,
        'nominal_Sigma_w': Sigma_w_hat.copy(),
        'nominal_mu_v': mu_v,
        'nominal_Sigma_v': Sigma_v_hat.copy(),
        'nominal_x0_mean': x0_mean,
        'nominal_x0_cov': Sigma_x0_hat.copy()
    }

def create_shared_noise_sequences(params, T, num_sim, seed_base):
    """Generate shared noise sequences for fair comparison"""
    shared_noise_sequences = []
    dist = 'normal'
    
    for sim_idx in range(num_sim):
        np.random.seed(seed_base + 2000 + sim_idx)
        
        x0_noise = normal(params['x0_mean'], params['x0_cov'])
        w_noise = [normal(params['mu_w'], params['Sigma_w']) for _ in range(T+1)]
        v_noise = [normal(params['mu_v'], params['Sigma_v']) for _ in range(T+1)]
            
        shared_noise_sequences.append({
            'x0': x0_noise,
            'w': w_noise,
            'v': v_noise
        })
    
    return shared_noise_sequences

def run_single_filter_simulation(filter_name, T, params, nominal_params, shared_noise_sequences, robust_val=1.0):
    """Run a single simulation for one filter and measure execution time"""
    dist = 'normal'
    system_data = (params['A'], params['C'])
    B = params['B']
    nx, ny, nu = params['nx'], params['ny'], params['nu']
    
    # Common parameters for all filters
    common_args = {
        'T': T, 'dist': dist, 'noise_dist': dist, 'system_data': system_data, 'B': B,
        'true_x0_mean': params['x0_mean'], 'true_x0_cov': params['x0_cov'],
        'true_mu_w': params['mu_w'], 'true_Sigma_w': params['Sigma_w'],
        'true_mu_v': params['mu_v'], 'true_Sigma_v': params['Sigma_v'],
        'nominal_x0_mean': nominal_params['nominal_x0_mean'],
        'nominal_x0_cov': nominal_params['nominal_x0_cov'],
        'nominal_mu_w': nominal_params['nominal_mu_w'],
        'nominal_Sigma_w': nominal_params['nominal_Sigma_w'],
        'nominal_mu_v': nominal_params['nominal_mu_v'],
        'nominal_Sigma_v': nominal_params['nominal_Sigma_v'],
        'x0_max': None, 'x0_min': None, 'w_max': None, 'w_min': None, 
        'v_max': None, 'v_min': None,
        'x0_scale': None, 'w_scale': None, 'v_scale': None
    }
    
    start_time = time.time()
    
    try:
        if filter_name == 'drkf_neurips':
            estimator = DRKF_neurips(**common_args, theta=robust_val)
        elif filter_name == 'bcot':
            estimator = BCOT(**common_args, radius=robust_val, maxit=20)
        elif filter_name == 'drkf_finite_cdc':
            estimator = DRKF_ours_finite_CDC(**common_args, theta_x=robust_val, theta_v=robust_val)
        elif filter_name == 'drkf_inf_cdc':
            estimator = DRKF_ours_inf_CDC(**common_args, theta_x=robust_val, theta_v=robust_val)
        elif filter_name == 'drkf_finite':
            estimator = DRKF_ours_finite(**common_args, theta_x=robust_val, theta_v=robust_val, theta_w=robust_val)
        elif filter_name == 'drkf_inf':
            estimator = DRKF_ours_inf(**common_args, theta_w=robust_val, theta_v=robust_val)
        else:
            raise ValueError(f"Unknown filter: {filter_name}")
        
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[0]  # Use first simulation
        estimator._noise_index = 0
        estimator.K_lqr = np.zeros((nu, nx))  # Dummy LQR gain (no control)
        
        result = estimator.forward()
        execution_time = time.time() - start_time
        
        return execution_time, True
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"Filter {filter_name} failed for T={T}: {str(e)}")
        return execution_time, False

def run_time_experiment():
    """Main experiment function"""
    print("Starting computation time comparison experiment...")
    
    # Setup
    params = setup_system_parameters()
    nominal_params = estimate_nominal_covariances(params)
    
    # Filter configuration
    filter_names = ['drkf_neurips', 'bcot', 'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf']
    filter_labels = {
        'drkf_neurips': "TV DRKF (NIPS2018)",
        'bcot': "TV DRKF (TAC2024)",
        'drkf_finite_cdc': "TV DRKF (CDC2025)",
        'drkf_inf_cdc': "SS DRKF (CDC2025)",
        'drkf_finite': "TV DRKF (ours)",
        'drkf_inf': "SS DRKF (ours)"
    }
    
    # Experiment parameters
    T_horizons = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Time horizons to test
    num_sim_per_T = 10  # Number of simulations per T value
    max_time_per_filter = 20.0 
    seed_base = 2024
    robust_val = 1.0
    
    # Storage for results
    execution_times = {name: {} for name in filter_names}
    filter_max_T = {name: 100 for name in filter_names}  # Track max T for each filter
    
    print(f"Testing time horizons: {T_horizons}")
    print(f"Timeout threshold: {max_time_per_filter} seconds per filter")
    
    for T in T_horizons:
        print(f"\nTesting T = {T}")
        
        # Generate shared noise sequences for this T
        shared_noise_sequences = create_shared_noise_sequences(params, T, num_sim_per_T, seed_base)
        
        for filter_name in filter_names:
            # Skip if this filter already exceeded timeout at smaller T
            if T > filter_max_T[filter_name]:
                print(f"  {filter_labels[filter_name]}: Skipped (exceeded timeout at T={filter_max_T[filter_name]})")
                continue
                
            print(f"  {filter_labels[filter_name]}: Running...", end=" ")
            
            # Run multiple simulations for this T and filter
            times_for_this_T = []
            all_successful = True
            
            for sim_idx in range(num_sim_per_T):
                exec_time, success = run_single_filter_simulation(
                    filter_name, T, params, nominal_params, shared_noise_sequences, robust_val
                )
                
                if success:
                    times_for_this_T.append(exec_time)
                else:
                    all_successful = False
                    break
                
                # Check if we exceeded the timeout
                if exec_time > max_time_per_filter:
                    print(f"TIMEOUT ({exec_time:.2f}s > {max_time_per_filter}s)")
                    filter_max_T[filter_name] = T
                    execution_times[filter_name][T] = times_for_this_T
                    all_successful = False
                    break
            
            if all_successful and times_for_this_T:
                avg_time = np.mean(times_for_this_T)
                execution_times[filter_name][T] = times_for_this_T
                print(f"OK ({avg_time:.3f}s avg)")
                
                # Check if average time is approaching timeout
                if avg_time > max_time_per_filter * 0.8:  # 80% of timeout
                    print(f"    Warning: Approaching timeout threshold")
            elif not all_successful:
                # Filter failed or timed out
                if times_for_this_T:  # Save partial results if any
                    execution_times[filter_name][T] = times_for_this_T
    
    return execution_times, filter_labels, filter_max_T

def save_and_summarize_results(execution_times, filter_labels, filter_max_T):
    """Save results and print summary"""
    
    # Create results directory
    results_path = "./results/time/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # Save raw execution time data
    save_data(os.path.join(results_path, 'execution_times_raw.pkl'), execution_times)
    save_data(os.path.join(results_path, 'filter_labels.pkl'), filter_labels)
    save_data(os.path.join(results_path, 'filter_max_T.pkl'), filter_max_T)
    
    # Calculate average execution times
    avg_execution_times = {}
    for filter_name in execution_times:
        avg_execution_times[filter_name] = {}
        for T in execution_times[filter_name]:
            times = execution_times[filter_name][T]
            if times:
                avg_execution_times[filter_name][T] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times)
                }
    
    save_data(os.path.join(results_path, 'avg_execution_times.pkl'), avg_execution_times)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPUTATION TIME COMPARISON RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nFilter Methods Tested:")
    for filter_name, label in filter_labels.items():
        max_T_tested = max(execution_times[filter_name].keys()) if execution_times[filter_name] else 0
        print(f"  • {label}: Tested up to T = {max_T_tested}")
    
    print(f"\nExecution Time Analysis:")
    print(f"{'Time Horizon (T)':<15} " + " ".join([f"{label[:15]:<20}" for label in filter_labels.values()]))
    print("-" * (15 + 20 * len(filter_labels)))
    
    # Find common T values tested by all filters
    all_T_values = set()
    for filter_name in execution_times:
        all_T_values.update(execution_times[filter_name].keys())
    
    for T in sorted(all_T_values):
        line = f"{T:<15} "
        for filter_name in filter_labels:
            if T in avg_execution_times[filter_name]:
                avg_time = avg_execution_times[filter_name][T]['mean']
                line += f"{avg_time:<20.4f}"
            else:
                line += f"{'N/A':<20}"
        print(line)
    
    print(f"\nTimeout Analysis (filters exceeding 30 seconds):")
    for filter_name, label in filter_labels.items():
        if filter_max_T[filter_name] < 100:
            print(f"  • {label}: Exceeded timeout at T = {filter_max_T[filter_name]}")
        else:
            print(f"  • {label}: Completed all time horizons (T = 2 to 100)")
    
    print(f"\nFastest Filter by Time Horizon:")
    for T in sorted(all_T_values)[:10]:  # Show first 10 T values
        valid_filters = [(name, avg_execution_times[name][T]['mean']) 
                        for name in filter_labels if T in avg_execution_times[name]]
        if valid_filters:
            fastest_filter, fastest_time = min(valid_filters, key=lambda x: x[1])
            print(f"  T = {T:3d}: {filter_labels[fastest_filter]} ({fastest_time:.4f}s)")
    
    print(f"\nOverall Performance Ranking (average across all T values):")
    overall_avg = {}
    for filter_name in filter_labels:
        if avg_execution_times[filter_name]:
            all_times = [stats['mean'] for stats in avg_execution_times[filter_name].values()]
            overall_avg[filter_name] = np.mean(all_times)
        else:
            overall_avg[filter_name] = float('inf')
    
    sorted_filters = sorted(overall_avg.items(), key=lambda x: x[1])
    for rank, (filter_name, avg_time) in enumerate(sorted_filters, 1):
        if avg_time != float('inf'):
            print(f"  {rank}. {filter_labels[filter_name]}: {avg_time:.4f}s average")
        else:
            print(f"  {rank}. {filter_labels[filter_name]}: Failed to complete")
    
    print("\n" + "="*80)
    print("Results saved to:", results_path)
    print("="*80)

def main():
    """Main function"""
    np.random.seed(2024)  # Global reproducibility
    
    execution_times, filter_labels, filter_max_T = run_time_experiment()
    save_and_summarize_results(execution_times, filter_labels, filter_max_T)
    
    print("\nComputation time comparison experiment completed successfully!")

if __name__ == "__main__":
    main()