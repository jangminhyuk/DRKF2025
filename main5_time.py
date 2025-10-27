#!/usr/bin/env python3
"""
Scalability experiment for computation time vs system dimension.
"""

import numpy as np
import os
import pickle
import time
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

from LQR_with_estimator.DRKF_ours_inf import DRKF_ours_inf
from common_utils import (save_data, is_stabilizable, is_detectable, is_positive_definite,
                         enforce_positive_definiteness)

from LQR_with_estimator.base_filter import BaseFilter

def normal(mu, Sigma, N=1, sampler=None):
    if sampler is None:
        return np.random.multivariate_normal(mu.flatten(), Sigma).reshape(-1, 1)
    return sampler.normal(mu, Sigma, N)

def generate_stable_system(n):
    """Generate a stable system of dimension n."""
    ny = n
    
    max_attempts = 100  # Prevent infinite loop
    for attempt in range(max_attempts):
        # Generate a random stable A matrix
        A = np.random.randn(n, n) * 0.3  # Scale to encourage stability
        # Ensure A is stable by scaling eigenvalues if necessary
        eigenvals = np.linalg.eigvals(A)
        max_eigval = np.max(np.abs(eigenvals))
        if max_eigval >= 1.0:
            A = 0.9 * A / max_eigval
        
        # Generate random C matrix (n×n)
        C = np.random.randn(n, n)
        
        # Check if (A, C) is detectable
        if is_detectable(A, C):
            return A, C
    
    # If we couldn't generate a detectable pair, fall back to a known detectable structure
    print(f"Warning: Could not generate detectable (A,C) pair for n={n}, using fallback")
    
    # Fallback: Create a stable A matrix and observable C
    A = 0.8 * np.eye(n) + 0.1 * np.random.randn(n, n)
    # Ensure stability
    eigenvals = np.linalg.eigvals(A)
    max_eigval = np.max(np.abs(eigenvals))
    if max_eigval >= 1.0:
        A = 0.9 * A / max_eigval
    
    # Create C matrix as identity plus small random perturbation
    C = np.eye(n) + 0.1 * np.random.randn(n, n)
    
    return A, C

def setup_system_parameters(n):
    """Setup system parameters for dimension n"""
    
    # Generate stable system matrices
    A, C = generate_stable_system(n)
    nx = n
    ny = C.shape[0]
    
    B = np.zeros((nx, 2))  # No control input
    
    # Noise parameters for normal distribution
    nu = 2
    mu_w = 0.0 * np.ones((nx, 1))
    Sigma_w = 0.01 * np.eye(nx)
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
    
    # Generate data for EM - use smaller dataset for efficiency
    N_data = 5
    x_true_all = np.zeros((N_data + 1, nx, 1))
    y_all = np.zeros((N_data, ny, 1))
    
    # Generate data
    x_true = normal(x0_mean, x0_cov)
    x_true_all[0] = x_true
    
    for t in range(N_data):
        true_w = normal(mu_w, Sigma_w)
        true_v = normal(mu_v, Sigma_v)
        y_t = C @ x_true + true_v
        y_all[t] = y_t
        x_true = A @ x_true + true_w
        x_true_all[t+1] = x_true
    
    y_all_em = y_all.squeeze()
    if y_all_em.ndim == 1:
        y_all_em = y_all_em.reshape(-1, 1)
    
    # EM algorithm with reduced iterations for efficiency
    mu_w_hat = np.zeros((nx, 1))
    mu_v_hat = np.zeros((ny, 1))
    mu_x0_hat = x0_mean.copy()
    Sigma_w_hat = np.eye(nx)
    Sigma_v_hat = np.eye(ny)
    Sigma_x0_hat = x0_cov.copy()
    
    try:
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
        
        max_iter = 20  # Reduced iterations for efficiency
        eps_log = 1e-3
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
        
    except Exception as e:
        print(f"EM algorithm failed for n={nx}, using default values: {e}")
        Sigma_w_hat = Sigma_w.copy()
        Sigma_v_hat = Sigma_v.copy()
        Sigma_x0_hat = x0_cov.copy()
    
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

def run_single_drkf_inf_simulation(n, T, params, nominal_params, shared_noise_sequences, robust_val=1.0):
    """Run a single simulation for drkf_inf and measure execution time"""
    dist = 'normal'
    system_data = (params['A'], params['C'])
    B = params['B']
    nx, ny, nu = params['nx'], params['ny'], params['nu']
    
    # Parameters for drkf_inf
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
        estimator = DRKF_ours_inf(**common_args, theta_w=robust_val, theta_v=robust_val)
        
        # Use shared noise sequences for fair comparison
        estimator.shared_noise_sequences = shared_noise_sequences[0]  # Use first simulation
        estimator._noise_index = 0
        estimator.K_lqr = np.zeros((nu, nx))  # Dummy LQR gain (no control)
        
        result = estimator.forward()
        execution_time = time.time() - start_time
        
        return execution_time, True
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"DRKF_inf failed for n={n}, T={T}: {str(e)}")
        return execution_time, False

def run_scalability_experiment():
    """Main experiment function for scalability testing"""
    print("Starting system dimension scalability experiment for drkf_inf...")
    
    # Experiment parameters
    T = 1  # Fixed time horizon , meaningless for this experiment
    dimension_range = [5, 10, 15, 20, 25, 30, 35, 40]  # System dimensions to test
    num_sim_per_n = 10  # Number of simulations per dimension
    max_time_per_dimension = 60.0  # 1 minute timeout per dimension
    seed_base = 2024
    robust_val = 1.0
    
    # Storage for results
    execution_times = {}
    dimension_max_n = max(dimension_range)
    
    print(f"Testing system dimensions: {dimension_range}")
    print(f"Fixed time horizon: T = {T}")
    print(f"Timeout threshold: {max_time_per_dimension} seconds per dimension")
    
    for n in dimension_range:
        print(f"\nTesting dimension n = {n}")
        
        try:
            # Generate system for this dimension
            np.random.seed(seed_base + n)  # Reproducible system generation
            params = setup_system_parameters(n)
            
            print(f"  System: {params['nx']}x{params['nx']} states, {params['ny']} outputs")
            print(f"  Estimating nominal parameters...", end=" ")
            nominal_params = estimate_nominal_covariances(params)
            print("OK")
            
            # Generate shared noise sequences for this dimension
            shared_noise_sequences = create_shared_noise_sequences(params, T, num_sim_per_n, seed_base)
            
            print(f"  Running drkf_inf simulations...", end=" ")
            
            # Run multiple simulations for this dimension
            times_for_this_n = []
            all_successful = True
            
            for sim_idx in range(num_sim_per_n):
                exec_time, success = run_single_drkf_inf_simulation(
                    n, T, params, nominal_params, shared_noise_sequences, robust_val
                )
                
                if success:
                    times_for_this_n.append(exec_time)
                    # Check if we exceeded the timeout
                    if exec_time > max_time_per_dimension:
                        print(f"TIMEOUT ({exec_time:.2f}s > {max_time_per_dimension}s)")
                        dimension_max_n = n
                        execution_times[n] = times_for_this_n
                        all_successful = False
                        break
                else:
                    all_successful = False
                    break
            
            if all_successful and times_for_this_n:
                avg_time = np.mean(times_for_this_n)
                execution_times[n] = times_for_this_n
                print(f"OK ({avg_time:.4f}s avg)")
                
                # Check if average time is approaching timeout
                if avg_time > max_time_per_dimension * 0.8:  # 80% of timeout
                    print(f"    Warning: Approaching timeout threshold")
            elif not all_successful:
                # Dimension failed or timed out
                if times_for_this_n:  # Save partial results if any
                    execution_times[n] = times_for_this_n
                break
                
        except Exception as e:
            print(f"FAILED: {str(e)}")
            break
    
    return execution_times, dimension_max_n

def save_and_summarize_results(execution_times, dimension_max_n):
    """Save results and print summary"""
    
    # Create results directory
    results_path = "./results/scalability/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # Save raw execution time data
    save_data(os.path.join(results_path, 'drkf_inf_scalability_raw.pkl'), execution_times)
    
    # Calculate average execution times
    avg_execution_times = {}
    for n in execution_times:
        times = execution_times[n]
        if times:
            avg_execution_times[n] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times)
            }
    
    save_data(os.path.join(results_path, 'drkf_inf_scalability_avg.pkl'), avg_execution_times)
    
    # Additional metadata
    metadata = {
        'T': 20,
        'filter': 'drkf_inf',
        'robust_val': 1.0,
        'dimensions_tested': list(execution_times.keys()),
        'dimension_max_n': dimension_max_n
    }
    save_data(os.path.join(results_path, 'drkf_inf_scalability_metadata.pkl'), metadata)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("DRKF_INF SYSTEM DIMENSION SCALABILITY RESULTS")
    print("="*80)
    
    print(f"\nExperiment Parameters:")
    print(f"  • Filter: Steady-State DRKF (ours)")
    print(f"  • Fixed time horizon: T = 20")
    print(f"  • Robust parameter: θ = 1.0")
    print(f"  • System dimensions tested: {sorted(execution_times.keys())}")
    
    print(f"\nExecution Time Results:")
    print(f"{'Dimension (n)':<15} {'Avg Time (s)':<15} {'Std Dev':<12} {'Min (s)':<10} {'Max (s)':<10} {'Samples':<8}")
    print("-" * 70)
    
    for n in sorted(avg_execution_times.keys()):
        stats = avg_execution_times[n]
        print(f"{n:<15} {stats['mean']:<15.4f} {stats['std']:<12.4f} {stats['min']:<10.4f} {stats['max']:<10.4f} {stats['count']:<8}")
    
    print(f"\nScalability Analysis:")
    dimensions = sorted(avg_execution_times.keys())
    times = [avg_execution_times[n]['mean'] for n in dimensions]
    
    if len(dimensions) >= 2:
        # Calculate growth rates
        for i in range(1, len(dimensions)):
            n_prev, n_curr = dimensions[i-1], dimensions[i]
            time_prev, time_curr = times[i-1], times[i]
            
            dim_ratio = n_curr / n_prev
            time_ratio = time_curr / time_prev
            
            print(f"  n={n_prev} to n={n_curr}: {dim_ratio:.1f}x dimension → {time_ratio:.2f}x time")
    
    # Estimate computational complexity
    if len(dimensions) >= 3:
        # Simple polynomial fit to log-log data
        log_dims = np.log(dimensions)
        log_times = np.log(times)
        coeffs = np.polyfit(log_dims, log_times, 1)
        complexity_exponent = coeffs[0]
        
        print(f"\nEstimated Computational Complexity:")
        print(f"  Time ∝ n^{complexity_exponent:.2f}")
        
        if complexity_exponent < 1.5:
            complexity_class = "Sub-quadratic (excellent scalability)"
        elif complexity_exponent < 2.5:
            complexity_class = "Approximately quadratic"
        elif complexity_exponent < 3.5:
            complexity_class = "Approximately cubic"
        else:
            complexity_class = "Higher-order polynomial (poor scalability)"
        
        print(f"  Classification: {complexity_class}")
    
    print(f"\nPractical Scalability Limits:")
    for n, stats in avg_execution_times.items():
        if stats['mean'] > 1.0:
            print(f"  n={n}: {stats['mean']:.2f}s (approaching practical limits for real-time)")
            break
    else:
        print(f"  All tested dimensions completed in < 1 second")
    
    if dimension_max_n < max(execution_times.keys()):
        print(f"  Maximum tested dimension: n={dimension_max_n} (timeout reached)")
    else:
        print(f"  All dimensions completed within timeout")
    
    print("\n" + "="*80)
    print("Results saved to:", results_path)
    print("="*80)

def main():
    """Main function"""
    np.random.seed(2024)  # Global reproducibility
    
    execution_times, dimension_max_n = run_scalability_experiment()
    save_and_summarize_results(execution_times, dimension_max_n)
    
    print("\nSystem dimension scalability experiment completed successfully!")

if __name__ == "__main__":
    main()