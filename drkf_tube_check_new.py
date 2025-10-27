#!/usr/bin/env python3
"""
DRKF Tube Check: Verify spectral bounds and KF sandwich property.
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.linalg import eigvals
from LQR_with_estimator.DRKF_ours_finite import DRKF_ours_finite
from LQR_with_estimator.KF import KF
from common_utils import is_positive_definite, enforce_positive_definiteness


def generate_random_pd_matrix(n, seed=None, scale=1.0):
    """Generate random positive definite matrix."""
    if seed is not None:
        np.random.seed(seed)
    
    A = np.random.randn(n, n)
    symmetric_matrix = scale * (A @ A.T)
    pd_matrix = enforce_positive_definiteness(symmetric_matrix)
    return pd_matrix


def compute_envelope_scalars(Sigma_hat, theta, nominal_matrix=None):
    """Compute envelope scalars for Bures-Wasserstein ball."""
    if nominal_matrix is not None:
        nominal_matrix = (nominal_matrix + nominal_matrix.T) / 2
        nominal_eigs = eigvals(nominal_matrix)
        lambda_min_nominal = np.min(np.real(nominal_eigs))
        lambda_max_nominal = np.max(np.real(nominal_eigs))
        
        underline_lambda = lambda_min_nominal
        overline_lambda = (np.sqrt(lambda_max_nominal) + theta)**2
        
    else:
        raise ValueError("nominal_matrix must be provided for correct envelope scalar computation")
    
    return underline_lambda, overline_lambda


class EnvelopeKF:
    """Envelope Kalman Filter using scalar noise covariances."""
    def __init__(self, A, C, Q_scalars, R_scalars, initial_prior_scalar):
        self.A = A
        self.C = C
        self.Q_scalars = Q_scalars
        self.R_scalars = R_scalars
        self.nx = A.shape[0]
        self.ny = C.shape[0]
        self.initial_prior_scalar = initial_prior_scalar
        
    def riccati_update(self, Sigma_prior, R):
        try:
            Sigma_prior_inv = np.linalg.inv(Sigma_prior)
            R_inv = np.linalg.inv(R)
            temp = Sigma_prior_inv + self.C.T @ R_inv @ self.C
            Sigma_posterior = np.linalg.inv(temp)
            return (Sigma_posterior + Sigma_posterior.T) / 2
        except np.linalg.LinAlgError:
            reg = 1e-12
            Sigma_prior_reg = Sigma_prior + reg * np.eye(self.nx)
            R_reg = R + reg * np.eye(self.ny)
            return self.riccati_update(Sigma_prior_reg, R_reg)
    
    def riccati_prediction(self, Sigma_posterior, Q):
        Sigma_prior_next = self.A @ Sigma_posterior @ self.A.T + Q
        return (Sigma_prior_next + Sigma_prior_next.T) / 2
    
    def run_trajectory(self, T):
        Sigma_prior_traj = []
        Sigma_posterior_traj = []
        
        for t in range(T + 1):
            if t == 0:
                Sigma_prior = self.initial_prior_scalar * np.eye(self.nx)
                if t < len(self.R_scalars):
                    R_t = self.R_scalars[t] * np.eye(self.ny)
                    Sigma_posterior = self.riccati_update(Sigma_prior, R_t)
                else:
                    Sigma_posterior = Sigma_prior.copy()
            else:
                Q_t = self.Q_scalars[t-1] * np.eye(self.nx) if (t-1) < len(self.Q_scalars) else self.Q_scalars[-1] * np.eye(self.nx)
                Sigma_prior = self.riccati_prediction(Sigma_posterior_prev, Q_t)
                if t < len(self.R_scalars):
                    R_t = self.R_scalars[t] * np.eye(self.ny)
                    Sigma_posterior = self.riccati_update(Sigma_prior, R_t)
                else:
                    Sigma_posterior = Sigma_prior.copy()
            
            Sigma_prior_traj.append(Sigma_prior.copy())
            Sigma_posterior_traj.append(Sigma_posterior.copy())
            Sigma_posterior_prev = Sigma_posterior.copy()
        
        return Sigma_prior_traj, Sigma_posterior_traj


def run_envelope_kf(A, C, initial_prior_scalar, Q_scalars, R_scalars, T):
    """Run Kalman filter with envelope covariances."""
    envelope_kf = EnvelopeKF(A, C, Q_scalars, R_scalars, initial_prior_scalar)
    return envelope_kf.run_trajectory(T)


def check_spectral_bounds(Sigma_star, underline_lambda, overline_lambda, label=""):
    """Check if matrix eigenvalues satisfy spectral bounds."""
    Sigma_star = (Sigma_star + Sigma_star.T) / 2
    
    eigs = eigvals(Sigma_star)
    lambda_min = np.min(np.real(eigs))
    lambda_max = np.max(np.real(eigs))
    
    bounds_satisfied = (lambda_min >= underline_lambda - 1e-10) and (lambda_max <= overline_lambda + 1e-10)
    
    return bounds_satisfied, lambda_min, lambda_max


def check_loewner_order(Sigma1, Sigma2, label="", tolerance=1e-8):
    """Check if Sigma1 ⪯ Sigma2 (Loewner order)."""
    diff = Sigma2 - Sigma1
    diff = (diff + diff.T) / 2
    
    eigs = eigvals(diff)
    min_eig = np.min(np.real(eigs))
    
    is_psd = min_eig >= -tolerance
    
    return is_psd, min_eig


def drkf_spectral_verification(A, C, nominal_Sigma_w_traj, nominal_Sigma_v_traj, 
                              theta_w, theta_v, nominal_x0_prior, T,
                              theta_x_0=None, system_params=None, verbose=True):
    
    if verbose:
        print("=== DRKF Spectral Boundedness Verification ===")
        print(f"Time horizon: T = {T}")
        print(f"Wasserstein radii: θ_w = {theta_w}, θ_v = {theta_v}")
        print()
    
    # Step 1: Compute envelope scalars
    if verbose:
        print("Step 1: Computing envelope scalars...")
    
    Q_scalars = []
    Q_bar_scalars = []
    R_scalars = []
    R_bar_scalars = []
    
    for t in range(T + 1):
        # For process noise bounds, use time-specific nominal covariance
        nominal_Sigma_w_t = nominal_Sigma_w_traj[t] if t < len(nominal_Sigma_w_traj) else nominal_Sigma_w_traj[-1]
        Q_t, Q_bar_t = compute_envelope_scalars(None, theta_w, nominal_Sigma_w_t)
            
        # For measurement noise bounds, use time-specific nominal covariance
        nominal_Sigma_v_t = nominal_Sigma_v_traj[t] if t < len(nominal_Sigma_v_traj) else nominal_Sigma_v_traj[-1]
        R_t, R_bar_t = compute_envelope_scalars(None, theta_v, nominal_Sigma_v_t)
            
        Q_scalars.append(Q_t)
        Q_bar_scalars.append(Q_bar_t)
        R_scalars.append(R_t)
        R_bar_scalars.append(R_bar_t)
        
        if verbose:
            print(f"  t={t}: Q_t={Q_t:.6f}, Q̄_t={Q_bar_t:.6f}, R_t={R_t:.6f}, R̄_t={R_bar_t:.6f}")
    
    # Step 2: Compute envelope scalars for initial state covariance
    if verbose:
        print("\nStep 2: Computing initial state envelope scalars...")
    
    # Use the same initial prior matrix for computing envelope bounds
    shared_initial_prior = nominal_x0_prior.copy()
    
    # Compute envelope scalars for initial state covariance using theta_x_0
    if theta_x_0 is None:
        theta_x_0 = theta_w  # Default to theta_w if not provided
    
    # Compute envelope scalars for initial state: use nominal_x0_prior as the nominal matrix
    initial_low_scalar, initial_high_scalar = compute_envelope_scalars(None, theta_x_0, shared_initial_prior)
    
    if verbose:
        print(f"  Initial state scalars: Low={initial_low_scalar:.6f}, High={initial_high_scalar:.6f} (θ_x_0={theta_x_0:.3f})")
    
    # Step 3: Run LOW and HIGH KFs
    if verbose:
        print("\nStep 3: Running LOW and HIGH KFs...")
    
    Sigma_low_prior, Sigma_low_posterior = run_envelope_kf(A, C, initial_low_scalar, Q_scalars, R_scalars, T)
    Sigma_high_prior, Sigma_high_posterior = run_envelope_kf(A, C, initial_high_scalar, Q_bar_scalars, R_bar_scalars, T)
    
    if verbose:
        print(f"  LOW KF: {len(Sigma_low_prior)} prior, {len(Sigma_low_posterior)} posterior covariances")
        print(f"  HIGH KF: {len(Sigma_high_prior)} prior, {len(Sigma_high_posterior)} posterior covariances")
    
    # Step 4: Run DRKF and capture least-favorable covariances
    if verbose:
        print("\nStep 4: Running DRKF...")
    
    # Run DRKF with modified implementation to capture covariances
    drkf_results = run_drkf_with_covariance_capture(A, C, nominal_Sigma_w_traj, nominal_Sigma_v_traj, 
                                                   theta_w, theta_v, shared_initial_prior, T, 
                                                   theta_x_0, system_params)
    
    # Step 5: Verify spectral bounds
    if verbose:
        print("\nStep 5: Verifying spectral bounds...")
    
    spectral_results = verify_spectral_bounds(drkf_results, Q_scalars, Q_bar_scalars, 
                                            R_scalars, R_bar_scalars, verbose)
    
    # Step 6: Verify KF sandwich property
    if verbose:
        print("\nStep 6: Verifying KF sandwich...")
    
    sandwich_results = verify_kf_sandwich(drkf_results, Sigma_low_prior, Sigma_low_posterior,
                                        Sigma_high_prior, Sigma_high_posterior, verbose)
    
    verification_results = {
        'envelope_scalars': {
            'Q_scalars': Q_scalars,
            'Q_bar_scalars': Q_bar_scalars,
            'R_scalars': R_scalars,
            'R_bar_scalars': R_bar_scalars
        },
        'low_kf': {
            'Sigma_prior': Sigma_low_prior,
            'Sigma_posterior': Sigma_low_posterior
        },
        'high_kf': {
            'Sigma_prior': Sigma_high_prior,
            'Sigma_posterior': Sigma_high_posterior
        },
        'drkf': drkf_results,
        'spectral_bounds_check': spectral_results,
        'sandwich_check': sandwich_results
    }
    
    # Step 7: Print summary
    if verbose:
        print_verification_summary(spectral_results, sandwich_results)
    
    return verification_results


class DRKFVerification:
    """
    DRKF implementation for verification that delegates to existing DRKF_ours_finite.
    """
    
    def __init__(self, A, C, nominal_Sigma_w_traj, nominal_Sigma_v_traj, 
                 theta_w, theta_v, nominal_x0_prior, T, theta_x_0=None):
        self.A = A
        self.C = C
        self.nx, self.ny = A.shape[0], C.shape[0]
        self.nominal_Sigma_w_traj = nominal_Sigma_w_traj
        self.nominal_Sigma_v_traj = nominal_Sigma_v_traj
        self.theta_w = theta_w
        self.theta_v = theta_v
        self.nominal_x0_prior = nominal_x0_prior
        self.theta_x_0 = theta_x_0 if theta_x_0 is not None else theta_w  # Default to theta_w if not provided
        self.T = T
        
        # Create DRKF instance for SDP solving
        self.drkf_instance = self._create_drkf_instance()
    
    def _create_drkf_instance(self):
        """Create a DRKF_ours_finite instance for SDP solving."""
        # Create dummy parameters for DRKF initialization
        dummy_B = np.zeros((self.nx, 1))
        dummy_mean = np.zeros((self.nx, 1))
        dummy_v_mean = np.zeros((self.ny, 1))
        
        drkf = DRKF_ours_finite(
            T=self.T, dist='normal', noise_dist='normal',
            system_data=(self.A, self.C), B=dummy_B,
            true_x0_mean=dummy_mean, true_x0_cov=self.nominal_x0_prior,
            true_mu_w=dummy_mean, true_Sigma_w=self.nominal_Sigma_w_traj[0],
            true_mu_v=dummy_v_mean, true_Sigma_v=self.nominal_Sigma_v_traj[0],
            nominal_x0_mean=dummy_mean, nominal_x0_cov=self.nominal_x0_prior,
            nominal_mu_w=dummy_mean, nominal_Sigma_w=self.nominal_Sigma_w_traj[0],
            nominal_mu_v=dummy_v_mean, nominal_Sigma_v=self.nominal_Sigma_v_traj[0],
            theta_x=self.theta_x_0, theta_v=self.theta_v, theta_w=self.theta_w
        )
        return drkf
    
    def run_covariance_capture(self):
        """Run DRKF and capture covariances using existing implementation."""
        # Storage for worst-case covariances
        wc_Sigma_v = np.zeros((self.T+1, self.ny, self.ny))
        wc_Sigma_w = np.zeros((self.T+1, self.nx, self.nx))
        wc_Xprior = np.zeros((self.T+1, self.nx, self.nx))
        wc_Xpost = np.zeros((self.T+1, self.nx, self.nx))
        
        # Start from initial prior covariance
        wc_Xprior_initial = self.nominal_x0_prior.copy()
        
        for t in range(self.T + 1):
            # Get nominal covariances for this time step
            nominal_Sigma_w_t = (self.nominal_Sigma_w_traj[t] if t < len(self.nominal_Sigma_w_traj) 
                               else self.nominal_Sigma_w_traj[-1])
            nominal_Sigma_v_t = (self.nominal_Sigma_v_traj[t] if t < len(self.nominal_Sigma_v_traj) 
                               else self.nominal_Sigma_v_traj[-1])
            
            if t == 0:
                # For t=0, solve SDP for initial stage with prior covariance
                self.drkf_instance.nominal_Sigma_v = nominal_Sigma_v_t
                result_initial = self.drkf_instance.solve_sdp_initial(wc_Xprior_initial)
                wc_Sigma_v[t], wc_Xprior[t], wc_Xpost[t] = result_initial
                wc_Xpost_prev = wc_Xpost[t].copy()
            else:
                # Use DRKF's SDP solver
                self.drkf_instance.nominal_Sigma_w = nominal_Sigma_w_t
                self.drkf_instance.nominal_Sigma_v = nominal_Sigma_v_t
                
                # Solve SDP using existing implementation
                result = self.drkf_instance.solve_sdp(wc_Xpost_prev)
                wc_Sigma_v[t], wc_Sigma_w[t-1], wc_Xprior[t], wc_Xpost[t] = result
                wc_Xpost_prev = wc_Xpost[t].copy()
        
        return {
            'Sigma_w_star': [wc_Sigma_w[t] for t in range(self.T+1)],
            'Sigma_v_star': [wc_Sigma_v[t] for t in range(self.T+1)],
            'Sigma_x_prior': [wc_Xprior[t] for t in range(self.T+1)],
            'Sigma_x_posterior': [wc_Xpost[t] for t in range(self.T+1)]
        }


def run_drkf_with_covariance_capture(A, C, nominal_Sigma_w_traj, nominal_Sigma_v_traj,
                                    theta_w, theta_v, nominal_x0_prior, T, 
                                    theta_x_0=None, system_params=None):
    drkf_verifier = DRKFVerification(A, C, nominal_Sigma_w_traj, nominal_Sigma_v_traj,
                                   theta_w, theta_v, nominal_x0_prior, T, theta_x_0)
    return drkf_verifier.run_covariance_capture()


def verify_spectral_bounds(drkf_results, Q_scalars, Q_bar_scalars, 
                          R_scalars, R_bar_scalars, verbose=True):
    """
    Verify spectral boundedness of least-favorable covariances.
    """
    results = {
        'process_noise_bounds': [],
        'measurement_noise_bounds': [],
        'all_bounds_satisfied': True
    }
    
    T = len(drkf_results['Sigma_w_star']) - 1
    
    for t in range(T + 1):
        Sigma_w_star = drkf_results['Sigma_w_star'][t]
        Sigma_v_star = drkf_results['Sigma_v_star'][t]
        
        # Check process noise bounds (skip if Sigma_w is all zeros at final time step)
        # This happens because SDP obtains Sigma_w[t-1], so at t=T there's no process noise
        w_all_zeros = np.allclose(Sigma_w_star, 0)
        if t == T and w_all_zeros:
            # At final time step, process noise is not used, so mark as satisfied
            w_satisfied = True
            w_min = w_max = 0.0
        else:
            w_satisfied, w_min, w_max = check_spectral_bounds(
                Sigma_w_star, Q_scalars[t], Q_bar_scalars[t], f"w_t={t}"
            )
        
        # Check measurement noise bounds  
        v_satisfied, v_min, v_max = check_spectral_bounds(
            Sigma_v_star, R_scalars[t], R_bar_scalars[t], f"v_t={t}"
        )
        
        results['process_noise_bounds'].append({
            't': t,
            'satisfied': w_satisfied,
            'lambda_min': w_min,
            'lambda_max': w_max,
            'Q_t': Q_scalars[t],
            'Q_bar_t': Q_bar_scalars[t],
            'is_final_step_zero': (t == T and w_all_zeros)
        })
        
        results['measurement_noise_bounds'].append({
            't': t,
            'satisfied': v_satisfied,
            'lambda_min': v_min,
            'lambda_max': v_max,
            'R_t': R_scalars[t],
            'R_bar_t': R_bar_scalars[t]
        })
        
        if not w_satisfied or not v_satisfied:
            results['all_bounds_satisfied'] = False
        
        if verbose:
            w_status = "✓" if w_satisfied else "✗"
            v_status = "✓" if v_satisfied else "✗"
            
            if t == T and w_all_zeros:
                print(f"  t={t}: Process {w_status} [0.000000, 0.000000] (final step, no process noise)")
            else:
                print(f"  t={t}: Process {w_status} [{w_min:.6f}, {w_max:.6f}] ⊆ [{Q_scalars[t]:.6f}, {Q_bar_scalars[t]:.6f}]")
            print(f"       Measurement {v_status} [{v_min:.6f}, {v_max:.6f}] ⊆ [{R_scalars[t]:.6f}, {R_bar_scalars[t]:.6f}]")
    
    return results


def verify_kf_sandwich(drkf_results, Sigma_low_prior, Sigma_low_posterior,
                      Sigma_high_prior, Sigma_high_posterior, verbose=True):
    """
    Verify KF sandwich property (Loewner order).
    
    Should satisfy LOW ⪯ DRKF ⪯ HIGH (Loewner order) for all time steps t≥0.
    At t=0, envelope KFs use different initial prior covariances based on envelope bounds.
    """
    results = {
        'prior_sandwich': [],
        'posterior_sandwich': [],
        'all_sandwich_satisfied': True
    }
    
    T = len(drkf_results['Sigma_x_prior']) - 1
    
    for t in range(T + 1):
        # Check prior covariances: Σ_LOW_prior ⪯ Σ_DRKF_prior ⪯ Σ_HIGH_prior
        Sigma_drkf_prior = drkf_results['Sigma_x_prior'][t]
        
        low_order, low_min_eig = check_loewner_order(
            Sigma_low_prior[t], Sigma_drkf_prior, f"LOW ⪯ DRKF prior t={t}"
        )
        
        high_order, high_min_eig = check_loewner_order(
            Sigma_drkf_prior, Sigma_high_prior[t], f"DRKF ⪯ HIGH prior t={t}"
        )
        
        prior_satisfied = low_order and high_order
        
        # Check posterior covariances: Σ_LOW_posterior ⪯ Σ_DRKF_posterior ⪯ Σ_HIGH_posterior
        if t < len(drkf_results['Sigma_x_posterior']):
            Sigma_drkf_posterior = drkf_results['Sigma_x_posterior'][t]
            
            # Standard Loewner order check for all time steps (including t=0)
            low_post_order, low_post_min_eig = check_loewner_order(
                Sigma_low_posterior[t], Sigma_drkf_posterior, f"LOW ⪯ DRKF posterior t={t}"
            )
            
            high_post_order, high_post_min_eig = check_loewner_order(
                Sigma_drkf_posterior, Sigma_high_posterior[t], f"DRKF ⪯ HIGH posterior t={t}"
            )
            
            post_satisfied = low_post_order and high_post_order
        else:
            post_satisfied = True
            low_post_min_eig = high_post_min_eig = 0.0
        
        results['prior_sandwich'].append({
            't': t,
            'satisfied': prior_satisfied,
            'low_min_eig': low_min_eig,
            'high_min_eig': high_min_eig
        })
        
        results['posterior_sandwich'].append({
            't': t,
            'satisfied': post_satisfied,
            'low_min_eig': low_post_min_eig,
            'high_min_eig': high_post_min_eig
        })
        
        if not prior_satisfied or not post_satisfied:
            results['all_sandwich_satisfied'] = False
        
        if verbose:
            prior_status = "✓" if prior_satisfied else "✗"
            post_status = "✓" if post_satisfied else "✗"
            
            # Standard output for all time steps
            print(f"  t={t}: Prior {prior_status} (difference eigenvalues: {low_min_eig:.2e}, {high_min_eig:.2e})")
            print(f"       Posterior {post_status} (difference eigenvalues: {low_post_min_eig:.2e}, {high_post_min_eig:.2e})")
    
    return results


def print_verification_summary(spectral_results, sandwich_results):
    """Print verification summary."""
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)
    
    # Spectral bounds summary
    spec_status = "PASS" if spectral_results['all_bounds_satisfied'] else "FAIL"
    print(f"Spectral Boundedness: {spec_status}")
    
    if not spectral_results['all_bounds_satisfied']:
        print("  Failed time steps:")
        for t, result in enumerate(spectral_results['process_noise_bounds']):
            if not result['satisfied']:
                print(f"    t={t}: Process noise bounds violated")
        for t, result in enumerate(spectral_results['measurement_noise_bounds']):
            if not result['satisfied']:
                print(f"    t={t}: Measurement noise bounds violated")
    
    # Sandwich property summary
    sandwich_status = "PASS" if sandwich_results['all_sandwich_satisfied'] else "FAIL"
    print(f"KF Sandwich Property: {sandwich_status}")
    
    if not sandwich_results['all_sandwich_satisfied']:
        print("  Failed time steps:")
        for t, result in enumerate(sandwich_results['prior_sandwich']):
            if not result['satisfied']:
                print(f"    t={t}: Prior sandwich violated")
        for t, result in enumerate(sandwich_results['posterior_sandwich']):
            if not result['satisfied']:
                print(f"    t={t}: Posterior sandwich violated")
    
    overall_status = "PASS" if (spectral_results['all_bounds_satisfied'] and 
                               sandwich_results['all_sandwich_satisfied']) else "FAIL"
    print(f"\nOVERALL VERIFICATION: {overall_status}")
    print("="*50)


def generate_random_system(nx, ny, seed=None):
    """Generate a random system with proper dimensions."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate completely random A matrix
    A = np.random.randn(nx, nx)
    
    # Generate random C matrix
    C = np.random.randn(ny, nx)
    
    return A, C


def run_multiple_experiments(num_exp=20, T=5, verbose_individual=False):
    """Run multiple experiments with random systems and summarize results."""
    print("DRKF Tube Check: Multiple Random Experiments")
    print("=" * 50)
    print(f"Running {num_exp} experiments with T={T}")
    print()
    
    results_summary = {
        'total_experiments': num_exp,
        'spectral_success': 0,
        'sandwich_success': 0,
        'overall_success': 0,
        'failed_experiments': []
    }
    
    for exp in range(num_exp):
        if verbose_individual:
            print(f"\n{'='*20} Experiment {exp+1}/{num_exp} {'='*20}")
        else:
            print(f"Running experiment {exp+1}/{num_exp}...", end=" ")
        
        # Generate random system
        nx, ny = np.random.choice([2, 3, 4]), np.random.choice([2, 3])
        A, C = generate_random_system(nx, ny, seed=100+exp)
        
        # Generate random positive definite nominal covariances
        nominal_Sigma_w = generate_random_pd_matrix(nx, seed=200+exp, scale=np.random.uniform(0.05, 0.2))
        nominal_Sigma_v = generate_random_pd_matrix(ny, seed=300+exp, scale=np.random.uniform(0.02, 0.1))
        nominal_x0_prior = generate_random_pd_matrix(nx, seed=400+exp, scale=np.random.uniform(0.05, 0.15))
        
        # Random Wasserstein radii
        theta_w = np.random.uniform(0.1, 0.3)
        theta_v = np.random.uniform(0.05, 0.2)
        theta_x_0 = np.random.uniform(0.05, 0.25)  # Initial stage Wasserstein radius
        
        # Create time-varying trajectories (constant in this example)
        nominal_Sigma_w_traj = [nominal_Sigma_w] * (T + 1)
        nominal_Sigma_v_traj = [nominal_Sigma_v] * (T + 1)
        
        if verbose_individual:
            print(f"System: nx={nx}, ny={ny}")
            print(f"Wasserstein radii: θ_w={theta_w:.3f}, θ_v={theta_v:.3f}, θ_x_0={theta_x_0:.3f}")
            print(f"A eigenvalues: {np.linalg.eigvals(A)}")
        
        try:
            # Run verification
            results = drkf_spectral_verification(
                A, C, nominal_Sigma_w_traj, nominal_Sigma_v_traj,
                theta_w, theta_v, nominal_x0_prior, T,
                theta_x_0=theta_x_0, verbose=verbose_individual
            )
            
            # Check results
            spectral_pass = results['spectral_bounds_check']['all_bounds_satisfied']
            sandwich_pass = results['sandwich_check']['all_sandwich_satisfied']
            overall_pass = spectral_pass and sandwich_pass
            
            if spectral_pass:
                results_summary['spectral_success'] += 1
            if sandwich_pass:
                results_summary['sandwich_success'] += 1
            if overall_pass:
                results_summary['overall_success'] += 1
            else:
                results_summary['failed_experiments'].append({
                    'exp_num': exp + 1,
                    'nx': nx, 'ny': ny,
                    'theta_w': theta_w, 'theta_v': theta_v, 'theta_x_0': theta_x_0,
                    'spectral_pass': spectral_pass,
                    'sandwich_pass': sandwich_pass
                })
            
            if not verbose_individual:
                status = "PASS" if overall_pass else "FAIL"
                print(f"{status} (nx={nx}, ny={ny})")
            
        except Exception as e:
            print(f"ERROR in experiment {exp+1}: {str(e)}")
            results_summary['failed_experiments'].append({
                'exp_num': exp + 1,
                'nx': nx, 'ny': ny,
                'theta_w': theta_w, 'theta_v': theta_v, 'theta_x_0': theta_x_0,
                'error': str(e)
            })
    
    return results_summary


def print_experiment_summary(results_summary):
    """Print a comprehensive summary of all experiments."""
    print("\n" + "="*60)
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print("="*60)
    
    total = results_summary['total_experiments']
    spectral_success = results_summary['spectral_success']
    sandwich_success = results_summary['sandwich_success']
    overall_success = results_summary['overall_success']
    
    print(f"Total Experiments: {total}")
    print(f"Spectral Boundedness Success: {spectral_success}/{total} ({100*spectral_success/total:.1f}%)")
    print(f"KF Sandwich Success: {sandwich_success}/{total} ({100*sandwich_success/total:.1f}%)")
    print(f"Overall Success: {overall_success}/{total} ({100*overall_success/total:.1f}%)")
    
    if results_summary['failed_experiments']:
        print(f"\nFailed Experiments ({len(results_summary['failed_experiments'])}):")
        print("-" * 40)
        for failure in results_summary['failed_experiments']:
            exp_num = failure['exp_num']
            if 'error' in failure:
                print(f"  Exp {exp_num}: ERROR - {failure['error']}")
            else:
                nx, ny = failure['nx'], failure['ny']
                theta_w, theta_v, theta_x_0 = failure['theta_w'], failure['theta_v'], failure['theta_x_0']
                spectral = "✓" if failure['spectral_pass'] else "✗"
                sandwich = "✓" if failure['sandwich_pass'] else "✗"
                print(f"  Exp {exp_num}: nx={nx}, ny={ny}, θ_w={theta_w:.3f}, θ_v={theta_v:.3f}, θ_x_0={theta_x_0:.3f}")
                print(f"           Spectral: {spectral}, Sandwich: {sandwich}")
    
    print("\n" + "="*60)
    
    # Interpretation
    if overall_success == total:
        print("EXCELLENT: All experiments passed! DRKF theory is well-validated.")
    elif overall_success >= 0.9 * total:
        print("GOOD: Most experiments passed. Minor numerical issues may exist.")
    elif overall_success >= 0.7 * total:
        print("MODERATE: Some experiments failed. Investigation recommended.")
    else:
        print("CONCERNING: Many experiments failed. Theory or implementation issues.")


if __name__ == "__main__":
    # Set global random seed for reproducibility
    np.random.seed(42)
    
    # Run multiple experiments
    num_experiments = 20
    summary = run_multiple_experiments(num_exp=num_experiments, T=5, verbose_individual=False)
    
    # Print comprehensive summary
    print_experiment_summary(summary)