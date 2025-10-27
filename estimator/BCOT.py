#!/usr/bin/env python3
"""
Bi-Causal Optimal Transport (BCOT) robust filter.
"""

import numpy as np
import time
from .utils import optimize

class BCOT:
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_Sigma_v,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_Sigma_v,
                 radius, maxit=20,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                 x0_scale=None, w_scale=None, v_scale=None,
                 input_lower_bound=None, input_upper_bound=None):
        """Initialize BCOT filter.
          
          The following are provided in two sets:
            (i) True parameters (for simulating the system)
            (ii) Nominal parameters (obtained via EM, used in filtering)
           
          Additional parameter:
            radius: robustness parameter (BCOT constraint radius)
            maxit: maximum iterations for the optimization routine.
          
          x0_max, x0_min, etc.: bounds for non–normal distributions.
        """
        self.T = T
        self.dist = dist
        self.noise_dist = noise_dist
        self.A, self.C = system_data
        self.B = B
        
        # True parameters (for simulation)
        self.true_x0_mean = true_x0_mean
        self.true_x0_cov = true_x0_cov
        self.true_mu_w = true_mu_w
        self.true_Sigma_w = true_Sigma_w
        self.true_mu_v = true_mu_v
        self.true_Sigma_v = true_Sigma_v
        
        # Nominal parameters (for filtering)
        self.nominal_x0_mean = nominal_x0_mean
        self.nominal_x0_cov = nominal_x0_cov
        self.nominal_mu_w = nominal_mu_w
        self.nominal_Sigma_w = nominal_Sigma_w
        self.nominal_mu_v = nominal_mu_v
        self.nominal_Sigma_v = nominal_Sigma_v
        
        # Bounds for non–normal distributions.
        if self.dist in ["uniform", "quadratic"]:
            self.x0_max = x0_max
            self.x0_min = x0_min
            self.w_max = w_max
            self.w_min = w_min
        if self.noise_dist in ["uniform", "quadratic"]:
            self.v_max = v_max
            self.v_min = v_min
        if self.dist == "laplace":
            self.x0_scale = x0_scale
            self.w_scale = w_scale
        if self.noise_dist == "laplace":
            self.v_scale = v_scale
        
        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        
        # BCOT-specific parameters:
        self.radius = radius
        self.maxit = maxit
        # Use nominal process and measurement noise covariances as references.
        self.Bp = self.nominal_Sigma_w  # reference process noise covariance
        self.Dp = self.nominal_Sigma_v         # reference measurement noise covariance
        
        # LQR gain will be assigned externally.
        self.K_lqr = None
        
        # Support for shared noise sequences (for fair comparison with other filters)
        self.shared_noise_sequences = None
        self._noise_index = 0
        
        # Input saturation bounds for trajectory tracking
        self.input_lower_bound = input_lower_bound
        self.input_upper_bound = input_upper_bound

    # --- Sampling Functions for True Noise ---
    def normal(self, mu, Sigma, N=1):
        return np.random.multivariate_normal(mu[:, 0], Sigma, size=N).T

    def uniform(self, a, b, N=1):
        n = a.shape[0]
        return a + (b - a) * np.random.rand(n, N)

    def quad_inverse(self, x, b, a):
        row, col = x.shape
        for i in range(row):
            for j in range(col):
                beta = (a[j] + b[j]) / 2.0
                alpha = 12.0 / ((b[j] - a[j])**3)
                tmp = 3 * x[i, j] / alpha - (beta - a[j])**3
                if tmp >= 0:
                    x[i, j] = beta + tmp ** (1.0/3.0)
                else:
                    x[i, j] = beta - (-tmp) ** (1.0/3.0)
        return x

    def quadratic(self, max_val, min_val, N=1):
        n = min_val.shape[0]
        x = np.random.rand(N, n).T  # shape: (n, N)
        return self.quad_inverse(x, max_val, min_val)

    def laplace(self, mu, scale, N=1):
        return np.random.laplace(mu[:, 0], scale, size=(N, mu.shape[0])).T

    def saturate_input(self, u):
        """Apply input saturation if bounds are specified."""
        if self.input_lower_bound is not None and self.input_upper_bound is not None:
            return np.clip(u, self.input_lower_bound, self.input_upper_bound)
        return u

    def sample_initial_state(self):
        if self.shared_noise_sequences is not None:
            return self.shared_noise_sequences['x0']
        
        if self.dist == "normal":
            return self.normal(self.true_x0_mean, self.true_x0_cov, N=1)
        elif self.dist == "quadratic":
            return self.quadratic(self.x0_max, self.x0_min, N=1)
        elif self.dist == "laplace":
            return self.laplace(self.true_x0_mean, self.x0_scale, N=1)
        else:
            raise ValueError("Unsupported distribution for initial state.")

    def sample_process_noise(self):
        if self.shared_noise_sequences is not None:
            w = self.shared_noise_sequences['w'][self._noise_index]
            return w
        
        if self.dist == "normal":
            return self.normal(self.true_mu_w, self.true_Sigma_w, N=1)
        elif self.dist == "quadratic":
            return self.quadratic(self.w_max, self.w_min, N=1)
        elif self.dist == "laplace":
            return self.laplace(self.true_mu_w, self.w_scale, N=1)
        else:
            raise ValueError("Unsupported distribution for process noise.")

    def sample_measurement_noise(self):
        if self.shared_noise_sequences is not None:
            v = self.shared_noise_sequences['v'][self._noise_index]
            return v
        
        if self.noise_dist == "normal":
            return self.normal(self.true_mu_v, self.true_Sigma_v, N=1)
        elif self.noise_dist == "quadratic":
            return self.quadratic(self.v_max, self.v_min, N=1)
        elif self.noise_dist == "laplace":
            return self.laplace(self.true_mu_v, self.v_scale, N=1)
        else:
            raise ValueError("Unsupported distribution for measurement noise.")

    # --- Forward Simulation Using the Robust BCOT Update with LQR ---
    def forward(self):
        start_time = time.time()
        T = self.T
        nx = self.nx
        ny = self.ny
        A = self.A
        C = self.C
        B = self.B
        nu = B.shape[1]  # Number of control inputs
        
        # Reset noise index for consistent sequences across experiments
        self._noise_index = 0
        
        # Allocate arrays for true state, measurements, and state estimates.
        x = np.zeros((T+1, nx, 1))
        y = np.zeros((T+1, ny, 1))
        x_est = np.zeros((T+1, nx, 1))
        u_traj = np.zeros((T, nu, 1))  # Input trajectory
        
        # --- Initialization ---
        x[0] = self.sample_initial_state()
        # Set initial robust filter estimate to the nominal initial mean.
        pre_mean = self.nominal_x0_mean.copy()
        pre_cov = self.nominal_x0_cov.copy()
        
        # First measurement.
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0
        # Initial robust update:
        update_mean, update_cov = optimize(self.ny, self.nx, self.radius, A, self.Bp, C, self.Dp,
                                           pre_cov, y[0], pre_mean, self.maxit, algo='BCOT')
        pre_mean = update_mean.copy()
        pre_cov = update_cov.copy()
        x_est[0] = update_mean.copy()
        
        mse = np.zeros(T+1)
        mse[0] = np.linalg.norm(x_est[0] - x[0])**2
        
        # --- Time Update and Robust Filtering with Control ---
        for t in range(T):
            # Compute control input: u[t] = -K_lqr * x_est[t]
            if self.K_lqr is None:
                raise ValueError("LQR gain (K_lqr) has not been assigned!")
            u = -self.K_lqr @ x_est[t]
            
            # Apply input saturation if bounds are specified
            u = self.saturate_input(u)
            
            # Store input trajectory
            u_traj[t] = u.copy()
            
            # True state propagation: x[t+1] = A*x[t] + B*u + w
            w = self.sample_process_noise()
            x[t+1] = A @ x[t] + B @ u + w
            
            # Increment noise index after sampling process noise
            self._noise_index += 1
            
            # Measurement:
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v
            
            # Robust prediction: incorporate control input and nominal process noise mean.
            x_pred = A @ x_est[t] + B @ u + self.nominal_mu_w
            # Robust update via BCOT optimization:
            update_mean, update_cov = optimize(ny, nx, self.radius, A, self.Bp, C, self.Dp,
                                               pre_cov, y[t+1], x_pred, self.maxit, algo='BCOT')
            x_est[t+1] = update_mean.copy()
            pre_mean = update_mean.copy()
            pre_cov = update_cov.copy()
            
            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2
        
        comp_time = time.time() - start_time
        return {'comp_time': comp_time,
                'state_traj': x,
                'output_traj': y,
                'est_state_traj': x_est,
                'input_traj': u_traj,
                'mse': mse}
    # --- Forward Trajectory Tracking Simulation with LQR Control ---
    def forward_track(self, desired_trajectory):
        """
        Performs closed-loop simulation for trajectory tracking using the robust BCOT filter.
        The control input is computed based on the tracking error:
            u[t] = -K_lqr * (x_est[t] - x_d[t]),
        where x_d[t] is the desired state at time t.
        The robust update is performed via the BCOT optimization routine.
        """
        start_time = time.time()
        T = self.T
        nx = self.nx
        ny = self.ny
        A = self.A
        C = self.C
        B = self.B
        nu = B.shape[1]  # Number of control inputs

        # Reset noise index for consistent sequences across experiments
        self._noise_index = 0

        # Allocate arrays.
        x = np.zeros((T+1, nx, 1))         # true state trajectory
        y = np.zeros((T+1, ny, 1))           # measurement trajectory
        x_est = np.zeros((T+1, nx, 1))       # BCOT state estimates
        u_traj = np.zeros((T, nu, 1))       # Input trajectory
        error = np.zeros((T+1, nx, 1))       # tracking error
        mse = np.zeros(T+1)                # mean squared error

        # --- Initialization ---
        x[0] = self.sample_initial_state()
        x_est[0] = self.nominal_x0_mean.copy()
        # Compute initial tracking error using the desired state at t=0.
        desired = desired_trajectory[:, 0].reshape(-1, 1)
        error[0] = x_est[0] - desired

        # First measurement.
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0
        # Initialize robust update.
        pre_mean = self.nominal_x0_mean.copy()
        pre_cov = self.nominal_x0_cov.copy()
        update_mean, update_cov = optimize(self.ny, self.nx, self.radius, A, self.Bp, C, self.Dp,
                                           pre_cov, y[0], pre_mean, self.maxit, algo='BCOT')
        pre_mean = update_mean.copy()
        pre_cov = update_cov.copy()
        x_est[0] = update_mean.copy()
        mse[0] = np.linalg.norm(x_est[0] - x[0])**2

        # --- Time Update and Robust Filtering with Trajectory Tracking ---
        for t in range(T):
            # Get desired state at current time step.
            desired = desired_trajectory[:, t].reshape(-1, 1)
            # Compute tracking error.
            error[t] = x_est[t] - desired
            # Compute control input based on tracking error.
            if self.K_lqr is None:
                raise ValueError("LQR gain (K_lqr) has not been assigned!")
            u = -self.K_lqr @ error[t]
            
            # Apply input saturation if bounds are specified
            u = self.saturate_input(u)
            
            # Store input trajectory
            u_traj[t] = u.copy()

            # True state propagation.
            w = self.sample_process_noise()
            x[t+1] = A @ x[t] + B @ u + w

            # Increment noise index after sampling process noise
            self._noise_index += 1

            # Measurement.
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v

            # Robust prediction.
            x_pred = A @ x_est[t] + B @ u + self.nominal_mu_w
            # Robust update via BCOT optimization.
            update_mean, update_cov = optimize(self.ny, self.nx, self.radius, A, self.Bp, C, self.Dp,
                                               pre_cov, y[t+1], x_pred, self.maxit, algo='BCOT')
            x_est[t+1] = update_mean.copy()
            pre_mean = update_mean.copy()
            pre_cov = update_cov.copy()

            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2

        comp_time = time.time() - start_time
        return {'comp_time': comp_time,
                'state_traj': x,
                'output_traj': y,
                'est_state_traj': x_est,
                'input_traj': u_traj,
                'mse': mse,
                'tracking_error': error}
    
    def forward_track_MPC(self, desired_trajectory):
        """
        Performs closed-loop simulation for trajectory tracking using the robust BCOT filter with MPC control.
        The control input is computed based on MPC tracking error:
            u[t] = -K_lqr * (x_est[t] - x_d[t]),
        where x_d[t] is the desired state at time t.
        The robust update is performed via the BCOT optimization routine.
        """
        start_time = time.time()
        T = self.T
        nx = self.nx
        ny = self.ny
        A = self.A
        C = self.C
        B = self.B
        nu = B.shape[1]  # Number of control inputs

        # Reset noise index for consistent sequences across experiments
        self._noise_index = 0

        # Allocate arrays.
        x = np.zeros((T+1, nx, 1))         # true state trajectory
        y = np.zeros((T+1, ny, 1))           # measurement trajectory
        x_est = np.zeros((T+1, nx, 1))       # BCOT state estimates
        u_traj = np.zeros((T, nu, 1))       # Input trajectory
        error = np.zeros((T+1, nx, 1))       # tracking error
        mse = np.zeros(T+1)                # mean squared error

        # --- Initialization ---
        x[0] = self.sample_initial_state()
        x_est[0] = self.nominal_x0_mean.copy()
        # Compute initial tracking error using the desired state at t=0.
        desired = desired_trajectory[:, 0].reshape(-1, 1)
        error[0] = x_est[0] - desired

        # First measurement.
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0
        # Initialize robust update.
        pre_mean = self.nominal_x0_mean.copy()
        pre_cov = self.nominal_x0_cov.copy()
        update_mean, update_cov = optimize(self.ny, self.nx, self.radius, A, self.Bp, C, self.Dp,
                                           pre_cov, y[0], pre_mean, self.maxit, algo='BCOT')
        pre_mean = update_mean.copy()
        pre_cov = update_cov.copy()
        x_est[0] = update_mean.copy()
        mse[0] = np.linalg.norm(x_est[0] - x[0])**2

        # --- Time Update and Robust Filtering with MPC Trajectory Tracking ---
        for t in range(T):
            # Get desired state at current time step.
            desired = desired_trajectory[:, t].reshape(-1, 1)
            # Compute tracking error.
            error[t] = x_est[t] - desired
            # Compute control input based on MPC tracking error.
            if self.K_lqr is None:
                raise ValueError("LQR gain (K_lqr) has not been assigned!")
            u = -self.K_lqr @ error[t]
            
            # Apply input saturation if bounds are specified
            u = self.saturate_input(u)
            
            # Store input trajectory
            u_traj[t] = u.copy()

            # True state propagation.
            w = self.sample_process_noise()
            x[t+1] = A @ x[t] + B @ u + w

            # Increment noise index after sampling process noise
            self._noise_index += 1

            # Measurement.
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v

            # Robust prediction.
            x_pred = A @ x_est[t] + B @ u + self.nominal_mu_w
            # Robust update via BCOT optimization.
            update_mean, update_cov = optimize(ny, nx, self.radius, A, self.Bp, C, self.Dp,
                                               pre_cov, y[t+1], x_pred, self.maxit, algo='BCOT')
            x_est[t+1] = update_mean.copy()
            pre_mean = update_mean.copy()
            pre_cov = update_cov.copy()

            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2

        comp_time = time.time() - start_time
        return {'comp_time': comp_time,
                'state_traj': x,
                'output_traj': y,
                'est_state_traj': x_est,
                'input_traj': u_traj,
                'mse': mse,
                'tracking_error': error}
