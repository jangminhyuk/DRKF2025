import numpy as np
import cvxpy as cp
import mosek
import control
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh
import os

np.random.seed(42)

def uniform(a, b, N=1):
    n = a.shape[0]
    return a[:, None] + (b[:, None] - a[:, None]) * np.random.rand(n, N)

def normal(mu, Sigma, N=1):
    return np.random.multivariate_normal(mu.ravel(), Sigma, size=N).T

def quad_inverse(x, b, a):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            beta = 0.5 * (a[j] + b[j])
            alpha = 12.0 / ((b[j] - a[j]) ** 3)
            tmp = 3 * x[i, j] / alpha - (beta - a[j]) ** 3
            x[i, j] = beta + (tmp if tmp >= 0 else -(-tmp) ** (1. / 3.)) ** (1. / 3.)
    return x

def quadratic(wmax, wmin, N=1):
    x = np.random.rand(N, wmin.shape[0]).T
    return quad_inverse(x, wmax, wmin)

def gen_sample_dist_inf(dist, N_sample, mu=None, Sigma=None, w_min=None, w_max=None):
    if dist == "normal":
        w = normal(mu, Sigma, N=N_sample)
    elif dist == "uniform":
        w = uniform(w_min, w_max, N=N_sample)
    elif dist == "quadratic":
        w = quadratic(wmax=w_max, wmin=w_min, N=N_sample)
    else:
        raise ValueError("Unsupported distribution.")
    mean_ = np.mean(w, axis=1, keepdims=True)
    var_ = np.cov(w)
    return mean_, var_

def enforce_positive_definiteness(M, epsilon=1e-3):
    M = (M + M.T) / 2.0
    eigvals = np.linalg.eigvalsh(M)
    min_eig = np.min(eigvals)
    if min_eig < epsilon:
        M = M + (epsilon - min_eig) * np.eye(M.shape[0])
    return M

def check_assumptions(A, Sigma_w_nom, C, Sigma_v_nom, T):
    n = A.shape[0]
    m = C.shape[0]
    if np.any(np.linalg.eigvals(Sigma_w_nom) <= 0):
        raise ValueError("Sigma_w_nom is not positive definite.")
    if np.any(np.linalg.eigvals(Sigma_v_nom) <= 0):
        raise ValueError("Sigma_v_nom is not positive definite.")
    O = control.obsv(A, C)
    if np.linalg.matrix_rank(O) < n:
        raise ValueError("The pair (A, C) is not observable.")
    try:
        B = np.linalg.cholesky(Sigma_w_nom)
    except np.linalg.LinAlgError:
        raise ValueError("Sigma_w_nom is not positive definite for Cholesky.")
    CC = control.ctrb(A, B)
    if np.linalg.matrix_rank(CC) < n:
        raise ValueError("The pair (A, sqrt(Sigma_w_nom)) is not reachable.")
    O_T_blocks = [C @ np.linalg.matrix_power(A, T-1-i) for i in range(T)]
    O_T = np.vstack(O_T_blocks)
    if np.linalg.matrix_rank(O_T) < n:
        raise ValueError(f"O_T does not have full column rank; (A, C) may not be observable with T={T}")
    print("Assumptions verified.")

def dr_kf_solve_inf(A, C, Sigma_w_nom, Sigma_v_nom, theta_w, theta_v, delta=1e-6):
    n = C.shape[1]
    m = Sigma_v_nom.shape[0]
    lam_min_w_nom = float(np.linalg.eigvalsh(Sigma_w_nom).min())
    lam_min_v_nom = float(np.linalg.eigvalsh(Sigma_v_nom).min())
    eps_psd = 1e-9

    Sigma_x = cp.Variable((n, n), PSD=True)
    Sigma_xm = cp.Variable((n, n), PSD=True)
    Sigma_w = cp.Variable((n, n), PSD=True)
    Sigma_v = cp.Variable((m, m), PSD=True)
    Y = cp.Variable((n, n))
    Z = cp.Variable((m, m))

    obj = cp.Maximize(cp.trace(Sigma_x) )

    M_meas = cp.bmat([
        [Sigma_xm - Sigma_x,         Sigma_xm @ C.T],
        [C @ Sigma_xm,     C @ Sigma_xm @ C.T + Sigma_v]
    ])
    constraints = [
        cp.trace(Sigma_w + Sigma_w_nom - 2*Y) <= theta_w**2,
        cp.bmat([[Sigma_w_nom, Y],
                 [Y.T,         Sigma_w]]) >> 0,

        cp.trace(Sigma_v + Sigma_v_nom - 2*Z) <= theta_v**2,
        cp.bmat([[Sigma_v_nom, Z],
                 [Z.T,         Sigma_v]]) >> 0,

        Sigma_xm == A @ Sigma_x @ A.T + Sigma_w,
        M_meas >> 0, 
        Sigma_w >> lam_min_w_nom* np.eye(n),
        Sigma_v >> lam_min_v_nom * np.eye(m)
    ]

    mosek_params = {
        'MSK_IPAR_INTPNT_MAX_ITERATIONS': 400,
        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-6,
        'MSK_DPAR_INTPNT_TOL_PFEAS':      1e-6,
        'MSK_DPAR_INTPNT_TOL_DFEAS':      1e-6,
    }

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False, mosek_params=mosek_params)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise cp.error.SolverError(f"MOSEK status: {prob.status}")
    return Sigma_x.value


def dr_kf_solve_onestep(Sigma_x_prev, A, C, Sigma_w_nom, Sigma_v_nom, theta_w, theta_v, delta=1e-6):
    n = Sigma_x_prev.shape[0]
    m = Sigma_v_nom.shape[0]

    lam_min_w_nom = float(np.linalg.eigvalsh(Sigma_w_nom).min())
    lam_min_v_nom = float(np.linalg.eigvalsh(Sigma_v_nom).min())
    Sigma_x = cp.Variable((n, n), PSD=True)
    Sigma_xm = cp.Variable((n, n), PSD=True)
    Sigma_w = cp.Variable((n, n), PSD=True)
    Sigma_v = cp.Variable((m, m), PSD=True)
    Y = cp.Variable((n, n))
    Z = cp.Variable((m, m))

    obj = cp.Maximize(cp.trace(Sigma_x))

    M_meas = cp.bmat([
        [Sigma_xm - Sigma_x,         Sigma_xm @ C.T],
        [C @ Sigma_xm,     C @ Sigma_xm @ C.T + Sigma_v]
    ])
    constraints = [
        cp.trace(Sigma_w + Sigma_w_nom - 2*Y) <= theta_w**2,
        cp.bmat([[Sigma_w_nom, Y],
                 [Y.T,         Sigma_w]]) >> 0,

        cp.trace(Sigma_v + Sigma_v_nom - 2*Z) <= theta_v**2,
        cp.bmat([[Sigma_v_nom, Z],
                 [Z.T,         Sigma_v]]) >> 0,

        Sigma_xm == A @ Sigma_x_prev @ A.T + Sigma_w,
        M_meas >> 0,
        Sigma_w >> lam_min_w_nom* np.eye(n),
        Sigma_v >> lam_min_v_nom * np.eye(m)
    ]

    mosek_params = {
        'MSK_IPAR_INTPNT_MAX_ITERATIONS': 400,
        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-6,
        'MSK_DPAR_INTPNT_TOL_PFEAS':      1e-6,
        'MSK_DPAR_INTPNT_TOL_DFEAS':      1e-6,
    }

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False, mosek_params=mosek_params)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise cp.error.SolverError(f"MOSEK status: {prob.status}")
    return Sigma_x.value









def run_dr_kf_once(n=2, m=1, steps=200, T=20, q=100,
                   tol_fro=1e-4, tol_trace=1e-4, progress_every=5):
    try:
        # System (2×2, 1×2)
        A = np.array([[0.1, 1.0],
                      [0.0, 1.2]])
        C = np.array([[1.0, -1.0]])
        Sigma_w_nom = 1.0 * np.eye(A.shape[0])
        Sigma_v_nom = 1.0 * np.eye(C.shape[0])
        # --- Use fixed theta values ---
        theta_w = 0.1
        theta_v = 0.1
        theta_x0 = 0.1

        print(f"[INFO] Using fixed theta values: theta_w = {theta_w}, theta_v = {theta_v}, theta_x0 = {theta_x0}")
        

        # --- initial posterior covariance (match system dimension!) ---
        nx = A.shape[0]
        Sigma_x_prev = np.eye(nx)

        # --- steady-state via SDP once ---
        Sigma_x_inf = dr_kf_solve_inf(A, C, Sigma_w_nom, Sigma_v_nom,
                                      theta_w, theta_v, delta=1e-6)

        posterior_list = []
        conv_norms = []
        trace_rel_diff_list = []

        print("[INFO] starting DR-KF one-step iterations…")
        for step in range(steps):
            try:
                Sigma_x_sol = dr_kf_solve_onestep(
                    Sigma_x_prev, A, C, Sigma_w_nom, Sigma_v_nom,
                    theta_w, theta_v, delta=1e-6
                )
            except cp.error.SolverError as e:
                print(f"[ERROR] SDP failed at step {step}: {e}")
                return None

            posterior_list.append(Sigma_x_sol)

            # Relative trace gap to steady-state (fast convergence check)
            rel_trace_gap = abs(np.trace(Sigma_x_inf) - np.trace(Sigma_x_sol)) / max(1e-12, np.trace(Sigma_x_inf))
            trace_rel_diff_list.append(rel_trace_gap)

            # Frobenius gap to previous iterate
            if step > 0:
                fro_gap = np.linalg.norm(posterior_list[-1] - posterior_list[-2], 'fro')
                conv_norms.append(fro_gap)

                # Early stopping on either criterion
                if (fro_gap < tol_fro) or (rel_trace_gap < tol_trace):
                    print(f"[INFO] early stop at step {step}: "
                          f"||Δ||_F={fro_gap:.2e}, trace_rel_gap={rel_trace_gap:.2e}")
                    break

            if (step % progress_every) == 0:
                print(f"[DBG] step {step:3d} | trace_rel_gap={rel_trace_gap:.2e}")

            Sigma_x_prev = Sigma_x_sol

        return {
            "A": A, "C": C,
            "Sigma_w_nom": Sigma_w_nom, "Sigma_v_nom": Sigma_v_nom,
            "theta_w": theta_w, "theta_v": theta_v,
            "posterior_list": posterior_list, "conv_norms": conv_norms,
            "trace_rel_diff_list": trace_rel_diff_list, "Sigma_x_inf": Sigma_x_inf
        }

    except (cp.error.SolverError, Exception) as e:
        print(f"[FATAL] run_dr_kf_once aborted: {e}")
        return None


if __name__=="__main__":
    tol = 1e-6  # convergence tolerance for final norm
    num_exp = 1  # number of valid experiments to collect
    success_count = 0
    valid_experiments = []
    
    batch_size = 1  # number of experiments to run in parallel per batch
    
    while len(valid_experiments) < num_exp:
        results = Parallel(n_jobs=1)(
            delayed(run_dr_kf_once)(n=2, m=1, steps=35, T=8, q=20, tol_fro=1e-4, tol_trace=1e-4)
            for _ in range(batch_size)
        )
        for res in results:
            if res is not None:
                valid_experiments.append(res)
                final_norm = res["conv_norms"][-1] if res["conv_norms"] else float('inf')
                if final_norm < tol:
                    success_count += 1
                #print(f"\nValid experiment count: {len(valid_experiments)}/{num_exp}")
                if len(valid_experiments) >= num_exp:
                    break

    success_rate = (success_count / num_exp) * 100
    print("\n==================================================")
    print("Computation finished!!")

    # --------------------------
    # Convergence Rate Analysis
    # --------------------------
    res = valid_experiments[0]
    posterior_list = res["posterior_list"]
    Sigma_x_inf = res["Sigma_x_inf"]
    
    converged_cov = posterior_list[-1]

    # Compute the 2-norm error for each time step with respect to the converged covariance
    errors_conv = [np.linalg.norm(S - converged_cov, 2) for S in posterior_list]


    plt.figure(figsize=(12, 6))
    plt.semilogy(range(len(res["trace_rel_diff_list"])), res["trace_rel_diff_list"],
                marker='o', linestyle='-', linewidth=1, markersize=5, color='black')
    plt.xlabel('Time Step', fontsize=24)
    plt.ylabel(r'$\left|\frac{\mathrm{Tr}[\Sigma_{x,\infty}^{*}] - \mathrm{Tr}[\Sigma_{x,t}]}{\mathrm{Tr}[\Sigma_{x,\infty}^{*}]}\right|$', fontsize=28)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35], fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    
    # Ensure results folder exists
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    output_path = os.path.join(results_dir, 'convergence_rate_analysis.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Convergence analysis plot saved as '{output_path}'")
    plt.show()


