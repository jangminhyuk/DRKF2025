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



def compute_matrices(T, A, Sigma_w_nom, C, Sigma_v_nom):
    n = A.shape[0]
    m = C.shape[0]
    Sigma_v_nom = enforce_positive_definiteness(Sigma_v_nom, epsilon=1e-3)
    B = np.linalg.cholesky(Sigma_w_nom)
    sqrt_Sigma_v_nom = np.linalg.cholesky(Sigma_v_nom)
    
    # 1. R_T: [B, A·B, A²·B, …, A^(T-1)·B]
    R_T_blocks = [np.linalg.matrix_power(A, i) @ B for i in range(T)]
    R_T = np.hstack(R_T_blocks)
    
    # 2. O_T: Vertical stacking of [C A^(T-1); C A^(T-2); ...; C]
    O_T_blocks = [C @ np.linalg.matrix_power(A, T-1-i) for i in range(T)]
    O_T = np.vstack(O_T_blocks)
    if np.linalg.matrix_rank(O_T) < n:
        raise ValueError(f"O_T does not have full column rank; (A,C) may not be observable with T={T}")
    
    # 3. O_T^R: Vertical stacking of [A^(T-1); A^(T-2); …; I]
    O_T_R_blocks = [np.linalg.matrix_power(A, T-1-i) for i in range(T)]
    O_T_R = np.vstack(O_T_R_blocks)
    
    # 4. D_T: I_T ⊗ sqrt(Sigma_v_nom)
    D_T = np.kron(np.eye(T), sqrt_Sigma_v_nom)
    
    # 5. Build block Hankel matrices L_T and H_T.
    L_blocks = [[(np.linalg.matrix_power(A, j-i-1) @ B) if (j - i >= 1) else np.zeros((n, n))
                 for j in range(T)] for i in range(T)]
    H_blocks = [[(C @ (np.linalg.matrix_power(A, j-i-1) @ B)) if (j - i >= 1) else np.zeros((m, n))
                 for j in range(T)] for i in range(T)]
    L_T = np.block(L_blocks)
    H_T = np.block(H_blocks)
    
    # 6. Compute tilde_phi_T.
    I_inner = np.eye(T * n)
    DDT = D_T @ D_T.T
    inv_DDT = np.linalg.inv(DDT)
    inner_term = I_inner + H_T.T @ inv_DDT @ H_T
    inner_inv = np.linalg.inv(inner_term)
    M = L_T @ inner_inv @ L_T.T
    eigvals = np.linalg.eigvals(M)
    lambda_max_val = np.max(np.real(eigvals))
    tilde_phi_T = 1.0 / lambda_max_val
    
    return {
        "R_T": R_T,
        "O_T": O_T,
        "O_T_R": O_T_R,
        "D_T": D_T,
        "L_T": L_T,
        "H_T": H_T,
        "tilde_phi_T": tilde_phi_T
    }

def find_phi_T(O_T, O_T_R, L_T, H_T, D_T, tilde_phi_T, tol_eig=1e-10, bisection_tol=1e-10, max_iter=1000):
    M = D_T @ D_T.T + H_T @ H_T.T
    M_inv = np.linalg.inv(M)
    J_T = O_T_R - L_T @ H_T.T @ M_inv @ O_T
    Omega_T = O_T.T @ M_inv @ O_T

    eig_vals = np.linalg.eigvals(Omega_T)
    lambda_min = np.min(np.real(eig_vals))
    if lambda_min < 0:
        raise ValueError("Omega_T is not positive definite. Check that all assumptions are met.")

    I_N = np.eye(L_T.shape[1])
    inv_DDT = np.linalg.inv(D_T @ D_T.T)
    inner_term = I_N + H_T.T @ inv_DDT @ H_T
    inner_inv = np.linalg.inv(inner_term)
    I_full = np.eye(L_T.shape[0])

    def lambda_min_Omega(phi):
        S_phi = - (1.0 / phi) * I_full + L_T @ inner_inv @ L_T.T
        try:
            S_phi_inv = np.linalg.inv(S_phi)
        except np.linalg.LinAlgError:
            return -np.inf
        Omega_phi = Omega_T + J_T.T @ S_phi_inv @ J_T
        Omega_phi = (Omega_phi + Omega_phi.T) / 2.0
        eigvals = np.linalg.eigvals(Omega_phi)
        return np.min(np.real(eigvals))

    phi_lower = 0.0
    phi_upper = tilde_phi_T
    iteration = 0
    while (phi_upper - phi_lower) > bisection_tol and iteration < max_iter:
        iteration += 1
        phi_mid = (phi_lower + phi_upper) / 2.0
        f_mid = lambda_min_Omega(phi_mid)
        if f_mid > tol_eig:
            phi_lower = phi_mid
        else:
            phi_upper = phi_mid

    phi_final = phi_lower
    return phi_final

def dbg(on, msg, **vals):
    if on:
        payload = " | ".join(f"{k}={v}" for k, v in vals.items())
        print(f"[θ-debug] {msg}" + (f" :: {payload}" if payload else ""))

def _solve_theta_from_delta(a_nonneg, delta_max):
    a = max(0.0, float(a_nonneg))
    return max(0.0, (a*a + float(delta_max))**0.5 - a)

def _theta_v_theorem3(phi_v, lam_min_v, lam_max_v, normC2, verbose=False, eps=1e-12):
    dbg(verbose, "θ_v (Thm3) input", phi_v=phi_v, lam_min_v=lam_min_v, lam_max_v=lam_max_v, normC2=normC2)
    bound = normC2 / (lam_max_v + eps)
    if not (phi_v < bound):
        dbg(verbose, "FAIL meas feasibility", need=f"phi_v < {bound}")
        return None
    denom = 1.0 - (phi_v / normC2) * lam_min_v
    dbg(verbose, "Thm3 denom", denom=denom)
    if denom <= 0:
        dbg(verbose, "FAIL denom≤0 in (34)")
        return None
    Bv = lam_min_v / denom
    dbg(verbose, "Thm3 RHS Bv", Bv=Bv)
    if Bv < lam_max_v - 1e-12:
        dbg(verbose, "FAIL RHS(34) < λ_max(Σ̂_v)", Bv=Bv, lam_max_v=lam_max_v)
        return None
    theta_v = max(0.0, Bv**0.5 - lam_max_v**0.5)
    dbg(verbose, "θ_v (Thm3) ok", theta_v=theta_v)
    return theta_v

def _theta_v_opnorm(phi_v, lam_min_v, lam_max_v, normC2, verbose=False, eps=1e-12):
    r = phi_v / (normC2 + eps)
    delta_max = (r * lam_min_v * lam_min_v) / (1.0 + r * lam_min_v)
    a = lam_max_v**0.5
    theta_v = _solve_theta_from_delta(a, delta_max)
    dbg(verbose, "θ_v (opnorm) calc", r=r, delta_max=delta_max, spd_margin=lam_min_v - delta_max, theta_v=theta_v)
    return theta_v

def _theta_w_cor3(phi_x, lam_min_w, lam_max_w, lam_min_v, normC2, normAplus2, verbose=False, eps=1e-12):
    m = 1.0 / (lam_min_w + eps) + normC2 / (lam_min_v + eps)
    dbg(verbose, "Cor3 m", m=m, phi_x=phi_x)
    if not (0.0 <= phi_x < m):
        dbg(verbose, "FAIL process feasibility", need=f"0 ≤ phi_x < m={m}")
        return None
    beta = phi_x / (normAplus2 * m * (m - phi_x))
    theta_w = _solve_theta_from_delta(lam_max_w**0.5, beta)
    dbg(verbose, "θ_w (Cor3) calc", beta=beta, theta_w=theta_w)
    return theta_w

def compute_theta_w_and_v(A, C, Sigma_w_nom, Sigma_v_nom, phi_T,
                          method_v='opnorm',
                          optimize_split=False,
                          verbose=False,
                          eps=1e-12):
    """Compute radii (theta_w, theta_v) ensuring DRKF contraction."""
    Sw = enforce_positive_definiteness((Sigma_w_nom + Sigma_w_nom.T) / 2.0)
    Sv = enforce_positive_definiteness((Sigma_v_nom + Sigma_v_nom.T) / 2.0)
    lam_w = eigvalsh(Sw); lam_v = eigvalsh(Sv)
    lam_min_w, lam_max_w = float(lam_w[0]), float(lam_w[-1])
    lam_min_v, lam_max_v = float(lam_v[0]), float(lam_v[-1])
    normC2 = float(np.linalg.norm(C, 2)**2)
    normAplus2 = float(np.linalg.norm(np.linalg.pinv(A), 2)**2)

    dbg(verbose, "Spectra & norms",
        lam_min_w=lam_min_w, lam_max_w=lam_max_w,
        lam_min_v=lam_min_v, lam_max_v=lam_max_v,
        normC2=normC2, normAplus2=normAplus2, phi_T=phi_T)

    def solve_at(phi_v):
        phi_x = phi_T - phi_v
        dbg(verbose, "Try split", phi_v=phi_v, phi_x=phi_x)

        tv_candidates = []
        if method_v in ('theorem3', 'max'):
            tv_t3 = _theta_v_theorem3(phi_v, lam_min_v, lam_max_v, normC2, verbose=verbose, eps=eps)
            if tv_t3 is not None:
                tv_candidates.append(tv_t3)
        if method_v in ('opnorm', 'max'):
            tv_op = _theta_v_opnorm(phi_v, lam_min_v, lam_max_v, normC2, verbose=verbose, eps=eps)
            if tv_op is not None:
                tv_candidates.append(tv_op)
        if not tv_candidates:
            dbg(verbose, "θ_v infeasible for this split")
            return None, None
        theta_v = max(tv_candidates) if method_v == 'max' else tv_candidates[0]
        dbg(verbose, "θ_v chosen", theta_v=theta_v, method=method_v)

        theta_w = _theta_w_cor3(phi_x, lam_min_w, lam_max_w, lam_min_v, normC2, normAplus2,
                                verbose=verbose, eps=eps)
        if theta_w is None:
            dbg(verbose, "θ_w infeasible for this split")
            return None, None

        dbg(verbose, "Split OK", theta_w=theta_w, theta_v=theta_v)
        return theta_w, theta_v

    phi_v0 = 0.5 * phi_T
    ans = solve_at(phi_v0)
    if ans[0] is not None and ans[1] is not None:
        return ans

    if not optimize_split:
        dbg(verbose, "No optimize_split; returning (None, None)")
        return (None, None)

    m = 1.0/(lam_min_w + eps) + normC2/(lam_min_v + eps)
    lo = max(0.0, phi_T - (m - 1e-9))
    hi = min(phi_T - 1e-9, normC2/(lam_max_v + eps) - 1e-9)
    dbg(verbose, "Search interval", lower_phi_v=lo, upper_phi_v=hi, m=m)
    if lo > hi:
        dbg(verbose, "Empty search interval")
        return (None, None)

    best = (None, None, -float('inf'))
    for pv in np.linspace(lo, hi, 200):
        tw, tv = solve_at(pv)
        if tw is None:
            continue
        score = tw + tv
        if score > best[2]:
            best = (tw, tv, score)

    if best[0] is None:
        dbg(verbose, "Search failed to find feasible split")
        return (None, None)

    dbg(verbose, "Search best", theta_w=best[0], theta_v=best[1], score=best[2])
    return (best[0], best[1])



def run_dr_kf_once(n=2, m=1, steps=200, T=20, q=100,
                   tol_fro=1e-4, tol_trace=1e-4, progress_every=5):
    try:
        # System (2×2, 1×2)
        A = np.array([[0.1, 1.0],
                      [0.0, 1.2]])
        C = np.array([[1.0, -1.0]])
        Sigma_w_nom = 1.0 * np.eye(A.shape[0])
        Sigma_v_nom = 1.0 * np.eye(C.shape[0])

        # --- φ_T ---
        matrices = compute_matrices(T, A, Sigma_w_nom, C, Sigma_v_nom)
        tilde_phi_T = matrices["tilde_phi_T"]
        phi_T = find_phi_T(matrices["O_T"], matrices["O_T_R"],
                           matrices["L_T"], matrices["H_T"],
                           matrices["D_T"], tilde_phi_T)

        print("[INFO] phi_T =", phi_T)

        # --- θ_w, θ_v (with debug prints inside) ---
        theta_w, theta_v = compute_theta_w_and_v(
            A, C, Sigma_w_nom, Sigma_v_nom, phi_T,
            method_v='opnorm',          # try both and take larger feasible θ_v
            optimize_split=True,     # search φ_v if symmetric split fails
            verbose=False             # inline θ-debug already shown in your log
        )

        print(f"[INFO] theta_w = {theta_w}, theta_v = {theta_v}")
        if (theta_w is None) or (theta_v is None) or np.isnan(theta_w) or np.isnan(theta_v):
            print("[WARN] theta computation infeasible.")
            return None

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
            "phi_T": phi_T, "theta_w": theta_w, "theta_v": theta_v,
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


