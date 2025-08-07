import numpy as np
from numba import njit, prange


@njit
def _predict_numba(beta, X_train, X_test, eps=1e-64):
    dim   = len(X_test)
    res   = np.empty(dim)
    d2    = np.sum((X_test - X_train)**2, axis=1)
    p_i   = np.exp(-d2 * beta)
    p_sum = np.sum(p_i) + eps

    for coord in range(dim):
        res[coord] = np.sum(X_train[:, coord] * p_i) / p_sum

    return res


@njit(parallel=True)
def _predict_values_numba(theta, X_train, Y_train, X_test, eps=1e-64):
    n_points = X_test.shape[0]
    res      = np.empty(n_points)
    beta     = 1.0 / (2.0 * theta) 

    for idx in prange(n_points):
        d2       = np.sum((X_test[idx] - X_train) ** 2, axis=1)
        prob     = np.exp( - d2 * beta )
        res[idx] = np.sum(Y_train * prob) / (np.sum(prob) + eps)

    return res

@njit
def _lagrange_mul_1D_numba(theta, x_test, x_train, max_iter):
    n_points     = x_test.shape[0]
    x_test_tilde = x_test.copy()
    beta         = 0.5 / theta 
    for idx in range( n_points ):
        err_old = np.inf
        distance = (x_test[idx,:] - x_train)

        for _ in range(max_iter):
            diff = (x_test_tilde[idx,:] - x_train)

            R_sqr_new = diff**2  * beta

            prob = np.exp(-R_sqr_new)

            num = np.sum(distance * prob)
            den = np.sum(distance * diff * prob)

            if den == 0 or np.isnan(den):
                break

            LM_step = theta * num / den
            res_tmp = _predict_numba(beta, x_train, x_test_tilde[idx]+LM_step)
            err_new = np.mean(res_tmp-x_test[idx])
            
            if err_new < err_old:
                    x_test_tilde[idx] += LM_step
                    err_old = err_new
            else:
                    break  

    return x_test_tilde


@njit
def _lagrange_mul_NDim_numba(theta, x_test, x_train, max_iter):
    n_points, dim = x_test.shape
    x_tilde       = x_test.copy()
    J = np.zeros((dim, dim))
    F = np.zeros(dim)

    beta = 0.5 / theta

    for idx in range(n_points):
        err_old = np.inf
        distance = x_test[idx] - x_train

        for _ in range(max_iter):
            d_beta = x_tilde[idx] - x_train
            R2     = np.sum(d_beta ** 2, axis=1) * beta
            p_i    = np.exp(-R2)

            F.fill(0.0)
            J.fill(0.0)

            for row in range(dim):
                temp   = distance[:, row] * p_i
                F[row] = np.sum(temp)
                for col in range(dim):
                    J[row, col] = np.sum(temp * d_beta[:, col])

            try:
                LM_step = theta * np.linalg.solve(J, F)
            except Exception:
                break

            res_tmp = _predict_numba(beta, x_train, x_tilde[idx] + LM_step)
            err_new = np.mean((res_tmp - x_test[idx])**2)

            if err_new < err_old:
                x_tilde[idx] += LM_step
                if abs(err_old - err_new) < abs(0.05 * err_old):
                    break
                err_old = err_new
            else:
                break

    return x_tilde


class KBRInterpolator:
    def __init__(self, max_iter=5, alpha=0.5, theta_min=1e-7, theta_steps=15, verbose=True):
        """
        Initialize KBR Interpolator.
        
        Parameters:
        -----------
        max_iter : int, default=10
            Maximum iterations for Lagrange multiplier optimization
        alpha : float, default=0.5
            Relaxation parameter for theta optimization (0 < alpha <= 1)
        theta_min : float, default=1e-8
            Minimum threshold for theta values
        theta_steps : int, default=20
            Number of theta values to test during optimization
        tolerance : float, default=1e-6
            Convergence tolerance for iterative methods
        verbose : bool, default=True
            Whether to print optimization progress
        """      

        self.max_iter = max_iter
        self.alpha = alpha
        self.theta_min = theta_min
        self.theta_steps = theta_steps
        self.verbose = verbose

        self.theta_optimum = None
        self.X_train = None
        self.Y_train = None
        self.dim = None
        self.is_fitted = False

    def _compute_l2_norm(self, A, B):
        return np.sqrt(np.mean((A - B) ** 2))

    def _optimize_theta(self, data, theta0=-1):
        n_points, dim = data.shape
        theta_lst = np.zeros(self.theta_steps)
        scale = dim / n_points
        R2 = np.sum((data - np.mean(data, axis=0))**2, axis=1)

        if theta0 < 0:
            theta0 = (1. / (dim * n_points)) * np.sum(R2)

        theta_lst[0] = theta_old = theta0
        for idx in range(1, self.theta_steps):
            coeff = 1.0 / (2.0 * theta_old)
            exp_term = np.exp(-R2 * coeff)

            num = np.sum(R2 * exp_term)
            den = np.sum(exp_term)
            theta_new = self.alpha * scale * (num / den) + (1.0 - self.alpha) * theta_old

            if theta_new < self.theta_min or not np.isfinite(theta_new):
                theta_lst[idx] = theta_old
                if self.verbose:
                    print(f"Theta optimization converged early: {theta_old}")
                return theta_lst[:idx + 1]

            theta_lst[idx] = theta_old = theta_new
        return theta_lst

    def fit(self, train_data, test_data, theta=None, apply_correction=True):

        self.dim     = train_data.shape[1] - 1
        self.X_train = train_data[:, :self.dim]
        X_test       = test_data[:, :self.dim]
        Y_train_base = train_data[:, self.dim]
        Y_test_base  = test_data[:, self.dim]

        self.Y_train = Y_train_base
        Y_test = Y_test_base

        if theta is None:

            theta_lst = self._optimize_theta(self.X_train)

            if self.verbose:
                print(f"Optimizing theta")

            mse_lst = np.zeros_like(theta_lst)

            for idx, theta in enumerate(theta_lst):
                if apply_correction:
                    if self.dim == 1:
                        x_train_hat = _lagrange_mul_1D_numba(theta, self.X_train, self.X_train, self.max_iter)
                        x_test_hat  = _lagrange_mul_1D_numba(theta, X_test, self.X_train, self.max_iter)
                    else:
                        x_train_hat = _lagrange_mul_NDim_numba(theta, self.X_train, self.X_train, self.max_iter)
                        x_test_hat  = _lagrange_mul_NDim_numba(theta, X_test, self.X_train, self.max_iter)

                    y_train_pred = _predict_values_numba(theta, self.X_train, self.Y_train, x_train_hat)
                    res = _predict_values_numba(theta, self.X_train, 2 * self.Y_train - y_train_pred, x_test_hat)
                else:
                    res = _predict_values_numba(theta, self.X_train, self.Y_train, X_test)

                mse_lst[idx] = self._compute_l2_norm(res, Y_test)

            best_idx   = np.argmin(mse_lst)
            best_theta = theta_lst[best_idx]

            if self.verbose:
                print(f"Best theta: {best_theta}, RMSE: {mse_lst[best_idx]}")

        else:

            if self.verbose:
                print(f"Using provided theta: {theta}")

            best_theta = theta

        self.theta_optimum = best_theta
        self.apply_correction = apply_correction
        self.is_fitted = True


    def predict(self, x):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        x_validation = x[:, :self.dim]

        if self.apply_correction:
            if self.dim == 1:
                x_train_hat = _lagrange_mul_1D_numba(self.theta_optimum, self.X_train, self.X_train, self.max_iter)
                x_val_hat   = _lagrange_mul_1D_numba(self.theta_optimum, x_validation, self.X_train, self.max_iter)
            else:
                x_train_hat = _lagrange_mul_NDim_numba(self.theta_optimum, self.X_train, self.X_train, self.max_iter)
                x_val_hat   = _lagrange_mul_NDim_numba(self.theta_optimum, x_validation, self.X_train, self.max_iter)

            y_train_pred = _predict_values_numba(self.theta_optimum, self.X_train,      self.Y_train                , x_train_hat)
            pred         = _predict_values_numba(self.theta_optimum, self.X_train, 2 * (self.Y_train) - y_train_pred, x_val_hat  )
        else:
            pred = _predict_values_numba(self.theta_optimum, self.X_train, self.Y_train, x_validation)

        if self.verbose:
            print("Prediction completed.")

        return pred