from typing import Dict, List, Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
import bisect

class GPFuser:
    
    def __init__(self):
        self.ego_keep_quantile = 1.5      # keep pool points where ego is in top X% uncertainty
        self.kernel_params_x = {'length_scale': 1.0, 'output_scale': 1.0}
        self.kernel_params_y = {'length_scale': 2.0, 'output_scale': 0.5}
        self.jitter = 1e-6
    
    @staticmethod    
    def plot_ego_vs_pool(xy_ego, var_ego, xy_pool, var_pool, title='Ego vs Pool Predictions'):
        """
        Visualizes ego and pool predictions as scatter plots with uncertainty ellipses (assuming diagonal covariance).
        
        Ego: Blue circles
        Pool: Red circles
        """
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(title)
        ax.set_xlabel("Y position")
        ax.set_ylabel("X position")
        ax.invert_xaxis()  # put Y on x-axis and X on y-axis, and flip X
    
        # --- Ego ---
        for pt, var in zip(xy_ego, var_ego):
            x, y = pt[1], pt[0]  # y on x-axis, x on y-axis
            sx, sy = np.sqrt(var[1]), np.sqrt(var[0])
            circle = patches.Ellipse((x, y), 2*sx, 2*sy, edgecolor='blue', facecolor='blue', alpha=0.3)
            ax.add_patch(circle)
            ax.plot(x, y, 'bo')
    
        # --- Pool ---
        for pt, var in zip(xy_pool, var_pool):
            x, y = pt[1], pt[0]
            sx, sy = np.sqrt(var[1]), np.sqrt(var[0])
            circle = patches.Ellipse((x, y), 2*sx, 2*sy, edgecolor='red', facecolor='red', alpha=0.3)
            ax.add_patch(circle)
            ax.plot(x, y, 'ro')
    
        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Ego', markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Pool', markerfacecolor='red', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
        ax.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod   
    def plot_fused_prediction(xy_ego, var_ego, xy_pool, var_pool, xy_fused, var_fused, title='Ego vs Pool vs Fused'):

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(title)
        ax.set_xlabel("Y position")
        ax.set_ylabel("X position")
        ax.invert_xaxis()
    
        # --- Ego (blue) ---
        for pt, var in zip(xy_ego, var_ego):
            x, y = pt[1], pt[0]
            sx, sy = np.sqrt(var[1]), np.sqrt(var[0])
            ellipse = Ellipse((x, y), 2*sx, 2*sy, edgecolor='blue', facecolor='blue', alpha=0.3)
            ax.add_patch(ellipse)
            ax.plot(x, y, 'bo')
    
        # --- Pool (red) ---
        for pt, var in zip(xy_pool, var_pool):
            x, y = pt[1], pt[0]
            sx, sy = np.sqrt(var[1]), np.sqrt(var[0])
            ellipse = Ellipse((x, y), 2*sx, 2*sy, edgecolor='red', facecolor='red', alpha=0.3)
            ax.add_patch(ellipse)
            ax.plot(x, y, 'ro')
    
        # --- Fused (green) ---
        for pt, var in zip(xy_fused, var_fused):
            x, y = pt[1], pt[0]
            sx, sy = np.sqrt(var[1]), np.sqrt(var[0])
            ellipse = Ellipse((x, y), 2*sx, 2*sy, edgecolor='green', facecolor='green', alpha=0.3)
            ax.add_patch(ellipse)
            ax.plot(x, y, 'go')
    
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Ego', markerfacecolor='blue', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Pool', markerfacecolor='red', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Fused', markerfacecolor='green', markersize=8)
        ]
        
        ax.legend(handles=legend_elements, loc='upper right')
        ax.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        
    def gate_by_relative_uncertainty(self, t_ego, var_ego, t_pool, var_pool):
        """
        Gate pool predictions by comparing their uncertainty to the closest ego prediction.
        Only keep pool points that are more confident (lower uncertainty) than ego.
    
        Parameters:
            t_ego (np.ndarray): Sorted 1D array of ego timestamps
            var_ego (np.ndarray): (N, D) array of ego variances
            t_pool (np.ndarray): 1D array of pool timestamps
            var_pool (np.ndarray): (M, D) array of pool variances
    
        Returns:
            keep_mask (np.ndarray): Boolean mask over t_pool indicating which pool predictions to keep
        """
        ego_unc_scalar = var_ego.sum(axis=1)  # (N,)
        pool_unc_scalar = var_pool.sum(axis=1)  # (M,)
    
        # Ensure ego times are sorted (should already be)
        assert np.all(np.diff(t_ego) >= 0), "t_ego must be sorted"
    
        keep_mask = np.zeros_like(t_pool, dtype=bool)
    
        for i, t in enumerate(t_pool):
            # Find index of closest ego timestamp via binary search
            idx = bisect.bisect_left(t_ego, t)
            if idx == 0:
                closest_idx = 0
            elif idx == len(t_ego):
                closest_idx = len(t_ego) - 1
            else:
                before = t_ego[idx - 1]
                after = t_ego[idx]
                closest_idx = idx - 1 if abs(t - before) <= abs(t - after) else idx
    
            # Compare uncertainties
            #if pool_unc_scalar[i] <= self.ego_keep_quantile*ego_unc_scalar[closest_idx]:
            keep_mask[i] = True
    
        return keep_mask

    @staticmethod
    def _ego_to_arrays(ego_pred: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ts = sorted(ego_pred['pred'].keys(), key=float)
        t = np.array(ts, float)
        xy = np.array([ego_pred['pred'][tt] for tt in ts], float)
        var = np.array([(ego_pred['cov'][tt][0][0], ego_pred['cov'][tt][1][1]) for tt in ts], float)
        return t, xy, var
    
    @staticmethod
    def _pool_to_arrays(pool: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        t_lst, xy_lst, var_lst = [], [], []
        for p in pool:
            t_arr = np.asarray(p['t'], float)
            xy_arr = np.asarray(p['xy'], float)
            var_arr = np.array([(C[0][0], C[1][1]) for C in p['cov']], float)
            t_lst.append(t_arr)
            xy_lst.append(xy_arr)
            var_lst.append(var_arr)

        t_all  = np.concatenate(t_lst, axis=0)
        xy_all = np.concatenate(xy_lst, axis=0)
        var_all= np.concatenate(var_lst, axis=0)
        return t_all, xy_all, var_all
    
        

    def fuse(self, preds_with_pools, visualize=True):
        """
        Returns fused predictions in the same ego format per object_id:
          {obj_id: {'timestamp': ego_ts_ms, 'pred': {t:[x,y]}, 'cov': {t:[[2x2]]}}}
        """
        fused: Dict[int, Dict] = {}
    
        for obj_id, (ego_pred, pool) in preds_with_pools.items():
            if not pool:
                fused[obj_id] = ego_pred
                continue
    
            # Step 1: Parse ego and pool into arrays
            t_ego, xy_ego, var_ego = self._ego_to_arrays(ego_pred)
            t_pool, xy_pool, var_pool = self._pool_to_arrays(pool)
            
            print(t_ego)
            print(t_pool)
    
            if visualize:
                self.plot_ego_vs_pool(xy_ego, var_ego, xy_pool, var_pool, title=f"Before Gating (Obj {obj_id})")
    
            # Step 2: Gating by relative uncertainty
            keep_mask = self.gate_by_relative_uncertainty(t_ego, var_ego, t_pool, var_pool)
            if not np.any(keep_mask):
                keep_mask[:] = True
    
            t_pool, xy_pool, var_pool = t_pool[keep_mask], xy_pool[keep_mask], var_pool[keep_mask]
    
            if visualize:
                self.plot_ego_vs_pool(xy_ego, var_ego, xy_pool, var_pool, title=f"After Gating (Obj {obj_id})")
    
            # Step 3: Combine ego and pool for GP training
            t_train = np.concatenate([t_ego, t_pool])
            xy_train = np.concatenate([xy_ego, xy_pool])
            var_train = np.concatenate([var_ego, var_pool])
    
            # Step 4: GP Fusion (NumPy-only) — train on combined ego+pool, predict on ego timeline
            fused_xy = np.zeros_like(xy_ego)
            fused_var = np.zeros_like(var_ego)
    
            X_train = t_train.reshape(-1, 1)
            X_ego   = t_ego.reshape(-1, 1)
    
            # --- RBF kernel helper (Constant * RBF) ---
            def rbf_kernel_1d(X1, X2, length_scale, output_scale):
                # X1: (N,1), X2: (M,1)
                d2 = (X1 - X2.T) ** 2  # (N,M)
                return output_scale * np.exp(-0.5 * d2 / (length_scale ** 2 + 1e-18))
    
            # --- robust Cholesky solve with tiny jitter growth ---
            def chol_solve(K, y):
                jitter_vals = [0.0, 1e-9, 1e-8, 1e-7, 1e-6]
                last_err = None
                for j in jitter_vals:
                    try:
                        L = np.linalg.cholesky(K + j * np.eye(K.shape[0]))
                        # Solve K a = y using Cholesky (L L^T a = y)
                        a = np.linalg.solve(L.T, np.linalg.solve(L, y))
                        return L, a
                    except np.linalg.LinAlgError as e:
                        last_err = e
                        continue
                raise last_err
    
            for dim in range(2):  # 0: x, 1: y
                # --- per-dim normalization of y and alpha (hetero noise) ---
                y_raw   = xy_train[:, dim].astype(float)
                alpha_r = var_train[:, dim].astype(float) + self.jitter  # raw units
    
                y_mean = float(y_raw.mean())
                y_std  = float(y_raw.std()) if y_raw.std() > 0 else 1.0
    
                y_train_n = (y_raw - y_mean) / y_std
                alpha_n   = np.maximum(alpha_r / (y_std ** 2), 1e-9)  # keep a small floor in normalized units
    
                # --- simple data-driven hyperparams (no optimizer) ---
                diffs = np.diff(np.sort(t_train))
                typical_gap = np.percentile(diffs, 50) if len(diffs) else 0.1
                ls = max(float(typical_gap), 1e-3)                # length-scale
                os = max(float(np.var(y_train_n) - np.median(alpha_n)), 1e-6)  # signal variance
    
                # --- build kernels ---
                K_xx  = rbf_kernel_1d(X_train, X_train, ls, os)             # (N,N)
                K_xX  = rbf_kernel_1d(X_ego,   X_train, ls, os)             # (T,N)
                K_XX  = rbf_kernel_1d(X_ego,   X_ego,   ls, os)             # (T,T)
                K_t   = K_xx + np.diag(alpha_n)                              # add heteroscedastic noise
    
                # --- posterior mean and covariance in normalized space ---
                L, a = chol_solve(K_t, y_train_n)                            # a = K_t^{-1} y
                mean_n = K_xX @ a                                            # (T,)
    
                # cov = K_XX - K_xX @ K_t^{-1} @ K_xX^T  ; compute via Cholesky:
                V = np.linalg.solve(L, K_xX.T)                               # (N,T)
                cov_n = K_XX - (V.T @ V)                                     # (T,T)
                var_n = np.clip(np.diag(cov_n), 0.0, np.inf)                 # per-point variance
    
                # --- de-normalize ---
                fused_xy[:, dim]  = y_mean + y_std * mean_n
                fused_var[:, dim] = (y_std ** 2) * var_n
    
            # Step 5: Visualization
            if visualize:
                self.plot_fused_prediction(
                    xy_ego, var_ego, xy_pool, var_pool, fused_xy, fused_var, title=f"Fused Result (Obj {obj_id})"
                )
    
            # Step 6: Convert back to dict format
            pred_dict = {float(t): [float(x), float(y)] for t, (x, y) in zip(t_ego, fused_xy)}
            cov_dict = {
                float(t): [[float(var[0]), 0.0], [0.0, float(var[1])]]
                for t, var in zip(t_ego, fused_var)
            }
    
            fused[obj_id] = {
                'timestamp': ego_pred['timestamp'],
                'pred': pred_dict,
                'cov': cov_dict,
            }
            
            print(fused[obj_id])
    
        return fused




