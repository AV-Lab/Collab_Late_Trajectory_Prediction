from typing import Dict, List, Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
import bisect

class GPFuser:
    
    @staticmethod
    def plot_predictions(
        xy_pool, var_pool,
        xy_ego = None, 
        var_ego = None,
        xy_fused = None,
        var_fused = None,
        xy_gt = None,
        title='Prediction Visualization'
    ):
        """
        Plots ego, pool, fused (optional), and ground-truth (optional) trajectories with uncertainty ellipses.
    
        - Ego:    blue
        - Pool:   red
        - Fused:  green
        - GT:     black crosses
        """
    
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(title)
        ax.set_xlabel("Y position")
        ax.set_ylabel("X position")
        ax.invert_xaxis()  # to match original layout
    
        legend_elements = []
        
        # --- Pool ---
        for pt, var in zip(xy_pool, var_pool):
            x, y = pt[1], pt[0]
            sx, sy = np.sqrt(var[1]), np.sqrt(var[0])
            ellipse = Ellipse((x, y), 2*sx, 2*sy, edgecolor='red', facecolor='red', alpha=0.05)
            ax.add_patch(ellipse)
            ax.plot(x, y, 'ro')
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Pool', markerfacecolor='red', markersize=8))
    
        # --- Ego ---
        if xy_ego is not None and var_ego is not None:
            for pt, var in zip(xy_ego, var_ego):
                x, y = pt[1], pt[0]
                sx, sy = np.sqrt(var[1]), np.sqrt(var[0])
                ellipse = Ellipse((x, y), 2*sx, 2*sy, edgecolor='blue', facecolor='blue', alpha=0.05)
                ax.add_patch(ellipse)
                ax.plot(x, y, 'bo')
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Ego', markerfacecolor='blue', markersize=8))
    
        # --- Fused (optional) ---
        if xy_fused is not None and var_fused is not None:
            for pt, var in zip(xy_fused, var_fused):
                x, y = pt[1], pt[0]
                sx, sy = np.sqrt(var[1]), np.sqrt(var[0])
                ellipse = Ellipse((x, y), 2*sx, 2*sy, edgecolor='green', facecolor='green', alpha=0.05)
                ax.add_patch(ellipse)
                ax.plot(x, y, 'go')
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Fused', markerfacecolor='green', markersize=8))
    
        # --- Ground Truth (optional, no variance) ---
        if xy_gt is not None:
            for pt in xy_gt:
                x, y = pt[1], pt[0]
                ax.plot(x, y, 'kx')  # black cross
            legend_elements.append(Line2D([0], [0], marker='x', color='k', label='Ground Truth', markersize=8))
    
        ax.legend(handles=legend_elements, loc='upper right')
        ax.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

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
   
    @staticmethod    
    def plot_normalized_training_data(t_train, y_train_norm, var_norm, dim_label="x"):
        """
        Plots normalized training outputs with error bars showing variance.
    
        Parameters:
            t_train (np.ndarray): (N,) training timepoints
            y_train_norm (np.ndarray): (N,) normalized training values
            var_norm (np.ndarray): (N,) normalized variances
            dim_label (str): 'x' or 'y', for axis labeling
        """

        std_dev = np.sqrt(var_norm)
    
        plt.figure(figsize=(8, 4))
        plt.errorbar(t_train, y_train_norm, yerr=std_dev, fmt='o', capsize=3, alpha=0.7, label=f'Normalized {dim_label}')
        plt.title(f"Normalized Training Values with Variance ({dim_label})")
        plt.xlabel("Time")
        plt.ylabel("Normalized Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()    

    def ego_based_gp(self, ego_pred, pool):
        t_ego, xy_ego, var_ego = self._ego_to_arrays(ego_pred)
        t_pool, xy_pool, var_pool = self._pool_to_arrays(pool)       
        t_train = np.concatenate([t_ego, t_pool])
        xy_train = np.concatenate([xy_ego, xy_pool])
        var_train = np.concatenate([var_ego, var_pool])
        xy_fused = np.zeros_like(xy_ego)
        var_fused = np.zeros_like(var_ego)
        X_train = t_train.reshape(-1, 1)
        X_ego   = t_ego.reshape(-1, 1)
        
        for dim in range(2):
            y_raw   = xy_train[:, dim].astype(float)
            var_raw = var_train[:, dim].astype(float)
            
            ego_on_train = np.interp(t_train, t_ego, xy_ego[:, dim])

            r_train = y_raw - ego_on_train
            alpha_r = np.clip(var_raw, 1e-6, 1e4)
            
            # Time-scale bounds as above
            diffs = np.diff(np.sort(t_train))
            typical_gap = np.percentile(diffs, 50) if len(diffs) else 0.1
            ls0 = max(float(typical_gap), 1e-3)
            
            kernel = C(1.0, (1e-2, 10.0)) * RationalQuadratic(length_scale=ls0, alpha=1.0, length_scale_bounds=(ls0/5.0, ls0*5.0))
            gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha_r, normalize_y=True, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0)
            gp.fit(X_train, r_train)
            
            r_mean, cov_r = gp.predict(X_ego, return_cov=True)
            xy_fused[:, dim] = xy_ego[:, dim] + r_mean
            var_fused[:, dim] = var_ego[:, dim] + np.clip(np.diag(cov_r), 0.0, np.inf)
                        
        fused_pred = {t: xy for t,xy in zip(t_ego, xy_fused)}
        fused_cov_matrices = {t: [[v[0], 0.0], [0.0, v[1]]] for t, v in zip(fused_pred, var_fused)}
        
        return fused_pred, fused_cov_matrices, xy_pool, var_pool, xy_ego, var_ego, xy_fused, var_fused
    
    def pool_gp(self, ego_ts, pool):
        t_pool, xy_pool, var_pool = self._pool_to_arrays(pool)
        t_q = np.asarray(ego_ts, float).reshape(-1)
        X_train = t_pool.reshape(-1, 1)
        X_query = t_q.reshape(-1, 1)
        xy_fused = np.zeros((len(t_q), 2), dtype=float)
        var_fused = np.zeros((len(t_q), 2), dtype=float)

        diffs = np.diff(np.sort(t_pool))
        typical_gap = np.percentile(diffs, 50) if len(diffs) else 0.1
        ls0 = max(float(typical_gap), 1e-3)

        for dim in range(2):
            y_raw = xy_pool[:, dim].astype(float)
            v_raw = var_pool[:, dim].astype(float)
            y_mean = float(y_raw.mean())
            y_std = float(y_raw.std()) if y_raw.std() > 0 else 1.0
            y_n = (y_raw - y_mean) / y_std
            alpha_n = np.maximum(v_raw / (y_std ** 2), 1e-8)

            kernel = C(1.0, (1e-2, 10.0)) * RationalQuadratic(length_scale=ls0, alpha=1.0, length_scale_bounds=(ls0/5.0, ls0*5.0))
            gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha_n, normalize_y=False, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0)
            gp.fit(X_train, y_n)

            mu_n, std_n = gp.predict(X_query, return_std=True)
            xy_fused[:, dim] = y_mean + y_std * mu_n
            var_fused[:, dim] = (y_std ** 2) * (std_n ** 2)

        fused_pred = {float(t): [float(x), float(y)] for t, (x, y) in zip(t_q, xy_fused)}
        fused_cov = {float(t): [[float(v[0]), 0.0], [0.0, float(v[1])]] for t, v in zip(t_q, var_fused)}

        return fused_pred, fused_cov, xy_pool, var_pool, xy_fused, var_fused
        

    def fuse(self, ego_ts, preds_with_pools, trajectories, visualize=False):
        """
        Returns fused predictions in the same ego format per object_id:
          {obj_id: {'timestamp': ego_ts_ms, 'pred': {t:[x,y]}, 'cov': {t:[[2x2]]}}}
        """
        fused: Dict[int, Dict] = {}
    
        for obj_id, (timestamp, ego_pred, pool, type_) in preds_with_pools.items():
            if type_ == 1:
                # NEW BEHAVIOR:
                # Ignore ego prediction completely IF there is something in the pool.
                # If pool is empty, keep ego_pred as a fallback.
                if not pool:
                    fused[obj_id] = ego_pred
                    continue

                # Optional ground truth for visualization
                xy_gt = None
                if obj_id in trajectories:
                    xy_gt = [(p.x, p.y) for p in trajectories[obj_id]["future"][:len(ego_ts)]]

                # Use pool-only GP fusion, just like type 2
                fused_pred, fused_cov, xy_pool, var_pool, xy_fused, var_fused = self.pool_gp(ego_ts, pool)
                fused[obj_id] = {"pred": fused_pred, "cov": fused_cov}

                if visualize:
                    # Ego is ignored in fusion; we also omit it from the plot here
                    self.plot_predictions(xy_pool, var_pool, None, None, xy_fused, var_fused, xy_gt)
                
            elif type_ == 2:  
                if not pool:
                    continue
                
                fused_pred, fused_cov, xy_pool, var_pool, xy_fused, var_fused = self.pool_gp(ego_ts, pool)
                fused[obj_id] = {"pred": fused_pred, "cov": fused_cov}

                if visualize:
                    self.plot_predictions(xy_pool, var_pool, None, None, xy_fused, var_fused)
                
    
        return fused

