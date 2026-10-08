from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import get_context
import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RationalQuadratic
from threadpoolctl import threadpool_limits


_FUSION_WORKER_FUSER = None
_FUSION_WORKER_THREAD_LIMITS = None


def _initialize_fusion_worker(default_dtype):
    """Use one numerical thread throughout this CPU worker's lifetime."""
    global _FUSION_WORKER_FUSER, _FUSION_WORKER_THREAD_LIMITS
    torch.set_num_threads(1)
    _FUSION_WORKER_THREAD_LIMITS = threadpool_limits(limits=1)
    torch.set_default_dtype(default_dtype)
    _FUSION_WORKER_FUSER = GPFuserVector()


def _run_fusion_group(ego_ts, jobs):
    """Execute complete independent nodes in a persistent CPU process."""
    return _FUSION_WORKER_FUSER._fuse_group(ego_ts, jobs)


class GPFuser:
    """Gaussian-process trajectory fusion.

    Both ego-based and pool-only fusion fit the x and y trajectory dimensions
    independently using ConstantKernel * RationalQuadratic. The initial length
    scale is the median gap between training timestamps (with a 0.1-second
    fallback), its optimization bounds are length_scale / 5 to length_scale * 5,
    and the RationalQuadratic alpha parameter is 1.0.
    """

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
        
        return fused_pred, fused_cov_matrices
    
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

        return fused_pred, fused_cov
        

    def fuse(self, ego_ts, preds_with_pools):
        """
        Return deterministic GP-fused trajectories per object ID.

        Input covariance is used as GP observation noise. The GP posterior
        covariance is intentionally not exposed as calibrated predictive
        uncertainty, so every fused result keeps the stable prediction schema
        with ``cov`` set to ``None``.
        """
        fused: Dict[int, Dict] = {}
    
        for obj_id, (timestamp, ego_pred, pool, type_) in preds_with_pools.items():
            if type_ == 1:
                if not pool:
                    continue
    
                fused_pred, _ = self.ego_based_gp(ego_pred, pool)
                fused[obj_id] = {"pred": fused_pred, "cov": None}
                
            elif type_ == 2:  
                if not pool:
                    continue
                
                fused_pred, _ = self.pool_gp(ego_ts, pool)
                fused[obj_id] = {"pred": fused_pred, "cov": None}
                
    
        return fused


class GPFuserVector:
    """Vector-valued GP fusion with a shared temporal and task kernel."""

    def __init__(self, workers=1):
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.workers = workers
        self._executor = None
        self._closed = False

    def close(self):
        """Join worker processes; repeated cleanup is harmless."""
        self._closed = True
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True)
            finally:
                self._executor = None

    @staticmethod
    def _sanitize_covariances(covariances, eigenvalue_floor=1e-6):
        covariances = np.asarray(covariances, dtype=np.float64)
        if covariances.ndim != 3 or covariances.shape[1:] != (2, 2):
            raise ValueError("Covariances must have shape [N, 2, 2].")
        if not np.isfinite(covariances).all():
            raise ValueError("Covariances must contain only finite values.")

        covariances = 0.5 * (
            covariances + np.swapaxes(covariances, -1, -2)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariances)
        eigenvalues = np.maximum(eigenvalues, float(eigenvalue_floor))
        return (
            eigenvectors * eigenvalues[..., None, :]
        ) @ np.swapaxes(eigenvectors, -1, -2)

    @classmethod
    def _ego_to_arrays(cls, ego_pred):
        timestamps = sorted(ego_pred["pred"].keys(), key=float)
        t = np.asarray(timestamps, dtype=np.float64)
        xy = np.asarray(
            [ego_pred["pred"][timestamp] for timestamp in timestamps],
            dtype=np.float64,
        )
        covariance = cls._sanitize_covariances([
            ego_pred["cov"][timestamp] for timestamp in timestamps
        ])
        return t, xy, covariance

    @classmethod
    def _pool_to_arrays(cls, pool):
        timestamps, trajectories, covariances = [], [], []
        for prediction in pool:
            t = np.asarray(prediction["t"], dtype=np.float64)
            xy = np.asarray(prediction["xy"], dtype=np.float64)
            covariance = cls._sanitize_covariances(prediction["cov"])
            if len(t) != len(xy) or len(t) != len(covariance):
                raise ValueError(
                    "Prediction timestamps, positions, and covariances must align."
                )
            timestamps.append(t)
            trajectories.append(xy)
            covariances.append(covariance)

        return (
            np.concatenate(timestamps),
            np.concatenate(trajectories),
            np.concatenate(covariances),
        )

    @classmethod
    def _block_diagonal_covariance(cls, covariances):
        covariances = cls._sanitize_covariances(covariances)
        matrix = np.zeros(
            (2 * len(covariances), 2 * len(covariances)),
            dtype=np.float64,
        )
        for index, covariance in enumerate(covariances):
            start = 2 * index
            matrix[start:start + 2, start:start + 2] = covariance
        return matrix

    @classmethod
    def _fit_gp(cls, t_train, xy_train, covariance_train, t_query):
        import gpytorch
        import torch
        from linear_operator.operators import DenseLinearOperator

        class FixedBlockNoiseGaussianLikelihood(
            gpytorch.likelihoods.FixedNoiseGaussianLikelihood
        ):
            """Fixed Gaussian likelihood with correlated 2D noise blocks."""

            def __init__(self, noise_matrix):
                super().__init__(
                    noise=noise_matrix.diagonal(),
                    learn_additional_noise=False,
                )
                self.register_buffer("full_noise", noise_matrix)

            def _shaped_noise_covar(self, base_shape, *params, **kwargs):
                if base_shape[-1] == self.full_noise.shape[-1]:
                    return DenseLinearOperator(self.full_noise)
                return super()._shaped_noise_covar(
                    base_shape, *params, **kwargs
                )

        class VectorExactGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, lengthscale):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ZeroMean()
                temporal_kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(
                        active_dims=(0,),
                        lengthscale_constraint=gpytorch.constraints.Interval(
                            lengthscale / 5.0,
                            lengthscale * 5.0,
                        ),
                    )
                )
                temporal_kernel.base_kernel.initialize(
                    lengthscale=lengthscale,
                )
                self.covar_module = temporal_kernel * gpytorch.kernels.IndexKernel(
                    num_tasks=2,
                    rank=2,
                    active_dims=(1,),
                )

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x),
                    self.covar_module(x),
                )

        t_train = np.asarray(t_train, dtype=np.float64)
        xy_train = np.asarray(xy_train, dtype=np.float64)
        covariance_train = cls._sanitize_covariances(covariance_train)
        t_query = np.asarray(t_query, dtype=np.float64)
        if xy_train.shape != (len(t_train), 2):
            raise ValueError("Training positions must have shape [N, 2].")
        if len(covariance_train) != len(t_train):
            raise ValueError("Training covariances must align with timestamps.")

        train_x = np.column_stack([
            np.repeat(t_train, 2),
            np.tile([0, 1], len(t_train)),
        ])
        query_x = np.column_stack([
            np.repeat(t_query, 2),
            np.tile([0, 1], len(t_query)),
        ])
        train_y = xy_train.reshape(-1)

        diffs = np.diff(np.sort(t_train))
        typical_gap = np.percentile(diffs, 50) if len(diffs) else 0.1
        lengthscale = max(float(typical_gap), 1e-3)

        train_x = torch.as_tensor(train_x, dtype=torch.float64)
        query_x = torch.as_tensor(query_x, dtype=torch.float64)
        train_y = torch.as_tensor(train_y, dtype=torch.float64)
        noise_matrix = torch.as_tensor(
            cls._block_diagonal_covariance(covariance_train),
            dtype=torch.float64,
        )

        likelihood = FixedBlockNoiseGaussianLikelihood(noise_matrix)
        model = VectorExactGP(train_x, train_y, likelihood, lengthscale).double()

        model.train()
        likelihood.train()
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=0.1,
            max_iter=10,
            line_search_fn="strong_wolfe",
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        def closure():
            optimizer.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            return loss

        optimizer.step(closure)

        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-6):
            posterior = model(query_x)

        mean = posterior.mean.reshape(-1, 2).cpu().numpy()
        covariance = posterior.covariance_matrix.cpu().numpy()
        per_step_covariance = np.stack([
            covariance[2 * i:2 * i + 2, 2 * i:2 * i + 2]
            for i in range(len(t_query))
        ])
        return mean, cls._sanitize_covariances(per_step_covariance)

    def ego_based_gp(self, ego_pred, pool):
        t_ego, xy_ego, covariance_ego = self._ego_to_arrays(ego_pred)
        t_pool, xy_pool, covariance_pool = self._pool_to_arrays(pool)
        t_train = np.concatenate([t_ego, t_pool])
        xy_train = np.concatenate([xy_ego, xy_pool])
        covariance_train = np.concatenate([
            covariance_ego, covariance_pool
        ])

        ego_on_train = np.column_stack([
            np.interp(t_train, t_ego, xy_ego[:, dim])
            for dim in range(2)
        ])
        residual_mean, residual_covariance = self._fit_gp(
            t_train,
            xy_train - ego_on_train,
            covariance_train,
            t_ego,
        )
        xy_fused = xy_ego + residual_mean
        covariance_fused = self._sanitize_covariances(
            residual_covariance + covariance_ego
        )

        fused_pred = {t: xy for t, xy in zip(t_ego, xy_fused)}
        fused_covariance = {t: cov for t, cov in zip(t_ego, covariance_fused)}
        return fused_pred, fused_covariance

    def pool_gp(self, ego_ts, pool):
        t_pool, xy_pool, covariance_pool = self._pool_to_arrays(pool)
        t_query = np.asarray(ego_ts, dtype=float)
        xy_mean = xy_pool.mean(axis=0)
        xy_std = xy_pool.std(axis=0)
        xy_std[xy_std == 0.0] = 1.0
        covariance_scale = np.outer(xy_std, xy_std)

        mean_normalized, covariance_normalized = self._fit_gp(
            t_pool,
            (xy_pool - xy_mean) / xy_std,
            covariance_pool / covariance_scale,
            t_query,
        )
        xy_fused = xy_mean + xy_std * mean_normalized
        covariance_fused = self._sanitize_covariances(
            covariance_normalized * covariance_scale
        )

        fused_pred = {
            float(t): [float(x), float(y)]
            for t, (x, y) in zip(t_query, xy_fused)
        }
        fused_covariance = {
            float(t): cov.tolist()
            for t, cov in zip(t_query, covariance_fused)
        }
        return fused_pred, fused_covariance

    def _fuse_group(self, ego_ts, jobs):
        """Run the original full GP for each node with its assigned CPU seed."""
        fused: Dict[int, Dict] = {}

        for obj_id, (_, ego_pred, pool, type_), seed in jobs:
            # Local CPU generators avoid touching predictor CUDA RNG state.
            with torch.random.fork_rng(devices=[]):
                generator = torch.Generator(device="cpu").manual_seed(seed)
                torch.set_rng_state(generator.get_state())
                if type_ == 1:
                    fused_pred, _ = self.ego_based_gp(ego_pred, pool)
                else:
                    fused_pred, _ = self.pool_gp(ego_ts, pool)
                fused[obj_id] = {"pred": fused_pred, "cov": None}

        return fused

    def fuse(self, ego_ts, preds_with_pools):
        """Distribute whole nodes and return the original ordered CPU schema.

        Both execution modes reserve the same per-node seeds before any work.
        This makes initialization independent of worker scheduling; it replaces
        the historical interleaved global RNG initialization sequence.
        """
        if self._closed:
            raise RuntimeError("Cannot use a closed GP fuser")
        eligible = [(obj_id, values) for obj_id, values in preds_with_pools.items()
                    if values[3] in (1, 2) and values[2]]
        if not eligible:
            return {}
        seeds = torch.randint(0, 2**63 - 1, (len(eligible),),
                              dtype=torch.int64, device="cpu").tolist()
        jobs = [(obj_id, values, seed) for (obj_id, values), seed in zip(eligible, seeds)]
        group_count = min(self.workers, len(jobs))
        if group_count == 1:
            previous_threads = torch.get_num_threads()
            with threadpool_limits(limits=1):
                try:
                    if previous_threads != 1:
                        torch.set_num_threads(1)
                    return self._fuse_group(ego_ts, jobs)
                finally:
                    # Restore Torch first; the context then restores each
                    # numerical library's original limit independently.
                    if previous_threads != 1:
                        torch.set_num_threads(previous_threads)

        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=get_context("spawn"),
                initializer=_initialize_fusion_worker,
                initargs=(torch.get_default_dtype(),),
            )
        futures = []
        try:
            for index in range(group_count):
                futures.append(self._executor.submit(
                    _run_fusion_group, ego_ts, jobs[index::group_count],
                ))
            wait(futures)
            fused = {}
            for future in futures:
                fused.update(future.result())
            return {obj_id: fused[obj_id] for obj_id, _ in eligible}
        except BaseException:
            for future in futures:
                future.cancel()
            wait(futures)
            raise
