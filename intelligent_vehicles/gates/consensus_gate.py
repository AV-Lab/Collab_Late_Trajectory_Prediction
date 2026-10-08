"""Training-free consensus gating for trajectory predictions."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ConsensusDecision:
    """Consensus result expressed as indices into the original pool."""

    matrix: np.ndarray
    scores: np.ndarray
    comparable_counts: np.ndarray
    medoid_index: Optional[int]
    medoid_score: Optional[float]
    threshold_exceeded_indices: List[int]
    retained_indices: List[int]
    rejected_indices: List[int]
    reason: str


class ConsensusGate:
    """Compute pairwise disagreement between candidate trajectories.

    Each pool entry must contain ``t`` (one-dimensional future timestamps) and
    ``xy`` (positions aligned with ``t`` and shaped ``[T, 2]``).

    Interpolation is used only in temporary comparison arrays. Pool entries
    and their original timestamps, positions, and covariances are not changed.
    """

    def __init__(
        self,
        min_overlap: float,
        threshold: float = 2.0,
        min_predictions: int = 3,
        enabled: bool = True,
    ):
        min_overlap = float(min_overlap)
        if not np.isfinite(min_overlap) or min_overlap < 0.0:
            raise ValueError("min_overlap must be a finite non-negative value.")
        self.min_overlap = min_overlap
        self.threshold = float(threshold)
        self.min_predictions = int(min_predictions)
        self.enabled = bool(enabled)
        self.decisions = {}

    def apply(self, preds_with_pools):
        """Filter remote inputs while keeping ego as the fusion reference.

        Category I decision indices refer to ``[ego, *pool]``; Category II
        indices refer to the remote pool. Disabled calls perform no scoring.
        """
        self.decisions = {}
        if not self.enabled:
            return preds_with_pools

        filtered_predictions = {}
        for target_id, values in preds_with_pools.items():
            timestamp, ego_prediction, pool, node_type = values
            filtered_pool = pool
            has_ego = node_type == 1
            if len(pool) + int(has_ego) >= self.min_predictions:
                if has_ego:
                    keys = sorted(ego_prediction["pred"], key=float)
                    ego_candidate = {
                        "t": [float(key) for key in keys],
                        "xy": [ego_prediction["pred"][key] for key in keys],
                        "cov": [ego_prediction["cov"][key] for key in keys],
                    }
                    _, decision = self.filter_pool(
                        [ego_candidate, *pool], ego_index=0,
                    )
                    filtered_pool = [
                        pool[index - 1] for index in decision.retained_indices
                        if index != 0
                    ]
                else:
                    filtered_pool, decision = self.filter_pool(pool)
                self.decisions[target_id] = decision

            filtered_predictions[target_id] = (
                timestamp, ego_prediction, filtered_pool, node_type,
            )
        return filtered_predictions

    @staticmethod
    def _trajectory_arrays(prediction: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Return validated, time-ordered comparison arrays for one entry."""
        if not isinstance(prediction, dict):
            raise TypeError("Each prediction in the pool must be a dictionary.")
        if "t" not in prediction or "xy" not in prediction:
            raise KeyError("Each prediction must contain 't' and 'xy'.")

        timestamps = np.asarray(prediction["t"], dtype=np.float64)
        positions = np.asarray(prediction["xy"], dtype=np.float64)

        order = np.argsort(timestamps, kind="stable")
        timestamps = timestamps[order]
        positions = positions[order]
        if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("Prediction timestamps must be unique.")

        return timestamps, positions

    def _pairwise_disagreement(
        self,
        first: Tuple[np.ndarray, np.ndarray],
        second: Tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Calculate time-normalized RMS disagreement for one trajectory pair.

        ``np.nan`` denotes a pair whose common future interval is too short.
        """
        first_t, first_xy = first
        second_t, second_xy = second

        overlap_start = max(float(first_t[0]), float(second_t[0]))
        overlap_end = min(float(first_t[-1]), float(second_t[-1]))
        overlap_duration = overlap_end - overlap_start

        if overlap_duration <= 0.0 or overlap_duration < self.min_overlap:
            return float("nan")

        first_inside = first_t[
            (first_t >= overlap_start) & (first_t <= overlap_end)
        ]
        second_inside = second_t[
            (second_t >= overlap_start) & (second_t <= overlap_end)
        ]
        comparison_t = np.unique(np.concatenate((
            np.asarray([overlap_start, overlap_end], dtype=np.float64),
            first_inside,
            second_inside,
        )))

        first_interpolated = np.column_stack((
            np.interp(comparison_t, first_t, first_xy[:, 0]),
            np.interp(comparison_t, first_t, first_xy[:, 1]),
        ))
        second_interpolated = np.column_stack((
            np.interp(comparison_t, second_t, second_xy[:, 0]),
            np.interp(comparison_t, second_t, second_xy[:, 1]),
        ))

        squared_distance = np.sum(
            (first_interpolated - second_interpolated) ** 2,
            axis=1,
        )
        integral = float(np.trapz(squared_distance, comparison_t))
        normalized = max(integral / overlap_duration, 0.0)
        return float(np.sqrt(normalized))

    def disagreement_matrix(self, pool: Sequence[Dict]) -> np.ndarray:
        """Return the symmetric pairwise disagreement matrix for ``pool``.

        The diagonal is zero. An off-diagonal entry is ``np.nan`` when the
        corresponding trajectories do not have sufficient overlap.
        """
        if pool is None:
            raise TypeError("pool must be a sequence of prediction dictionaries.")

        trajectories = [self._trajectory_arrays(prediction) for prediction in pool]
        count = len(trajectories)
        matrix = np.full((count, count), np.nan, dtype=np.float64)
        np.fill_diagonal(matrix, 0.0)

        for first_index in range(count):
            for second_index in range(first_index + 1, count):
                disagreement = self._pairwise_disagreement(
                    trajectories[first_index],
                    trajectories[second_index],
                )
                matrix[first_index, second_index] = disagreement
                matrix[second_index, first_index] = disagreement

        return matrix

    @staticmethod
    def _row_statistics(
        matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return median, mean, and comparison count for every matrix row."""
        count = matrix.shape[0]
        scores = np.full(count, np.nan, dtype=np.float64)
        means = np.full(count, np.nan, dtype=np.float64)
        comparable_counts = np.zeros(count, dtype=np.int64)

        for index in range(count):
            comparable = np.isfinite(matrix[index])
            comparable[index] = False
            values = matrix[index, comparable]
            comparable_counts[index] = values.size
            if values.size:
                scores[index] = float(np.median(values))
                means[index] = float(np.mean(values))

        return scores, means, comparable_counts

    @staticmethod
    def _average_uncertainty(prediction: Dict) -> float:
        """Return mean positional variance for uncertainty tie-breaking."""
        covariance = prediction.get("cov")
        if covariance is None:
            return float("inf")
        covariance = np.asarray(covariance, dtype=np.float64)
        if covariance.ndim != 3 or covariance.shape[1:] != (2, 2):
            return float("inf")
        return float(np.mean(np.trace(covariance, axis1=1, axis2=2) / 2.0))

    def decide(
        self, pool: Sequence[Dict], ego_index: Optional[int] = None,
    ) -> ConsensusDecision:
        """Select a medoid and identify predictions beyond the threshold.

        The disagreement matrix is always computed. For an eligible pool, the
        medoid and threshold-exceeding predictions are identified even when
        consensus is disabled; disabling only prevents their removal. Rejection
        is also skipped when the pool is too small or no pair is comparable.
        An incomparable prediction is retained, and the medoid is always kept.
        When supplied, ego_index participates in scoring but is always retained
        as the residual GP reference, even if it exceeds the threshold.
        """
        matrix = self.disagreement_matrix(pool)
        scores, means, comparable_counts = self._row_statistics(matrix)
        count = len(pool)
        all_indices = list(range(count))

        if count < self.min_predictions:
            return ConsensusDecision(
                matrix=matrix,
                scores=scores,
                comparable_counts=comparable_counts,
                medoid_index=None,
                medoid_score=None,
                threshold_exceeded_indices=[],
                retained_indices=all_indices,
                rejected_indices=[],
                reason="insufficient_predictions",
            )

        candidates = [index for index in all_indices if np.isfinite(scores[index])]
        if not candidates:
            return ConsensusDecision(
                matrix=matrix,
                scores=scores,
                comparable_counts=comparable_counts,
                medoid_index=None,
                medoid_score=None,
                threshold_exceeded_indices=[],
                retained_indices=all_indices,
                rejected_indices=[],
                reason="no_comparable_predictions",
            )

        uncertainties = [self._average_uncertainty(prediction) for prediction in pool]
        medoid_index = min(
            candidates,
            key=lambda index: (
                scores[index],
                means[index],
                uncertainties[index],
                index,
            ),
        )

        threshold_exceeded_indices = [
            index
            for index in all_indices
            if index != medoid_index
            and np.isfinite(matrix[medoid_index, index])
            and matrix[medoid_index, index] > self.threshold
        ]

        if self.enabled:
            rejected_indices = [
                index for index in threshold_exceeded_indices if index != ego_index
            ]
            reason = "consensus_applied"
        else:
            rejected_indices = []
            reason = "disabled"

        rejected = set(rejected_indices)
        retained_indices = [
            index for index in all_indices if index not in rejected
        ]

        return ConsensusDecision(
            matrix=matrix,
            scores=scores,
            comparable_counts=comparable_counts,
            medoid_index=medoid_index,
            medoid_score=float(scores[medoid_index]),
            threshold_exceeded_indices=threshold_exceeded_indices,
            retained_indices=retained_indices,
            rejected_indices=rejected_indices,
            reason=reason,
        )

    def filter_pool(
        self,
        pool: Sequence[Dict],
        ego_index: Optional[int] = None,
    ) -> Tuple[List[Dict], ConsensusDecision]:
        """Return retained original predictions and the consensus decision."""
        decision = self.decide(pool, ego_index=ego_index)
        retained_pool = [pool[index] for index in decision.retained_indices]
        return retained_pool, decision
