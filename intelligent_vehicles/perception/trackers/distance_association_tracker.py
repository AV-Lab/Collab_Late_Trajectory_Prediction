"""Category-wise Euclidean association with confirmed tracks matched first."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from intelligent_vehicles.perception.trackers.metric_association_tracker import (
    MIN_CONFIRM_HITS,
    MetricAssociationTracker,
)


class DistanceAssociationTracker(MetricAssociationTracker):
    def _match_group(self, track_indices, detection_indices, detections):
        if not track_indices or not detection_indices:
            return []
        tracks = [self.active_tracklets[i] for i in track_indices]
        predicted = np.array([(tr.kf.H @ (tr.kf.F @ tr.kf.x)).ravel() for tr in tracks])
        observed = np.array([[detections[j]['x'], detections[j]['y']]
                             for j in detection_indices])
        distances = np.linalg.norm(predicted[:, None] - observed[None, :], axis=2)
        limit = self._assoc_dist_for(tracks[0])
        n_tracks, n_detections = distances.shape
        # Dummy columns permit misses; the penalty prioritizes valid match count.
        penalty = (min(n_tracks, n_detections) + 1) * limit
        cost = np.full((n_tracks, n_detections + n_tracks), penalty)
        cost[:, :n_detections] = np.where(distances <= limit, distances, np.inf)
        rows, columns = linear_sum_assignment(cost)
        return [(track_indices[i], detection_indices[j]) for i, j in zip(rows, columns)
                if j < n_detections]

    def track(self, detections):
        matched_tracks = set()
        remaining = set(range(len(detections)))
        for category in MIN_CONFIRM_HITS:
            for confirmed in (True, False):
                track_indices = [i for i, tr in enumerate(self.active_tracklets)
                                 if tr.category == category and tr.is_confirmed == confirmed]
                detection_indices = [j for j in sorted(remaining)
                                     if detections[j]['label'] == category]
                for i, j in self._match_group(track_indices, detection_indices, detections):
                    self.active_tracklets[i].update(detections[j])
                    matched_tracks.add(i)
                    remaining.remove(j)

        for i, tr in enumerate(self.active_tracklets):
            if i not in matched_tracks:
                tr.predict()
                tr.miss_count += 1
        for j in sorted(remaining):
            self._create_track(detections[j])
        self.active_tracklets = [tr for tr in self.active_tracklets
                                 if tr.miss_count < self.track_timeout[tr.category]]

    def _create_track(self, det):
        # Omitting previous detections gives zero initial velocity/acceleration.
        tr = self.Track(self.history_len, det, dt=self.dt, track_id=self.next_track_id)
        self.next_track_id += 1
        self.active_tracklets.append(tr)
        return tr
