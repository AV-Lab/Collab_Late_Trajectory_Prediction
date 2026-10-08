"""Track-layout adapter for the TrajZoo runtime API."""

import numpy as np

from intelligent_vehicles.predictors.base_wrapper import BaseWrapper


class TrackWrapper(BaseWrapper):
    layout = "track"

    def _position(self, record):
        return [float(record.x), float(record.y)]

    def format_input(self, tracklets):
        return [
            [
                self._position(record)
                for record in tracklet["tracklet"]
            ]
            for tracklet in tracklets
        ]

    def predict(self, formatted_input):
        output = self.predictor.predict(formatted_input)
        trajectories = np.asarray(output["trajectories"])
        covariances = np.asarray(output["covariances"])
        timestamps = [
            float(f"{step / self.fps:.3f}")
            for step in range(1, self.prediction_horizon + 1)
        ]

        mean_trajs = []
        cov_trajs = []
        for index in range(len(formatted_input)):
            means = trajectories[index, 0]
            covs = covariances[index, 0]
            mean_trajs.append({
                timestamp: means[step].tolist()
                for step, timestamp in enumerate(timestamps)
            })
            cov_trajs.append({
                timestamp: covs[step].tolist()
                for step, timestamp in enumerate(timestamps)
            })
        return mean_trajs, cov_trajs
