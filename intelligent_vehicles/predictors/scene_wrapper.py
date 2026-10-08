"""Scene-layout adapter for the TrajZoo runtime API."""

import numpy as np

from intelligent_vehicles.predictors.base_wrapper import BaseWrapper


class SceneWrapper(BaseWrapper):
    layout = "scene"

    def __init__(self, predictor_config):
        self._sample_index = 0
        super().__init__(predictor_config)

    def _position(self, record):
        return [float(record.x), float(record.y)]

    def format_input(self, tracklets):
        self._sample_index += 1
        return {
            "sample_id": f"coltp-{self._sample_index}",
            "scenario_id": None,
            "scene_id": None,
            "objects": [
                {
                    "track_id": tracklet["id"],
                    "agent_class": tracklet["category"],
                    "target_agent": True,
                    "use_as_past": True,
                    "use_as_context": True,
                    "past_traj": [
                        self._position(record)
                        for record in tracklet["tracklet"]
                    ],
                    "current_state": None,
                    "future_traj": None,
                }
                for tracklet in tracklets
            ],
        }

    def predict(self, formatted_input):
        output = self.predictor.predict(formatted_input)
        trajectories = np.asarray(output["trajectories"])
        covariances = np.asarray(output["covariances"])
        input_ids = [obj["track_id"] for obj in formatted_input["objects"]]
        output_index = {
            track_id: index
            for index, track_id in enumerate(output["track_ids"])
        }
        timestamps = [
            float(f"{step / self.fps:.3f}")
            for step in range(1, self.prediction_horizon + 1)
        ]

        mean_trajs = []
        cov_trajs = []
        for track_id in input_ids:
            index = output_index[track_id]
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
