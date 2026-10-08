"""Base interface for COLTP predictor wrappers."""

from abc import ABC, abstractmethod

from predictor import Predictor


class BaseWrapper(ABC):
    def __init__(self, predictor_config):
        self.device = predictor_config["device"]
        self.fps = float(predictor_config["fps"])
        self.observed_past = int(predictor_config["observed_past"])
        self.prediction_horizon = int(predictor_config["prediction_horizon"])
        checkpoint = predictor_config.get("checkpoint")

        if checkpoint:
            self.predictor = Predictor.from_checkpoint(
                checkpoint_path=checkpoint,
                device=self.device,
            )
        else:
            self.predictor = Predictor.from_config(
                model_name=predictor_config["config"],
                obs_len=self.observed_past,
                pred_len=self.prediction_horizon,
                fps=self.fps,
                params=predictor_config.get("params_file_path"),
                device=self.device,
            )

    @abstractmethod
    def format_input(self, tracklets):
        raise NotImplementedError

    @abstractmethod
    def predict(self, formatted_input):
        raise NotImplementedError
