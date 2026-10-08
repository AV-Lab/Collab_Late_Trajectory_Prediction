from importlib import import_module

from evaluation.plotting.plot_calibration import plot_calibration_curve
from evaluation.plotting.plot_uncertainty import plot_uncertainty_metrics


def __getattr__(name):
    # Runtime helpers import legacy stats paths. Load them only when requested,
    # so explicit uncertainty plotting does not initialize global destinations.
    if name in {"plot_delay", "plot_message_size", "plot_runtime"}:
        module = import_module(f"evaluation.plotting.{name}")
        function = getattr(module, name)
        globals()[name] = function
        return function
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def plot_runtime_stats():
    from evaluation.paths import DELAY_TMP_DIR, MESSAGE_SIZE_TMP_DIR, RUNTIME_TMP_DIR

    plots = {}

    if any(MESSAGE_SIZE_TMP_DIR.glob("*.txt")):
        plots["message_size"] = __getattr__("plot_message_size")()

    if any(DELAY_TMP_DIR.glob("*.txt")):
        plots["delay"] = __getattr__("plot_delay")()

    if any(RUNTIME_TMP_DIR.glob("*.csv")):
        plots["runtime"] = __getattr__("plot_runtime")()

    return plots


__all__ = [
    "plot_delay",
    "plot_calibration_curve",
    "plot_uncertainty_metrics",
    "plot_message_size",
    "plot_runtime",
    "plot_runtime_stats",
]
