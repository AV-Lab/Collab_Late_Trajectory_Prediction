"""Evaluation API; importing output paths does not initialize the evaluator."""

__all__ = ["Evaluator"]


def __getattr__(name):
    if name == "Evaluator":
        from evaluation.evaluator import Evaluator
        globals()[name] = Evaluator
        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
