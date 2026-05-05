from .evaluation_run import EvaluationRun


def evaluation_fn(parameters: dict, cls: str) -> None:
    globals()[cls](parameters).launch()