from .summarization_run import SummarizationRun


def summarize_fn(parameters: dict, cls: str) -> None:
    globals()[cls](parameters)()