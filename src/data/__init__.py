from .dataset_generator import DatasetGenerator


def generate_fn(parameters: dict) -> None:
    return DatasetGenerator(parameters)()