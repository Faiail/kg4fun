import click
from src.utils import load_ruamel

@click.group()
def main():
    pass

@main.command("generate-dataset")
@click.option("--parameters", help="Path to parameter file", required=True)
def geterate_dataset(parameters: str) -> None:
    from src.data import generate_fn
    generate_fn(load_ruamel(parameters))

@main.command("summarize")
@click.option("--parameters", help="Path to parameter file", required=True)
@click.option("--cls", help="Right class to execute for either node or edge types summarization", required=True)
def summarize(parameters: str, cls: str) -> None:
    from src.summarization import summarize_fn
    summarize_fn(load_ruamel(parameters), cls)

@main.command("evaluate")
@click.option("--parameters", help="Path to parameter file", required=True)
@click.option("--cls", help="Right class to execute for either node or edge types summarization", required=True)
def evaluate(parameters: str, cls: str) -> None:
    from src.evalutation import evaluation_fn
    evaluation_fn(load_ruamel(parameters), cls)