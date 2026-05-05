from ruamel.yaml import YAML
from pathlib import Path
import json


def load_ruamel(path: str, typ: str = "safe") -> dict:
    yaml = YAML(typ=typ)
    return yaml.load(Path(path))


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        content = json.load(f)
    return content


def save_json(path: str, content: dict) -> None:
    with open(path, "w+") as f:
        json.dump(content, f)