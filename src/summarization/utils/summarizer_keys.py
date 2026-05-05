from src.utils import StrEnum


class SummarizerKeys(StrEnum):
    GENERAL = "general"
    IN_DIR = "in_dir"
    OUT_DIR = "out_dir"
    MODEL = "model"
    NAME = "name"
    CONFIG = "config"
    PBAR = "pbar"
    VARS = "vars"
    TARGET_DATA = "target_data"
    NODE_SUMMARY_DIR = "node_summary_dir"
    DATA = "data"
    LOADER = "loader"
    SHOTS = "shots"
    OUT_FNAME = "out_fname"