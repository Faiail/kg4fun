from src.utils import StrEnum


class DatasetFiles(StrEnum):
    SEED_QIDS = ""  # EMPTY NAMES
    GOLD_DECISIONS = "gold_decisions"
    LABELS = "labels"


class LOGColNames(StrEnum):
    EDGE = "edge"
    PARTITION = "partition"


class Target(StrEnum):
    KEEP = 1
    PRUNE = 0