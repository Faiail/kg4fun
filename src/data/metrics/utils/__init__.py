from src.utils import StrEnum


class BatchKeys(StrEnum):
    ROOT = "root"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    DESC = "desc"
    TARGET = "target"
    IDX = "idx"