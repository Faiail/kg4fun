from src.utils import StrEnum


class PromptKeys(StrEnum):
    LABEL = "Item label"
    DESCRIPTION = "Item description"
    DEPTH = "Depth from ROOT"
    ROOT = "ROOT"
    POSITIVE = "Positive examples"
    NEGATIVE = "Negative examples"
    HEAD = "Head Node Type"
    TAIL = "Tail Node Type"
    CONTENT = "content"
    PID = "PID"
    EDGE_TYPES = "Edge types"
    RELATIONS = "Edge Relations"
    THINKING = "thinking"