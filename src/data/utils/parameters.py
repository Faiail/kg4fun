from src.utils import StrEnum


class DatasetGeneratorParameters(StrEnum):
    GENERAL = "general"
    IN_DIR = "in_dir"
    OUT_DIR = "out_dir"
    DATASET = "dataset"
    FILTERED = "filtered"
    SIZE = "size"
    NAME = "name"
    PBAR = "pbar"
    WRAPPER = "wrapper"
    URL = "url"
    LIMIT = "limit"
    SAVE_SIZE = "save_size"
    CHUNK_NODE_SIZE = "chunk_node_size"
    ETYPE_REL = "edge_type_relations"
