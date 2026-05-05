from src.utils import StrEnum


class OutputFiles(StrEnum):
    NODE_TYPES = "node_types"
    NODES = "nodes"
    EDGE_TYPES = "edge_types"
    EDGES = "edges"
    LOG = "log"
    PARTITION = "partition"
    CONNECTED_COMOPONENTS = "connected_components"
    EDGE_COMPONENT_MAPPING = "edge_component_mapping"
    EDGE_INFO = "edge_info"
    NODE_TYPE_INFO = "node_type_info"
    EDGE_TYPE_INFO = "edge_type_info"
    NODE_INFO = "node_info"