from torch.utils.data import Dataset
from src.utils import load_json
from src.data.graph import ConnectedComponent
from .utils import BatchKeys, PromptKeys, RDFKeys


class EdgeTypeDataset(Dataset):
    def __init__(
        self,
        connected_components: str,
        edge_info: str,
        edge_types: str,
        node_summaries: str,
    ) -> None:
        self._init_connected_components(connected_components)
        self.edge_info = load_json(edge_info)
        self.edge_types = load_json(edge_types)
        self._init_node_summaries(node_summaries)

    def _init_node_summaries(self, node_summaries: str) -> None:
        self.node_summaries = load_json(node_summaries)
        self.node_summaries = {x.get(BatchKeys.IDX): x for x in self.node_summaries}

    def _init_connected_components(self, connected_components: str) -> None:
        self.connected_components = load_json(connected_components)
        self.connected_components = [
            ConnectedComponent(**component) for component in self.connected_components
        ]
        self.connected_components = {
            component.component_id: component for component in self.connected_components
        }

    def __len__(self) -> int:
        return len(self.connected_components)

    def get_rel_edge_type_str(self, edge: tuple[str, str, str]) -> str:
        head_pid, pid, tail_pid = edge
        return f"""
        {"-"*20}
        {PromptKeys.HEAD}:{self.get_pid_str(head_pid)}{PromptKeys.PID}: {pid}
        {PromptKeys.TAIL}:{self.get_pid_str(tail_pid)}{"-"*20}
        """

    def get_edge_type_str(self, edge_type: tuple[int, str, int]) -> str:
        head, pid, tail = edge_type
        return f"""
        {"-"*20}
        {PromptKeys.HEAD}:{self.get_sum_node_type_str(head)}{PromptKeys.PID}:{self.get_pid_str(pid)}{PromptKeys.TAIL}:{self.get_sum_node_type_str(tail)}{"-"*20}
        """

    def get_pid_str(self, pid: str) -> str:
        return f"""
        {PromptKeys.LABEL}: {self.edge_info.get(pid, dict()).get(RDFKeys.LABEL, "-")}
        {PromptKeys.DESCRIPTION}: {self.edge_info.get(pid, dict()).get(RDFKeys.DESCRIPTION, "-")}
        """

    def get_sum_node_type_str(self, idx: int) -> str:
        content = self.node_summaries.get(idx)
        return f"""
        {PromptKeys.LABEL}: {content.get(RDFKeys.LABEL)}
        {PromptKeys.DESCRIPTION}: {content.get(RDFKeys.DESCRIPTION)}
        """

    def __getitem__(self, index) -> dict:
        component = self.connected_components[index]
        edges = component.edges
        edge_types = component.edge_types
        sentence = f"""
        {PromptKeys.EDGE_TYPES}:{"".join(list(map(self.get_edge_type_str, edge_types)))}
        

        {PromptKeys.RELATIONS}:{"".join(list(map(self.get_rel_edge_type_str, edges)))}
        /no_think
        """
        return {
            BatchKeys.IDX: index,
            BatchKeys.QUERY: sentence,
        }
