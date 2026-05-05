from torch.utils.data import Dataset
from src.utils import load_json
from .utils import BatchKeys
from src.data.graph import ConnectedComponent
from src.data.loading.utils import RDFKeys


class EdgeTypeDataset(Dataset):
    def __init__(
        self,
        generated_edge_types: str,
        edge_type_info: str,
        connected_components: str,
        target_content_key: str = RDFKeys.LABEL,
    ) -> None:
        super().__init__()
        self.generated_edge_types = load_json(generated_edge_types)
        self.generated_edge_types = {
            x.get(BatchKeys.IDX): x for x in self.generated_edge_types
        }
        self.edge_type_info = load_json(edge_type_info)
        self.connected_components = load_json(connected_components)
        self.connected_components = [
            ConnectedComponent(**x) for x in self.connected_components
        ]
        self.connected_components = {
            x.component_id: x for x in self.connected_components
        }
        self.target_content_key = target_content_key
        self.keys = list(self.generated_edge_types.keys())
        assert self.target_content_key in [RDFKeys.LABEL, RDFKeys.DESCRIPTION]

    def _get_generated_content(self, edge_idx: int) -> str:
        return self.generated_edge_types[edge_idx].get(self.target_content_key, "-")

    def _get_connected_edge_types(self, edge_idx: int) -> list[str]:
        return list(map(lambda x: x[1], self.connected_components[edge_idx].edge_types))

    def _get_edge_type_content(self, edge_pid: str) -> str:
        return self.edge_type_info[edge_pid].get(self.target_content_key)

    def _get_edge_types(self, edge_idx: int) -> list[str]:
        return list(
            map(
                lambda x: self._get_edge_type_content(x),
                self._get_connected_edge_types(edge_idx),
            )
        )

    def __getitem__(self, index) -> dict:
        edge_type_id = self.keys[index]
        desc = self._get_generated_content(edge_type_id)
        targets = self._get_edge_types(edge_type_id)
        return {
            BatchKeys.DESC: desc,
            BatchKeys.TARGET: {
                BatchKeys.POSITIVE: targets,
            },
        }

    def __len__(self):
        return len(self.keys)

    def collate_fn(self, batch: list[dict]) -> dict:
        desc = [x.get(BatchKeys.DESC) for x in batch]
        targets = [x.get(BatchKeys.TARGET).get(BatchKeys.POSITIVE) for x in batch]
        return {
            BatchKeys.DESC: desc,
            BatchKeys.TARGET: {
                BatchKeys.POSITIVE: targets,
            },
        }
