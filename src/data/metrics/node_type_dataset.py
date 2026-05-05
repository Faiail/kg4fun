from torch.utils.data import Dataset
from src.utils import load_json
from src.data.loading.utils import RDFKeys
from .utils import BatchKeys


class NodeTypeDataset(Dataset):
    def __init__(
        self,
        generated_dataset: str,
        node_info: str,
        decisions: str,
        target_content_key: str = RDFKeys.LABEL,  # label or description
    ) -> None:
        super().__init__()
        self.generated_dataset = load_json(generated_dataset)
        self.generated_dataset = {
            x.get(BatchKeys.IDX): x for x in self.generated_dataset
        }
        self.node_info = load_json(node_info)
        self.decisions = load_json(decisions)
        self.roots = list(self.decisions.keys())
        self.target_content_key = target_content_key
        assert self.target_content_key in [RDFKeys.LABEL, RDFKeys.DESCRIPTION]

    def _get_content(self, qid) -> str:
        return self.node_info.get(qid).get(self.target_content_key, "-")

    def _get_decisions(self, decisions: list[dict]) -> list[str]:
        return [self._get_content(x.get(RDFKeys.QID)) for x in decisions]

    def _get_generated_content(self, cls_idx: int) -> str:
        return self.generated_dataset.get(cls_idx).get(self.target_content_key)

    def __getitem__(self, index) -> dict:
        root_qid = self.roots[index]
        decision = self.decisions.get(root_qid)
        root_content = self._get_content(root_qid)
        positives = self._get_decisions(
            self.decisions.get(root_qid).get(BatchKeys.POSITIVE)
        )
        negatives = self._get_decisions(
            self.decisions.get(root_qid).get(BatchKeys.NEGATIVE)
        )

        generated = self._get_generated_content(decision.get(RDFKeys.CLS_IDX))
        return {
            BatchKeys.DESC: generated,
            BatchKeys.TARGET: {
                BatchKeys.ROOT: root_content,
                BatchKeys.POSITIVE: positives,
                BatchKeys.NEGATIVE: negatives,
            },
        }

    def __len__(self):
        return len(self.roots)

    def collate_fn(self, batch: list) -> dict:
        desc = [x.get(BatchKeys.DESC) for x in batch]
        targets = [x.get(BatchKeys.TARGET) for x in batch]
        roots = [x.get(BatchKeys.ROOT) for x in targets]
        positives = [x.get(BatchKeys.POSITIVE, []) for x in targets]
        negatives = [x.get(BatchKeys.NEGATIVE, []) for x in targets]
        return {
            BatchKeys.DESC: desc,
            BatchKeys.TARGET: {
                BatchKeys.ROOT: roots,
                BatchKeys.POSITIVE: positives,
                BatchKeys.NEGATIVE: negatives,
            },
        }
