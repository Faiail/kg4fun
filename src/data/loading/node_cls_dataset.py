from torch.utils.data import Dataset
from .utils.prompt_keys import PromptKeys
from src.utils import load_json
from collections import defaultdict
from .utils import RDFKeys, BatchKeys


class NodeCLSDataset(Dataset):
    def __init__(self, node_type_info: str, node_types: str) -> None:
        self.node_type_info = load_json(node_type_info)
        self.node_types = load_json(node_types)
        self.dataset = defaultdict(dict)
        for qid, data in self.node_types.items():
            cls_idx = data.get(RDFKeys.CLS_IDX)
            self.dataset[cls_idx][qid] = data

    def __len__(self) -> int:
        return len(self.dataset)

    def get_qid_str(self, qid, depth: int = None) -> str:
        label = self.node_type_info[qid].get(RDFKeys.LABEL, "-")
        description = self.node_type_info[qid].get(RDFKeys.DESCRIPTION, "-")
        label_str = f"{PromptKeys.LABEL}: {label}\n"
        description_str = (
            f"{PromptKeys.DESCRIPTION}: {description}\n"
        )
        depth_str = f"{PromptKeys.DEPTH}: {depth}" if depth is not None else ""
        return f"{label_str}{description_str}{depth_str}"

    def __getitem__(self, idx: int) -> dict:
        root_node = list(self.dataset[idx].keys())[0]
        positive_examples = self.node_types.get(root_node).get(RDFKeys.POSITIVE, [])
        positive_sentences = list(
            map(
                lambda x: self.get_qid_str(x.get(RDFKeys.QID), x.get(RDFKeys.DEPTH)),
                positive_examples,
            )
        )
        negative_examples = self.node_types.get(root_node).get(RDFKeys.NEGATIVE, [])
        negative_sentences = list(
            map(
                lambda x: self.get_qid_str(x.get(RDFKeys.QID), x.get(RDFKeys.DEPTH)),
                negative_examples,
            )
        )
        sentence = f"""
        {PromptKeys.ROOT}:\n{self.get_qid_str(root_node)}
        

        {PromptKeys.POSITIVE}:\n{'\n\n'.join(positive_sentences)} 
        """
        sentence += (
            f"\n\n{PromptKeys.NEGATIVE}:\n{'\n\n'.join(negative_sentences)}"
            if len(negative_sentences) > 0
            else ""
        )
        sentence += "\n/no_think"
        return {BatchKeys.IDX: idx, BatchKeys.QUERY: sentence}
