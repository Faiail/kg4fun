from torchmetrics import Metric
import torch
from sentence_transformers import SentenceTransformer


class SEMScore(Metric):
    def __init__(self, model_name: str, device: str) -> None:
        super().__init__()
        self._device = device
        self.model = SentenceTransformer(model_name, device=self._device)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
    
    def update_fn(self, desc: list[str], target: list[str]) -> torch.Tensor:
        # compute description embedding
        desc_emb = self.model.encode(desc, convert_to_tensor=True, device=self._device)
        # compute gt embedding
        target_emb = self.model.encode(target, convert_to_tensor=True, device=self._device)
        # compute cosine similatity
        return torch.nn.functional.cosine_similarity(desc_emb, target_emb).cpu()
    
    def update(self, desc: list[str], target: list[str]) -> None:
        self.scores.append(self.update_fn(desc, target))

    def compute(self) -> float:
        return torch.cat(self.scores).mean().item()