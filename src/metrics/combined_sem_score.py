from torchmetrics import Metric
from sentence_transformers import SentenceTransformer
import torch


class CombinedSEMScore(Metric):
    def __init__(
        self,
        model_name,
        device,
        root_key: str,
        positive_key: str,
        negative_key: str,
    ) -> None:
        super().__init__()
        self._device = device
        self.model = SentenceTransformer(model_name, device=self._device)
        self.root_key = root_key
        self.positive_key = positive_key
        self.negative_key = negative_key
        self.add_state("root_scores", default=[], dist_reduce_fx="cat")
        self.add_state("positive_scores", default=[], dist_reduce_fx="cat")
        self.add_state("negative_scores", default=[], dist_reduce_fx="cat")

    def update_root_fn(self, desc: torch.Tensor, roots: list[str]) -> torch.Tensor:
        roots_embs = self.model.encode(
            roots, convert_to_tensor=True, device=self.device
        )
        return torch.nn.functional.cosine_similarity(desc, roots_embs).cpu()

    def update_decision_fn(
        self, desc: torch.Tensor, targets: list[list[str]]
    ) -> torch.Tensor:
        flat_targets = [t for sublist in targets for t in sublist]

        target_emb = self.model.encode(
            flat_targets, convert_to_tensor=True, device=self._device
        )

        sizes = [len(t) for t in targets]
        expanded_desc = torch.repeat_interleave(
            desc, torch.tensor(sizes, device=self._device), dim=0
        )

        sims = torch.nn.functional.cosine_similarity(expanded_desc, target_emb)

        # 4. split e media per sample
        split_sims = torch.split(sims, sizes)
        per_sample_scores = torch.stack([s.mean() for s in split_sims])

        return per_sample_scores.cpu().nan_to_num(nan=0.0)

    def update(self, desc: list[str], target: dict) -> None:
        desc_emb = self.model.encode(desc, convert_to_tensor=True, device=self.device)
        roots = target.get(self.root_key)
        positives = target.get(self.positive_key)
        negatives = target.get(self.negative_key)
        self.root_scores.append(self.update_root_fn(desc_emb, roots))
        self.positive_scores.append(self.update_decision_fn(desc_emb, positives))
        self.negative_scores.append(self.update_decision_fn(desc_emb, negatives))

    def compute(self) -> float:
        return (
            (torch.cat(self.root_scores) + torch.cat(self.positive_scores))/2 - torch.cat(self.negative_scores)
        ).mean().item()
