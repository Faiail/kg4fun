from .root_sem_score import RootSEMScore
import torch


class DecisionSEMScore(RootSEMScore):
    def update_fn(self, desc: list[str], targets: list[list[str]]) -> torch.Tensor:
        # 1. flatten targets
        flat_targets = [t for sublist in targets for t in sublist]

        # 2. encode
        desc_emb = self.model.encode(desc, convert_to_tensor=True, device=self._device)
        target_emb = self.model.encode(
            flat_targets, convert_to_tensor=True, device=self._device
        )

        # 3. compute similarities
        # espandi preds per matchare i target flatten
        sizes = [len(t) for t in targets]
        expanded_desc = torch.repeat_interleave(
            desc_emb, torch.tensor(sizes, device=self._device), dim=0
        )

        sims = torch.nn.functional.cosine_similarity(expanded_desc, target_emb)

        # 4. split e media per sample
        split_sims = torch.split(sims, sizes)
        per_sample_scores = torch.stack([s.mean() for s in split_sims])

        return per_sample_scores.cpu().nan_to_num(nan=0.0)
