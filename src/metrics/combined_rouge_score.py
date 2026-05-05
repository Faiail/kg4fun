from torchmetrics import Metric
from torchmetrics.text import ROUGEScore
import torch


class CombinedROUGEScore(Metric):
    def __init__(
        self,
        root_key: str,
        positive_key: str,
        negative_key: str,
        rouge_params: dict = dict(),
    ):
        super().__init__()
        self.root_key = root_key
        self.positive_key = positive_key
        self.negative_key = negative_key
        self.keys = {self.root_key, self.positive_key, self.negative_key}
        self.rouge = ROUGEScore(**rouge_params)
        for key in self.keys:
            for state in self.rouge._defaults.keys():
                self.add_state(f"{key}_{state}", [], dist_reduce_fx=None)

    def update_root_fn(self, desc: list[str], target: dict) -> None:
        gts = target.get(self.root_key)
        for desc_i, target_i in zip(desc, gts):
            self.rouge.update(desc_i, target_i)
            computed = self.rouge.compute()
            self.rouge.reset()
            for metric, val in computed.items():
                getattr(self, f"{self.root_key}_{metric}").append(val.nan_to_num(nan=0.0))

    def update_decision_fn(self, desc: list[str], target: dict, key: str) -> None:
        gts = target.get(key)
        for desc_i, target_i in zip(desc, gts):
            desc_rep = [desc_i * len(target_i)]
            self.rouge.update(desc_rep, target_i)
            computed = self.rouge.compute()
            self.rouge.reset()
            for metric, val in computed.items():
                getattr(self, f"{key}_{metric}").append(val.nan_to_num(nan=0.0))

    def update(self, desc: list[str], target: dict) -> None:
        for key in self.keys:
            (
                self.update_root_fn(desc, target)
                if key == self.root_key
                else self.update_decision_fn(desc, target, key)
            )

    def compute(self) -> dict:
        updated_metrics = {}
        output_metrics = {}
        for key in self.keys:
            for metric in self.rouge._defaults.keys():
                updated_metrics[f"{key}_{metric}"] = torch.stack(
                    getattr(self, f"{key}_{metric}")
                )

        for metric in self.rouge._defaults.keys():
            output_metrics[metric] = (
                (updated_metrics[f"{self.root_key}_{metric}"] + updated_metrics[f"{self.positive_key}_{metric}"])
                / 2
                - updated_metrics[f"{self.negative_key}_{metric}"]
            ).mean().detach().cpu().item()
        return output_metrics
        
