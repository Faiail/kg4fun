from torchmetrics import Metric
from torchmetrics.text import ROUGEScore
from torchmetrics.functional.text.rouge import _rouge_score_compute
import re

def camel_to_spaces(s):
    return re.sub(r"(?<!^)(?=[A-Z])", " ", s)


class DecisionROUGEScore(Metric):
    def __init__(self, gt_key: str, rouge_params: dict = dict()):
        super().__init__()
        self.gt_key = gt_key
        self.rouge = ROUGEScore(**rouge_params)
        # replicating the states of the rouge for our same score
        for state in self.rouge._defaults.keys():
            self.add_state(state, [], dist_reduce_fx=None)

    def update(self, desc: list[str], target: dict) -> None:
        gts = target.get(self.gt_key)
        desc = list(map(camel_to_spaces, desc))
        # do the computation separately
        for desc_i, target_i in zip(desc, gts):
            # replicate the description for all the targets
            desc_rep = [desc_i * len(target_i)]
            # leverage the rouge engine to compute the rouge score
            self.rouge.update(desc_rep, target_i)
            computed = self.rouge.compute()
            self.rouge.reset()
            # save the scores for
            for metric, val in computed.items():
                getattr(self, metric).append(val.nan_to_num(nan=0.0))

    def compute(self) -> dict[str, float]:
        update_output = {}
        for rouge_key in self.rouge.rouge_keys_values:
            for tp in ["fmeasure", "precision", "recall"]:
                update_output[f"rouge{rouge_key}_{tp}"] = getattr(
                    self, f"rouge{rouge_key}_{tp}"
                )

        computed = _rouge_score_compute(update_output)
        return {k: v.detach().item() for k, v in computed.items()}
