from .sem_score import SEMScore


class RootSEMScore(SEMScore):
    def __init__(self, model_name, device, gt_key: str) -> None:
        super().__init__(model_name, device)
        self.gt_key = gt_key

    def update(self, desc: list[str], target: list[dict]) -> None:
        return super().update(desc, target.get(self.gt_key))
