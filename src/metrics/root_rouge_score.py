from torchmetrics.text import ROUGEScore
from typing import Optional, Callable, Sequence, Any, Literal, Union


class RootROUGEScore(ROUGEScore):
    def __init__(
        self,
        gt_key: str,
        use_stemmer: bool = False,
        normalizer: Optional[Callable[[str], str]] = None,
        tokenizer: Optional[Callable[[str], Sequence[str]]] = None,
        accumulate: Literal["avg", "best"] = "best",
        rouge_keys: Union[str, tuple[str, ...]] = (
            "rouge1",
            "rouge2",
            "rougeL",
            "rougeLsum",
        ),
        **kwargs: Any,
    ):
        super().__init__(
            use_stemmer, normalizer, tokenizer, accumulate, rouge_keys, **kwargs
        )
        self.gt_key = gt_key

    def update(self, desc: list[str], target: dict) -> None:
        super().update(desc, target.get(self.gt_key))
    
    def compute(self) -> dict[str, float]:
        computed_metrics = super().compute()
        return {k: v.item() for k, v in computed_metrics.items()}
