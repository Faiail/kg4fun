import src.metrics as metric_pkg
import src.data.metrics as data_pkg
from src.metrics import MetricCollection
from torch.utils.data import DataLoader
from .utils import EvaluationKeys
from torchmetrics import Metric
import os
from tqdm import tqdm
from src.utils import save_json


class EvaluationRun:
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        self.init()

    def init(self) -> None:
        self._init_general()
        self._init_loader()
        self._init_metrics()

    def _init_general(self) -> None:
        general_parameters = self.parameters.get(EvaluationKeys.GENERAL)
        self.out_dir = general_parameters.get(EvaluationKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.pbar = general_parameters.get(EvaluationKeys.PBAR, False)
        self.out_fname = general_parameters.get(EvaluationKeys.OUT_FNAME)

    def _init_loader(self) -> None:
        dataset_parameters = self.parameters.get(EvaluationKeys.DATA)
        loader_parameters = self.parameters.get(EvaluationKeys.LOADER, dict())
        dataset_name = dataset_parameters.get(EvaluationKeys.NAME)
        dataset_config = dataset_parameters.get(EvaluationKeys.CONFIG)
        dataset = getattr(data_pkg, dataset_name)(**dataset_config)
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        self.dataloader = DataLoader(
            dataset, collate_fn=collate_fn, **loader_parameters
        )

    def _init_single_metric(self, metric: dict) -> Metric:
        metric_name = metric.get(EvaluationKeys.NAME)
        metric_cfg = metric.get(EvaluationKeys.CONFIG, dict())
        return getattr(metric_pkg, metric_name)(**metric_cfg)

    def _init_metrics(self) -> None:
        metric_parameters = self.parameters.get(EvaluationKeys.METRICS)
        self.metrics = MetricCollection(
            {
                metric_name: self._init_single_metric(metric_cfg)
                for metric_name, metric_cfg in metric_parameters.items()
            }
        )

    def get_pbar(self, population, **kwargs):
        return population if not self.pbar else tqdm(population, **kwargs)

    def launch(self) -> None:
        loader = self.get_pbar(
            self.dataloader,
            desc=f"Computing metrics for {self.dataloader.dataset.target_content_key}",
        )

        for batch in loader:
            self.metrics.update(**batch)
        metrics_computed = self.metrics.compute()
        self.metrics.reset()
        save_json(f"{self.out_dir}/{self.out_fname}", metrics_computed)
