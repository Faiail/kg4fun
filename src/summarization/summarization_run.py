import src.data.loading as dataset_pkg
import src.models as model_pkg
from .utils import SummarizerKeys
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy
from src.data.loading.utils import BatchKeys, RDFKeys
from src.models.utils import LLMKeys
from src.utils import save_json


class SummarizationRun:
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        self.init()

    def init(self) -> None:
        self._init_general()
        self._init_dataloader()
        self._init_model()

    def _init_general(self) -> None:
        general_parameters = self.parameters.get(SummarizerKeys.GENERAL, dict())
        self.out_dir = general_parameters.get(SummarizerKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.pbar = general_parameters.get(SummarizerKeys.PBAR, False)
        self.out_fname = general_parameters.get(SummarizerKeys.OUT_FNAME)

    def get_pbar(self, population, **kwargs):
        return population if not self.pbar else tqdm(population, **kwargs)

    def _init_dataloader(self) -> None:
        dataset_parameters = self.parameters.get(SummarizerKeys.DATA)
        dataset_name = dataset_parameters.get(SummarizerKeys.NAME)
        dataset_config = dataset_parameters.get(SummarizerKeys.CONFIG, dict())
        dataset = getattr(dataset_pkg, dataset_name)(**dataset_config)
        dataloader_parameters = self.parameters.get(SummarizerKeys.LOADER, dict())
        self.dataloader = DataLoader(dataset=dataset, **dataloader_parameters)

    def get_shots(self, shots_params: dict[int, str]) -> dict:
        return {self.dataloader.dataset[k][BatchKeys.QUERY]: fr"{v}" for k, v in shots_params.items()}

    def _init_model(self) -> None:
        model_parameters = deepcopy(self.parameters.get(SummarizerKeys.MODEL, dict()))
        model_name = model_parameters.get(SummarizerKeys.NAME)
        model_config = model_parameters.get(SummarizerKeys.CONFIG, dict())
        shots = model_config.pop(SummarizerKeys.SHOTS, dict())
        shots = self.get_shots(shots)
        model_config[SummarizerKeys.SHOTS] = shots
        self.model = getattr(model_pkg, model_name)(**model_config)

    def format(self, generated_content: dict) -> dict:
        out_dict = deepcopy(generated_content)
        content = out_dict.pop(LLMKeys.CONTENT)
        thinking = out_dict.pop(LLMKeys.THINKING, [""] * len(content))
        splitted = [entry.rsplit("\n", 1) for entry in content]
        splitted = [entry if len(entry) == 2 else (entry[0], "") for entry in splitted]
        splitted = [(entry[0].rsplit("\n", 1)[-1], entry[1]) for entry in splitted]
        labels, descriptions = zip(*splitted)
        out_dict[RDFKeys.LABEL] = [label.strip() for label in labels]
        out_dict[RDFKeys.DESCRIPTION] = [
            description.strip() for description in descriptions
        ]
        return [
            {
                LLMKeys.THINKING: thought,
                RDFKeys.LABEL: label,
                RDFKeys.DESCRIPTION: description,
            }
            for (thought, label, description) in zip(thinking, labels, descriptions)
        ]

    def __call__(self) -> None:
        bar = self.get_pbar(self.dataloader, desc="Summarizing")
        infos = list()
        for batch in bar:
            idxs = batch.pop(BatchKeys.IDX)
            out = self.model(batch[BatchKeys.QUERY])
            formatted = self.format(out)
            infos.extend(
                {BatchKeys.IDX: idx, **formatted}
                for idx, formatted in zip(idxs.tolist(), formatted)
            )
        save_json(f"{self.out_dir}/{self.out_fname}", infos)
