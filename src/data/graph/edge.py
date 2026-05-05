from dataclasses import asdict, dataclass
from src.data.utils import Target


@dataclass(frozen=True)
class PID:
    pid: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class EdgeType:
    pid: PID
    label: str
    head_cls: int
    tail_cls: int
    target: Target

    def __eq__(self, value):
        if not isinstance(value, EdgeType):
            raise NotImplementedError()
        return (
            self.pid == value.pid
            and self.head_cls == value.head_cls
            and self.tail_cls == value.tail_cls
        )

    def __hash__(self):
        return hash(self.pid) + (hash(self.head_cls) - hash(self.tail_cls))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Edge:
    edge_type: EdgeType
    head_qid: str
    tail_qid: str

    def to_dict(self):
        return asdict(self)
