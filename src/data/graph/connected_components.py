from dataclasses import asdict, dataclass


@dataclass
class EdgeGraph:
    nodes: list[int]  # real pids
    edges: list[tuple[int, int, int]]  # edges among edges
    edge_mapping: dict[int, str]  # 0 -> p123
    node_mapping: dict[int, str]  # 0 -> p123
    edge_descriptions: dict[int, str]  # 0 -> related_property


class RelEdgeTypeCollection:
    def __init__(self) -> None:
        self.population = set()
        self.mapping = dict()
        self.descriptions = dict()
        self.num_types = 0

    def add(self, pid: str, label: str) -> None:
        if pid in self.mapping:
            return
        self.population.add((pid, label))
        self.mapping[pid] = self.num_types
        self.descriptions[pid] = label
        self.num_types += 1

    def get(self, pid: str) -> int:
        return self.mapping.get(pid)


@dataclass(frozen=True)
class ConnectedComponent:
    component_id: int
    edge_types: list[
        tuple[int, str, int]
    ]  # TODO: update with complete edge type -> head + pid + tail
    edges: list[tuple[str, str, str]]

    def to_dict(self):
        return asdict(self)


class ConnectedComponentCollection:
    def __init__(self) -> None:
        self.components = list()

    def add(
        self,
        edge_graph: EdgeGraph,
        cc: list[set],
        head_cls: int,
        tail_cls: int,
    ) -> None:
        for connected_edges in cc:
            component = self.create_component(
                connected_edges,
                edge_graph,
                head_cls,
                tail_cls,
            )
            self.components.append(component)

    def create_component(
        self,
        component: set[int],
        edge_graph: EdgeGraph,
        head_cls: int,
        tail_cls: int,
    ) -> ConnectedComponent:
        edge_types = list(map(lambda x: edge_graph.node_mapping[x], component))
        edge_types = list(map(lambda x: (head_cls, x, tail_cls), edge_types))
        edges = list(
            filter(lambda x: x[0] in component or x[2] in component, edge_graph.edges)
        )
        edges = list(
            map(
                lambda x: (
                    edge_graph.node_mapping[x[0]],
                    edge_graph.edge_descriptions[x[1]],
                    edge_graph.node_mapping[x[2]],
                ),
                edges,
            )
        )
        conn_comp = ConnectedComponent(
            component_id=len(self.components),
            edge_types=edge_types,
            edges=edges,
        )
        return conn_comp
