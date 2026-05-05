from src.utils import StrEnum, load_json, save_json
import pandas as pd
import random
from tqdm import tqdm
import os
from SPARQLWrapper import SPARQLWrapper, JSON
from itertools import product, permutations
import networkx as nx
from .utils import (
    OutputFiles,
    DatasetGeneratorParameters,
    DatasetFiles,
    LOGColNames,
    Target,
)
from .graph import Edge, EdgeType
from .graph import (
    EdgeGraph,
    RelEdgeTypeCollection,
    ConnectedComponentCollection,
)


class DatasetGenerator:
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        self.init()

    def init(self) -> None:
        self._init_general()
        self._init_dataset()
        self._init_wrapper()

    def _init_general(self) -> None:
        general_parameters = self.parameters.get(DatasetGeneratorParameters.GENERAL, {})
        # in dir
        self.input_dir = general_parameters.get(DatasetGeneratorParameters.IN_DIR, "./")
        # out dir
        self.output_dir = general_parameters.get(
            DatasetGeneratorParameters.OUT_DIR, "./"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.edge_dir = f"{self.output_dir}/{OutputFiles.EDGES}"
        os.makedirs(self.edge_dir, exist_ok=True)
        self.partition_dir = f"{self.edge_dir}/{OutputFiles.PARTITION}"
        os.makedirs(self.partition_dir, exist_ok=True)
        self.pbar = general_parameters.get(DatasetGeneratorParameters.PBAR, False)
        self.limit = general_parameters.get(DatasetGeneratorParameters.LIMIT, None)
        self.save_every = general_parameters.get(
            DatasetGeneratorParameters.SAVE_SIZE, None
        )
        self.chunk_node_size = general_parameters.get(
            DatasetGeneratorParameters.CHUNK_NODE_SIZE, 100
        )

    def _init_dataset(self) -> None:
        dataset_parameters = self.parameters.get(DatasetGeneratorParameters.DATASET, {})
        # dataset name (e.g., dataset1, dataset2, dataset1-2, etc.)
        self.dataset_name = dataset_parameters.get(DatasetGeneratorParameters.NAME)
        # if to take the filtered version of the dataset
        self.filtered = dataset_parameters.get(
            DatasetGeneratorParameters.FILTERED, False
        )
        self.filtered_postfix = "_filtered" if self.filtered else ""
        # if to limit the seed nodes
        self.seed_size = dataset_parameters.get(DatasetGeneratorParameters.SIZE, None)
        # attributes that will be loaded lazily
        self._decisions = None
        self._seeds = None
        self.etypes_relations = dataset_parameters.get(
            DatasetGeneratorParameters.ETYPE_REL
        )
        self.etypes_relations = " ".join(
            map(lambda x: f"wdt:{x}", self.etypes_relations)
        )

    def _init_wrapper(self) -> None:
        wrapper_parameters = self.parameters.get(DatasetGeneratorParameters.WRAPPER, {})
        url = wrapper_parameters.get(DatasetGeneratorParameters.URL)
        self.wrapper = SPARQLWrapper(url)

    @property
    def seeds(self):
        if self._seeds is not None:
            return self._seeds
        fname = f"{self.input_dir}/{self.dataset_name}{self.filtered_postfix}.csv"
        self._seeds = pd.read_csv(fname, header=None)[0].unique().tolist()
        if self.seed_size is not None:
            assert self.seed_size <= len(
                self._seeds
            ), f"Requested seed size is larger than available seeds. Maximum seed size for this dataset: {len(self._seeds)}"
            self._seeds = random.sample(self._seeds, self.seed_size)
        return self._seeds

    @property
    def decisions(self):
        if self._decisions is not None:
            return self._decisions
        fname = f"{self.input_dir}/{self.dataset_name}_{DatasetFiles.GOLD_DECISIONS.value}{self.filtered_postfix}.csv"
        self._decisions = pd.read_csv(fname)
        self._decisions = self._decisions[
            self._decisions["from"].isin(self.seeds)
        ].set_index("from")
        return self._decisions

    def get_pbar(self, population, **kwargs):
        return self.population if not self.pbar else tqdm(population, **kwargs)

    def update_pbar(self, pbar, **kwargs):
        if not self.pbar:
            return
        pbar.set_postfix(**kwargs)

    def get_node_types(self) -> None:
        out_classes = dict()
        for ix, seed_qid in self.get_pbar(
            list(enumerate(self.seeds)), desc="Getting node types"
        ):
            positive_decisions, negative_decisions = self.get_single_node_type(seed_qid)
            out_classes[seed_qid] = {
                "positive": positive_decisions,
                "negative": negative_decisions,
                "cls_idx": ix,
            }
        return out_classes

    def get_single_node_type(self, seed_qid: str) -> None:
        seed_decisions = self.decisions.loc[seed_qid]
        if isinstance(seed_decisions, pd.Series):
            seed_decisions = seed_decisions.to_frame().T
        positive_decisions = seed_decisions[seed_decisions["target"] == 1]
        if isinstance(positive_decisions, pd.Series):
            positive_decisions = positive_decisions.to_frame().T
        positive_decisions = positive_decisions[["QID", "label", "depth"]]
        negative_decisions = seed_decisions[seed_decisions["target"] == 0]
        if isinstance(negative_decisions, pd.Series):
            negative_decisions = negative_decisions.to_frame().T
        negative_decisions = negative_decisions[["QID", "label", "depth"]]
        return (
            positive_decisions.to_dict(orient="records"),
            negative_decisions.to_dict(orient="records"),
        )

    def get_instances(self, cls_info) -> None:
        dataset = list()
        for seed_qid, cls_data in self.get_pbar(
            cls_info.items(), desc="Getting instances"
        ):
            positive_qids = list(
                map(lambda x: f"wd:{x}", [x["QID"] for x in cls_data["positive"]])
            )
            negative_qids = list(
                map(lambda x: f"wd:{x}", [x["QID"] for x in cls_data["negative"]])
            )
            cls_idx = cls_data["cls_idx"]
            dataset += self.get_cls_instances(positive_qids, negative_qids, cls_idx)
        return dataset

    def get_cls_instances(self, positive_qids, negative_qids, cls_idx) -> dict:
        cls_dataset = list()
        cls_dataset += self.get_positive_instances(
            positive_qids=positive_qids,
            negative_qids=negative_qids,
            cls_idx=cls_idx,
        )

        cls_dataset += self.get_negative_instances(
            positive_qids=negative_qids,
            negative_qids=positive_qids,
        )
        return cls_dataset

    def get_positive_instances(
        self, positive_qids, negative_qids, cls_idx
    ) -> list[dict]:
        resultset = self.exe_query(
            self.get_node_instance_query(
                positive_qids=positive_qids,
                negative_qids=negative_qids,
            )
        )
        for row_dict in resultset:
            row_dict.update({"cls_idx": cls_idx})
        return resultset

    def get_negative_instances(self, positive_qids, negative_qids) -> list[dict]:
        if len(positive_qids) == 0:
            return list()
        resultset = self.exe_query(
            self.get_node_instance_query(
                positive_qids=positive_qids,
                negative_qids=negative_qids,
            )
        )
        for row_dict in resultset:
            row_dict.update({"cls_idx": -1})
        return resultset

    def get_base_query(self):
        return """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX schema: <http://schema.org/>
        """

    def get_node_instance_query(self, positive_qids, negative_qids) -> str:
        positive_set = " ".join(positive_qids)
        negative_set = " ".join(negative_qids)
        query = self.get_base_query()
        query += f"""SELECT (STRAFTER(STR(?item), "http://www.wikidata.org/entity/") AS ?qid) ?itemLabel WHERE
        {{
            VALUES ?includeClass {{ {positive_set} }} 
            ?item wdt:P31 ?includeClass.
        """
        if len(negative_qids) > 0:
            query += f"""
            FILTER NOT EXISTS {{
                VALUES ?excludeClass {{ {negative_set}}}
                ?item wdt:P31 ?excludeClass.
            }}
            """
        query += f"""
        ?item rdfs:label ?itemLabel.
        FILTER (LANG(?itemLabel) = "en")
        }} limit {self.limit}
        """
        return query

    def exe_query(self, query: str) -> dict:
        self.wrapper.setQuery(query)
        self.wrapper.setReturnFormat(JSON)
        resultset = self.wrapper.query().convert()
        # filter results
        headers = resultset["head"]["vars"]
        results = resultset["results"].get("bindings", list())
        results = (
            results
            if isinstance(results, list) and len(results) > 0 and results[0] is not None
            else list()
        )
        return [
            {header: result.get(header, {}).get("value", "-") for header in headers}
            for result in results
        ]

    def get_pos_edge_query_for_node_types(
        self, kb: dict, head_node_t: int, tail_node_t: int
    ) -> list:
        node_head = (x["qid"] for x in kb if x["cls_idx"] == head_node_t)
        node_head = list(map(lambda x: f"wd:{x}", node_head))
        node_tail = (x["qid"] for x in kb if x["cls_idx"] == tail_node_t)
        node_tail = list(map(lambda x: f"wd:{x}", node_tail))
        queries = []
        for head_chunk in range(0, len(node_head), self.chunk_node_size):
            head_chunk_list = node_head[head_chunk : head_chunk + self.chunk_node_size]
            for tail_chunk in range(0, len(node_tail), self.chunk_node_size):
                tail_chunk_list = node_tail[
                    tail_chunk : tail_chunk + self.chunk_node_size
                ]
                query = self.get_base_query()
                query += f"""
                SELECT DISTINCT (STRAFTER(STR(?s), "http://www.wikidata.org/entity/") AS ?head_qid) (STRAFTER(STR(?o), "http://www.wikidata.org/entity/") AS ?tail_qid) (STRAFTER(STR(?propEntity), "http://www.wikidata.org/entity/") AS ?pid) ?rel_label WHERE {{
                ?s ?rel ?o. 
                VALUES ?s {{ {" ".join(head_chunk_list)} }}.
                VALUES ?o {{ {" ".join(tail_chunk_list)} }}.
                BIND( IRI(REPLACE(STR(?rel),
                            "^http://www.wikidata.org/prop/direct/",
                            "http://www.wikidata.org/entity/")) AS ?propEntity)

                    ?propEntity rdfs:label ?rel_label.
                    FILTER (LANG(?rel_label) = "en")
                }}
                """
                queries.append(query)
        return queries

    def get_log(self):
        log_fname = f"{self.edge_dir}/{OutputFiles.LOG}.pkl"
        if not os.path.exists(log_fname):
            return None
        return pd.read_pickle(log_fname)

    def get_saved_edges(self):
        log = self.get_log()
        return set(log[LOGColNames.EDGE].tolist()) if log is not None else set()

    def save_edges(self, edges: set, save_count: int, log_cls_etypes) -> set:
        # save the cls in the log
        # save the edges in the next partition
        if save_count != self.save_every:
            return edges, save_count, log_cls_etypes
        partition = len(
            list(
                filter(lambda x: x.startswith("part_"), os.listdir(self.partition_dir))
            )
        )
        print(f"Saving to partition {partition}")
        # save to the log
        log = self.get_log()
        local_log = pd.DataFrame(
            {
                LOGColNames.EDGE: list(log_cls_etypes),
                LOGColNames.PARTITION: [partition] * len(log_cls_etypes),
            }
        )
        updated_log = (
            local_log if log is None else pd.concat([log, local_log], ignore_index=True)
        )
        updated_log.to_pickle(f"{self.edge_dir}/{OutputFiles.LOG}.pkl")
        print("Log updated")
        # save edges in the partition
        save_json(
            f"{self.partition_dir}/part_{partition}.json",
            [obj.to_dict() for obj in list(edges)],
        )
        print(f"Partition n.{partition} saved")
        return set(), 0, set()

    def save_etypes(self):
        etypes = set()
        for partition in os.listdir(self.partition_dir):
            edges = load_json(f"{self.partition_dir}/{partition}")
            etypes = etypes.union(set(EdgeType(**e["edge_type"]) for e in edges))
        save_json(
            f"{self.edge_dir}/{OutputFiles.EDGE_TYPES}.json",
            [obj.to_dict() for obj in list(etypes)],
        )

    def get_and_save_edges(self, node_types, nodes):
        save_count = 0
        edges = set()
        edges_pop = filter(lambda edge: edge[0] != edge[1], product(nodes, repeat=2))
        num_edges = len(nodes) * (len(nodes) - 1)
        class_types = list(set(x["cls_idx"] for x in node_types.values())) + [-1]
        # instanciate the log
        saved_edges = self.get_saved_edges()
        edges_pop = product(class_types, repeat=2)
        num_edges = len(class_types) ** 2
        bar = self.get_pbar(edges_pop, total=num_edges, desc="Edges")
        cls_etype = set()
        for head_node, tail_node in bar:
            if (head_node, tail_node) in saved_edges:
                print(f"Edge ({head_node}, {tail_node}) already saved. Skipping")
                save_count = 0
                continue
            cls_etype.add((head_node, tail_node))
            save_count += 1
            queries = self.get_pos_edge_query_for_node_types(
                kb=nodes, head_node_t=head_node, tail_node_t=tail_node
            )
            for query in queries:
                result_set = self.exe_query(query)
                for row in result_set:
                    edge = self.build_edge(
                        head_node=row["head_qid"],
                        head_cls=head_node,
                        tail_cls=tail_node,
                        tail_node=row["tail_qid"],
                        pid=row["pid"],
                        label=row["rel_label"],
                    )
                    edges.add(edge)
                    self.update_pbar(bar, edges=len(edges))
            edges, save_count, cls_etype = self.save_edges(edges, save_count, cls_etype)
        if save_count != 0:
            save_count = self.save_every
            self.save_edges(edges, save_count, cls_etype)
        self.save_etypes()

    def build_edge(
        self,
        head_node,
        head_cls,
        tail_node,
        tail_cls,
        pid,
        label,
    ) -> Edge:
        target = (
            Target.KEEP if all(x != -1 for x in [head_cls, tail_cls]) else Target.PRUNE
        )
        etype = EdgeType(
            pid=pid,
            label=label,
            head_cls=head_cls,
            tail_cls=tail_cls,
            target=target,
        )
        edge = Edge(edge_type=etype, head_qid=head_node, tail_qid=tail_node)
        return edge

    def get_edge_query(self, head_qid, tail_qid) -> str:
        query = self.get_base_query()
        query += f"""
        SELECT DISTINCT (STRAFTER(STR(?propEntity), "http://www.wikidata.org/entity/") AS ?pid) ?relLabel WHERE {{
            ?s ?rel ?o.
            VALUES ?s {{ wd:{head_qid} }}.
            VALUES ?o {{ wd:{tail_qid} }}.
            
            BIND( IRI(REPLACE(STR(?rel),
                      "^http://www.wikidata.org/prop/direct/",
                      "http://www.wikidata.org/entity/")) AS ?propEntity)

            ?propEntity rdfs:label ?relLabel.
            FILTER (LANG(?relLabel) = "en")
        }}
        """
        return query

    def get_etype_rel_query(self, head_pid, tail_pid) -> str:
        query = self.get_base_query()
        query += f"""
        SELECT DISTINCT (STRAFTER(STR(?headProp), "http://www.wikidata.org/entity/") AS ?headPid)
        (STRAFTER(STR(?relProp), "http://www.wikidata.org/entity/") AS ?relPid)
        ?relLabel
        (STRAFTER(STR(?tailProp), "http://www.wikidata.org/entity/") AS ?tailPid) WHERE {{
            ?s ?rel ?o.
            VALUES ?s {{ wd:{head_pid} }}.
            VALUES ?o {{ wd:{tail_pid} }}.
            VALUES ?rel {{ {self.etypes_relations} }}.

            BIND( IRI(REPLACE(STR(?s),
                      "^http://www.wikidata.org/prop/direct/",
                      "http://www.wikidata.org/entity/")) AS ?headProp)
            
            BIND( IRI(REPLACE(STR(?o),
                      "^http://www.wikidata.org/prop/direct/",
                      "http://www.wikidata.org/entity/")) AS ?tailProp)
            
            BIND( IRI(REPLACE(STR(?rel),
                      "^http://www.wikidata.org/prop/direct/",
                      "http://www.wikidata.org/entity/")) AS ?relProp)
            
            ?relProp rdfs:label ?relLabel.
            FILTER (LANG(?relLabel) = "en")

        }}
        """
        return query

    def get_edge_types(
        self,
        partition_id: int,
        head_node_cls: int,
        tail_node_cls: int,
    ) -> list[str]:
        edges = load_json(f"{self.partition_dir}/part_{partition_id}.json")
        return list(
            set(
                map(
                    lambda x: x["edge_type"]["pid"],
                    filter(
                        lambda x: x["edge_type"]["head_cls"] == head_node_cls
                        and x["edge_type"]["tail_cls"] == tail_node_cls,
                        edges,
                    ),
                )
            )
        )

    def get_edge_types_graph(
        self,
        partition_id: int,
        head_cls: int,
        tail_cls: int,
    ) -> EdgeGraph:
        edge_types = self.get_edge_types(partition_id, head_cls, tail_cls)
        edge_types = {pid: idx for (idx, pid) in enumerate(edge_types)}
        pairs = permutations(edge_types.keys(), 2)
        num_pairs = len(edge_types.keys()) * (len(edge_types.keys()) - 1)
        triplets = list()
        edge_rel_types = RelEdgeTypeCollection()
        bar = self.get_pbar(
            population=pairs,
            total=num_pairs,
            desc="Connections among edge types",
            disable=True,
        )
        for head, tail in bar:
            query = self.get_etype_rel_query(head_pid=head, tail_pid=tail)
            result_set = self.exe_query(query)
            for row in result_set:
                edge_rel_types.add(row["relPid"], row["relLabel"])
                triplets.append(
                    (
                        edge_types.get(row["headPid"]),
                        edge_rel_types.get(row["relPid"]),
                        edge_types.get(row["tailPid"]),
                    )
                )
            self.update_pbar(bar, edges=len(triplets))
        graph = EdgeGraph(
            edge_mapping={v: k for k, v in edge_rel_types.mapping.items()},
            edges=triplets,
            nodes=list(edge_types.values()),
            node_mapping={v: k for k, v in edge_types.items()},
            edge_descriptions={
                v: edge_rel_types.descriptions[k]
                for k, v in edge_rel_types.mapping.items()
            },
        )
        return graph

    def compute_connected_components(self, edge_graph: EdgeGraph) -> list:
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(edge_graph.nodes)
        nx_graph.add_edges_from(map(lambda x: (x[0], x[2]), edge_graph.edges))
        return list(nx.connected_components(nx_graph))

    def save_connected_components(
        self,
        connected_components: ConnectedComponentCollection,
    ) -> None:
        rev_connected_components_mapping = dict()
        for cc in connected_components.components:
            rev_connected_components_mapping.update({e: cc.component_id for e in cc.edge_types})
        save_json(
            f"{self.edge_dir}/{OutputFiles.CONNECTED_COMOPONENTS}.json",
            [obj.to_dict() for obj in connected_components.components],
        )
        self.save_component_mapping(rev_connected_components_mapping)

    def save_component_mapping(self, mapping: dict[tuple, int]) -> None:
        mapping_to_save = [{"edge_type": k, "component_id": v} for k, v in mapping.items()]
        save_json(f"{self.edge_dir}/{OutputFiles.EDGE_COMPONENT_MAPPING}.json", mapping_to_save)
    
    def load_component_mapping(self) -> dict[tuple, int]:
        mapping = load_json(f"{self.edge_dir}/{OutputFiles.EDGE_COMPONENT_MAPPING}.json")
        return {tuple(m["edge_type"]): m["component_id"] for m in mapping}

    def get_description_query(self, qid: str) -> str:
        query = self.get_base_query()
        query += f"""
        SELECT ?itemLabel ?itemComment ?itemDescription WHERE {{
            OPTIONAL{{
                wd:{qid} rdfs:label ?itemLabel.
                FILTER (LANG(?itemLabel) = "en")
            }}
            OPTIONAL {{
                wd:{qid} rdfs:comment ?itemComment.
                FILTER (LANG(?itemComment) = "en")
            }}
            OPTIONAL {{
                wd:{qid} schema:description ?itemDescription.
                FILTER (LANG(?itemDescription) = "en")
            }}
        }}
        """
        return query

    def compute_and_save_edge_type_desc(self) -> None:
        if os.path.exists(f"{self.output_dir}/{OutputFiles.EDGE_INFO}.json"):
            return
        etypes = [
            EdgeType(**e).pid
            for e in load_json(f"{self.edge_dir}/{OutputFiles.EDGE_TYPES}.json")
        ]
        edge_descriptions = dict()
        bar = self.get_pbar(etypes, total=len(etypes), desc="Getting edge type info")
        for pid in bar:
            result_set = self.exe_query(self.get_description_query(pid))
            desc = result_set[0] if len(result_set) > 0 else dict()
            edge_descriptions.update({pid: desc})
        save_json(f"{self.edge_dir}/{OutputFiles.EDGE_INFO}.json", edge_descriptions)

    def comptute_and_save_connected_links_components(self) -> None:
        self.log = self.get_log()
        true_log = self.log[
            self.log[LOGColNames.EDGE].apply(lambda x: -1 not in x)
        ].sort_values(by=LOGColNames.EDGE)
        population = true_log.iterrows()
        bar = self.get_pbar(
            population=population,
            total=len(true_log),
            desc="Edge types",
        )
        connected_components = ConnectedComponentCollection()
        for _, edge_series in bar:
            edge_graph = self.get_edge_types_graph(
                edge_series[LOGColNames.PARTITION],
                *edge_series[LOGColNames.EDGE],
            )
            cc = self.compute_connected_components(edge_graph)
            connected_components.add(edge_graph, cc, *edge_series[LOGColNames.EDGE])
        self.save_connected_components(connected_components)

    def compute_node_types(self) -> None:
        if os.path.exists(f"{self.output_dir}/{OutputFiles.NODE_TYPES}.json"):
            return
        cls_info = self.get_node_types()
        save_json(rf"{self.output_dir}/{OutputFiles.NODE_TYPES}.json", cls_info)

    def compute_node_instances(self) -> None:
        if os.path.exists(f"{self.output_dir}/{OutputFiles.NODES}.json"):
            return
        cls_info = load_json(f"{self.output_dir}/{OutputFiles.NODE_TYPES}.json")
        instances = self.get_instances(cls_info)
        save_json(rf"{self.output_dir}/{OutputFiles.NODES}.json", instances)

    def compute_and_save_node_type_desc(self) -> None:
        if os.path.exists(f"{self.output_dir}/{OutputFiles.NODE_TYPE_INFO}.json"):
            return
        node_types = load_json(f"{self.output_dir}/{OutputFiles.NODE_TYPES}.json")
        node_descriptions = dict()
        seeds = list(node_types.keys())
        bar = self.get_pbar(
            seeds, total=len(node_types), desc="Getting seed node descriptions"
        )
        for qid in bar:
            result_set = self.exe_query(self.get_description_query(qid))
            desc = result_set[0] if len(result_set) > 0 else dict()
            node_descriptions.update({qid: desc})
        decisions = node_types.values()
        bar = self.get_pbar(
            population=decisions,
            total=len(node_types),
            desc="Getting class node descriptions",
        )
        for seed_val in bar:
            for positive_qid in map(lambda x: x["QID"], seed_val.get("positive", [])):
                result_set = self.exe_query(self.get_description_query(positive_qid))
                desc = result_set[0] if len(result_set) > 0 else dict()
                node_descriptions.update({positive_qid: desc})
            for negative_qid in map(lambda x: x["QID"], seed_val.get("negative", [])):
                result_set = self.exe_query(self.get_description_query(negative_qid))
                desc = result_set[0] if len(result_set) > 0 else dict()
                node_descriptions.update({negative_qid: desc})
        save_json(
            f"{self.output_dir}/{OutputFiles.NODE_TYPE_INFO}.json", node_descriptions
        )

    def compute_and_save_node_instance_desc(self) -> None:
        if os.path.exists(f"{self.output_dir}/{OutputFiles.NODE_INFO}.json"):
            return
        nodes = load_json(f"{self.output_dir}/{OutputFiles.NODES}.json")
        node_descriptions = dict()
        bar = self.get_pbar(
            nodes, total=len(nodes), desc="Getting node instance descriptions"
        )
        for node in bar:
            qid = node["qid"]
            result_set = self.exe_query(self.get_description_query(qid))
            desc = result_set[0] if len(result_set) > 0 else dict()
            node_descriptions.update({qid: desc})
        save_json(f"{self.output_dir}/{OutputFiles.NODE_INFO}.json", node_descriptions)

    def process_nodes(self):
        self.compute_node_types()
        # get nodes involved (from Wikidata)
        self.compute_node_instances()
        self.compute_and_save_node_type_desc()
        self.compute_and_save_node_instance_desc()

    def process_edges(self):
        # load node types
        node_types = load_json(f"{self.output_dir}/{OutputFiles.NODE_TYPES}.json")
        nodes = load_json(f"{self.output_dir}/{OutputFiles.NODES}.json")
        self.get_and_save_edges(node_types=node_types, nodes=nodes)
        self.comptute_and_save_connected_links_components()
        self.compute_and_save_edge_type_desc()

    def __call__(self) -> None:
        self.process_nodes()
        self.process_edges()
