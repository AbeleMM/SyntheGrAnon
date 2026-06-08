import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from joblib import delayed
from pydantic import BaseModel, ConfigDict, field_validator
from tqdm import tqdm

from anonymeter.evaluators import (
    EdgeInferenceEvaluator,
    EdgeLinkabilityEvaluator,
    EdgeSinglingOutEvaluator,
    InferenceEvaluator,
    LinkabilityEvaluator,
    NodeInferenceEvaluator,
    NodeLinkabilityEvaluator,
    NodeSinglingOutEvaluator,
    SinglingOutEvaluator,
)
from anonymeter.graph_consts import (
    AUX_COLS_INFERENCE_TYPE,
    AUX_COLS_LINKABILITY_TYPE,
    MEMORY,
    PARALLEL,
    STRUCT_ATTRS_TYPE,
)

MODEL_TYPE = str | tuple[float, float]

CL = 0.95

GRAPH_STRUCT_ATTR_FUNCS: dict[str, Callable] = {
    "n_nodes": nx.number_of_nodes,
    "avg_deg": (lambda g: g.number_of_edges() / g.number_of_nodes()),
    "avg_shortest_path_len": nx.average_shortest_path_length,
    "diameter": nx.diameter,
    "radius": nx.radius,
    "transitivity": nx.transitivity,
    "avg_clustering": nx.average_clustering,
    "deg_assortativity_coef": nx.degree_assortativity_coefficient,
}


def _noise_graph_data(
        graph: nx.Graph,
        p_edge_add=0.05, p_edge_remove=0.05,
        p_attr_flip=0.05, seed=0) -> nx.Graph:
    graph_noisy = graph.copy()
    rng = np.random.default_rng(seed=seed)

    # graph noise

    graph_noisy.add_edges_from(rng.choice(
        list(nx.complement(graph).edges), round(graph.number_of_edges() * p_edge_add),
        replace=False))
    graph_noisy.remove_edges_from(rng.choice(
        list(graph.edges), round(graph.number_of_edges() * p_edge_remove), replace=False))

    old_node_attrs = nx.get_node_attributes(graph_noisy, 'x')

    if old_node_attrs == {}:
        return graph_noisy

    n_nodes = graph_noisy.number_of_nodes()
    # Assume graph nodes are 0, 1, ..., N-1
    swap_nodes = rng.choice(n_nodes, size=round(p_attr_flip * n_nodes), replace=False)

    if isinstance(old_node_attrs[0], int):
        # Assume all node attributes are in 0, 1, ..., K
        n_classes = max(old_node_attrs.values()) + 1
        swaps = rng.integers(1, n_classes, len(swap_nodes))

        for i, node in enumerate(swap_nodes):
            graph_noisy.nodes[node]['x'] = (
                old_node_attrs[node] + swaps[i]) % n_classes
    else:
        # Assume all node attributes are equal-size 0|1 lists
        n_attr = len(old_node_attrs[0])
        swaps = rng.integers(2, size=(len(swap_nodes), n_attr)).tolist()

        for i, node in enumerate(swap_nodes):
            graph_noisy.nodes[node]['x'] = swaps[i]

    return graph_noisy


class GraphExpConf(ABC, BaseModel):
    dataset: str
    model: MODEL_TYPE
    n_attacks: int

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        ...


class GraphInferenceExpConf(GraphExpConf):
    aux_cols: list[str]
    secret: str

    @field_validator("aux_cols", mode="after")
    @classmethod
    def sort_aux_cols(cls, v):
        if isinstance(v, list):
            return sorted(v)

        return v

    @staticmethod
    def get_name() -> str:
        return "graph_inference"


class GraphLinkabilityExpConf(GraphExpConf):
    aux_cols: tuple[list[str], list[str]]
    n_neighbors: int

    @field_validator("aux_cols", mode="after")
    @classmethod
    def sort_aux_cols_inner(cls, v):
        if isinstance(v, tuple):
            return tuple(sorted(sorted(w) for w in v))

        return v

    @staticmethod
    def get_name() -> str:
        return "graph_linkability"


class GraphSinglingOutExpConf(GraphExpConf):
    n_cols: int

    @staticmethod
    def get_name() -> str:
        return "graph_singling-out"


class AttrExpConf(GraphExpConf):
    struct_attrs: STRUCT_ATTRS_TYPE

    @field_validator("struct_attrs", mode="after")
    @classmethod
    def sort_struct_attrs(cls, v):
        if isinstance(v, list):
            return sorted(v)

        return v


class NodeExpConf(AttrExpConf):
    embed: bool


class NodeInferenceExpConf(GraphInferenceExpConf, NodeExpConf):
    aux_cols: AUX_COLS_INFERENCE_TYPE

    @staticmethod
    def get_name() -> str:
        return "node_inference"


class NodeLinkabilityExpConf(GraphLinkabilityExpConf, NodeExpConf):
    aux_cols: AUX_COLS_LINKABILITY_TYPE

    @staticmethod
    def get_name() -> str:
        return "node_linkability"


class NodeSinglingOutExpConf(GraphSinglingOutExpConf, NodeExpConf):
    feat_attr: bool
    min_matching_syns: int

    @staticmethod
    def get_name() -> str:
        return "node_singling-out"


class EdgeExpConf(AttrExpConf):
    pass


class EdgeInferenceExpConf(GraphInferenceExpConf, EdgeExpConf):
    aux_cols: AUX_COLS_INFERENCE_TYPE

    @staticmethod
    def get_name() -> str:
        return "edge_inference"


class EdgeLinkabilityExpConf(GraphLinkabilityExpConf, EdgeExpConf):
    aux_cols: AUX_COLS_LINKABILITY_TYPE

    @staticmethod
    def get_name() -> str:
        return "edge_linkability"


class EdgeSinglingOutExpConf(GraphSinglingOutExpConf, EdgeExpConf):
    feat_attr: bool
    min_matching_syns: int

    @staticmethod
    def get_name() -> str:
        return "edge_singling-out"


class ExpRes(BaseModel):
    model_config = ConfigDict(frozen=True)

    @field_validator("risk_value", "risk_ci_lwr", "risk_ci_upr", mode="after")
    @classmethod
    def round_risk_floats(cls, v):
        return round(v, 4)

    gt_baseline: bool
    risk_value: float
    risk_ci_lwr: float
    risk_ci_upr: float


@MEMORY.cache
def _graph_to_row(g: nx.Graph) -> dict[str, int | float] | None:
    n_cc = nx.number_connected_components(g)

    if n_cc != 1:
        return None

    return {p: func(g) for p, func in GRAPH_STRUCT_ATTR_FUNCS.items()}


def _graphs_to_df(
        gs: list[nx.Graph],
        graph_struct_attr_props: Sequence[str] | None = None) -> pd.DataFrame:
    rows = PARALLEL(
        delayed(_graph_to_row)(g)
        for g in gs)
    df = pd.DataFrame(data=[row for row in rows if row is not None])

    if graph_struct_attr_props is None:
        return df

    return df[graph_struct_attr_props]


# TODO consider automatically detecting cols/fields containing tuples/lists
def _df_seq_to_str(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()

    for col in ["aux_cols", "struct_attrs"]:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype(str)

    return df_new


def _attack_graph_single(
        conf: GraphExpConf,
        ori: nx.Graph | list[nx.Graph],
        syn: list[nx.Graph],
        control: nx.Graph | list[nx.Graph]) -> ExpRes:

    # TODO re-extend if needed to take a specific number/subset of graphs

    evaluator_args: dict[str, Any]

    if isinstance(conf, AttrExpConf):
        evaluator_args = {"ori_g": ori, "control_g": control}
    else:
        assert isinstance(ori, list)
        assert isinstance(control, list)

        evaluator_args = {
            "ori": _graphs_to_df(ori),
            "control": _graphs_to_df(control)}

    evaluator_args |= conf.model_dump()

    if isinstance(conf, AttrExpConf):
        evaluator_args["syn_gs"] = syn
    else:
        evaluator_args["syn"] = _graphs_to_df(syn)

    del evaluator_args["dataset"]
    del evaluator_args["model"]

    evaluate_args = {}

    if type(conf) == NodeInferenceExpConf:
        evaluator_cls = NodeInferenceEvaluator
    elif type(conf) == NodeLinkabilityExpConf:
        evaluator_cls = NodeLinkabilityEvaluator
    elif type(conf) == NodeSinglingOutExpConf:
        evaluate_args |= {k: evaluator_args.pop(k) for k in ["n_cols", "min_matching_syns"]}
        evaluator_cls = NodeSinglingOutEvaluator
    elif type(conf) == EdgeInferenceExpConf:
        evaluator_cls = EdgeInferenceEvaluator
    elif type(conf) == EdgeLinkabilityExpConf:
        evaluator_cls = EdgeLinkabilityEvaluator
    elif type(conf) == EdgeSinglingOutExpConf:
        evaluate_args |= {k: evaluator_args.pop(k) for k in ["n_cols", "min_matching_syns"]}
        evaluator_cls = EdgeSinglingOutEvaluator
    #
    elif type(conf) == GraphInferenceExpConf:
        evaluator_cls = InferenceEvaluator
    elif type(conf) == GraphLinkabilityExpConf:
        evaluator_cls = LinkabilityEvaluator
    elif type(conf) == GraphSinglingOutExpConf:
        evaluate_args["mode"] = (
            "multivariate" if evaluator_args.pop("n_cols") > 1 else "univariate")
        evaluator_cls = SinglingOutEvaluator
    #
    else:
        raise AssertionError("Unknown configuration type.")

    evaluator = evaluator_cls(**evaluator_args)
    evaluator.evaluate(**evaluate_args)

    results = evaluator.results(confidence_level=CL)
    risk = results.risk()
    gt_baseline = results.baseline_rate.value < results.attack_rate.value
    ci_lwr, ci_upr = risk.ci

    return ExpRes(
        gt_baseline=gt_baseline,
        risk_value=risk.value,
        risk_ci_lwr=ci_lwr,
        risk_ci_upr=ci_upr)


def attack_graph_dataset(
        dataset: str,
        attack_type: type[GraphExpConf],
        models: Sequence[MODEL_TYPE],
        param_grid: dict[str, Sequence],
        rerun: bool) -> pd.DataFrame:
    data_path = Path(__file__).parent / "datasets" / dataset
    ds_fname = (
        f"{dataset}_"
        f"{'attr' if issubclass(attack_type, AttrExpConf) else 'unattr'}"
        ".pickle"
    )
    # TODO change real data, evaluators, and models to expect ori_g/control_g as list[nx.Graph]
    # then optionally check in `build_dfs` that ori only has one item
    # and remove logic for both cases from this script

    with open(data_path / ds_fname, "rb") as f:
        real_data: dict[str, nx.Graph | list[nx.Graph]] = pickle.load(f)

    ori_g = real_data["train"]
    control_g = real_data["control"]
    model_to_syn_gs: dict[MODEL_TYPE, list[nx.Graph]] = {}

    for model in models:
        if isinstance(model, str):
            model_data_path = data_path / f"{model}_{ds_fname}"

            if not model_data_path.exists():
                continue

            with open(model_data_path, "rb") as f:
                gs = pickle.load(f)
        else:
            assert isinstance(ori_g, nx.Graph)

            p_edge, p_attr = model
            gs = [_noise_graph_data(ori_g, p_edge, p_edge, p_attr)]

        model_to_syn_gs[model] = gs

    if not model_to_syn_gs:
        return pd.DataFrame()

    param_grid_with_meta = param_grid | {
        "dataset": [dataset], "model": list(model_to_syn_gs.keys())}
    param_grid_keys, param_grid_values = zip(*param_grid_with_meta.items())
    exp_configs: list[GraphExpConf] = [
        attack_type.model_validate(dict(zip(param_grid_keys, param_values)))
        for param_values in product(*param_grid_values)
    ]
    # create the DataFrame from GraphExpConf, where data is validated/normalized
    exp_configs_df = pd.DataFrame([c.model_dump() for c in exp_configs])
    res_path = Path(__file__).parent / "results" / f"{attack_type.get_name()}.csv"

    if res_path.exists() and not rerun:
        res_df = _df_seq_to_str(pd.read_csv(res_path))
        merge_df = _df_seq_to_str(exp_configs_df).merge(
            res_df,
            how="left",
            on=list(attack_type.model_fields),
            indicator=True
        )
        inds, *_ = np.nonzero(merge_df["_merge"] == "left_only")
        exp_configs = [exp_configs[i] for i in inds]

    res_df_rows: list[pd.DataFrame] = []

    for config in tqdm(exp_configs):
        exp_res = _attack_graph_single(config, ori_g, model_to_syn_gs[config.model], control_g)

        res_df_row = pd.DataFrame([config.model_dump() | exp_res.model_dump()])
        res_df_rows.append(res_df_row)
        res_df_row.to_csv(
            res_path,
            mode='w' if rerun else 'a',
            header=not res_path.exists(),
            index=False
        )

    return pd.concat(res_df_rows, ignore_index=True) if len(res_df_rows) > 0 else pd.DataFrame()
