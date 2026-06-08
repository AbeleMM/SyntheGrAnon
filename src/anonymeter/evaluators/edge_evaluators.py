from collections.abc import Callable, Iterable
from typing import cast

import networkx as nx
import pandas as pd
from joblib import delayed

from anonymeter.evaluators.inference_evaluator import InferenceEvaluator
from anonymeter.evaluators.linkability_evaluator import LinkabilityEvaluator
from anonymeter.evaluators.node_evaluators import (
    NodeSinglingOutEvaluator,
    _graph_to_node_attr_df,
    rng,
)
from anonymeter.evaluators.singling_out_evaluator import SinglingOutEvaluator
from anonymeter.graph_consts import (
    AUX_COLS_INFERENCE_TYPE,
    AUX_COLS_LINKABILITY_TYPE,
    MEMORY,
    PARALLEL,
    STRUCT_ATTRS_TYPE,
)


def _make_tuple_generator(fn: Callable[[nx.Graph], dict]) -> Callable[[nx.Graph], Iterable[tuple]]:
    def wrapper(g: nx.Graph) -> Iterable[tuple]:
        for (u, v), p in fn(g).items():
            yield (u, v, p)

    return wrapper


def _make_edge_filterer(fn: Callable) -> Callable[[nx.Graph], Iterable[tuple]]:
    def wrapper(g: nx.Graph) -> Iterable[tuple]:
        return fn(g, ebunch=g.edges)

    return wrapper


def _edge_clustering_coefficient(g: nx.Graph) -> Iterable[tuple[int, int, float]]:
    for u, v in g.edges:
        if u == v:
            yield u, v, 0.
            continue

        u_neighbors = set(g.neighbors(u)) - {v}
        v_neighbors = set(g.neighbors(v)) - {u}
        denominator = min(len(u_neighbors), len(v_neighbors))
        score = 0. if denominator == 0 else len(u_neighbors & v_neighbors) / denominator
        yield u, v, score


EDGE_STRUCT_ATTR_FUNCS: dict[str, Callable] = {
    "edge_betweenness_centrality": _make_tuple_generator(nx.edge_betweenness_centrality),
    "jaccard_coefficient": _make_edge_filterer(nx.jaccard_coefficient),
    "adamic_adar_index": _make_edge_filterer(nx.adamic_adar_index),
    "preferential_attachment_score": _make_edge_filterer(nx.preferential_attachment),
    "edge_clustering_coefficient": _edge_clustering_coefficient,
}


def _sorted_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _graph_to_edge_attr_df(g: nx.Graph) -> pd.DataFrame:
    edge_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(
        (_sorted_edge(u, v) for u, v in g.edges), names=["node_0", "node_1"]))
    node_df = _graph_to_node_attr_df(g)
    edge_df = edge_df.join(node_df.add_suffix("-node_0"), on="node_0")
    edge_df = edge_df.join(node_df.add_suffix("-node_1"), on="node_1")

    return edge_df


@MEMORY.cache
def _graph_to_edge_struct_df(g: nx.Graph) -> pd.DataFrame:
    edge_df = pd.DataFrame({
        f"_{struct_attr}": {
            _sorted_edge(u, v): p for (u, v, p) in struct_attr_func(g)}
        for struct_attr, struct_attr_func in EDGE_STRUCT_ATTR_FUNCS.items()})
    edge_df.index = pd.MultiIndex.from_tuples(
        edge_df.index, names=["node_0", "node_1"])

    return edge_df


def _graph_to_edge_df(
        g: nx.Graph,
        feat_attr: bool,
        struct_attrs: STRUCT_ATTRS_TYPE) -> pd.DataFrame:
    attr_df = _graph_to_edge_attr_df(g) if feat_attr else pd.DataFrame()
    struct_cols = [
        f"_{attr}" for attr in (
            EDGE_STRUCT_ATTR_FUNCS.keys()
            if struct_attrs == "all" else struct_attrs)]

    if struct_cols:
        struct_df = _graph_to_edge_struct_df(g)[struct_cols]
    else:
        struct_df = pd.DataFrame()

    # avoid join containing an empty DataFrame
    if attr_df.empty:
        return struct_df

    if struct_df.empty:
        return attr_df

    return attr_df.join(struct_df, how="outer").reset_index(drop=True)


def build_edge_dfs(
        ori_g: nx.Graph,
        syn_gs: list[nx.Graph],
        control_g: nx.Graph | None,
        feat_attr: bool,
        struct_attrs: STRUCT_ATTRS_TYPE,
        add_syn_id=False) -> tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    ori_df = _graph_to_edge_df(ori_g, feat_attr, struct_attrs)
    syn_dfs = cast(list[pd.DataFrame], PARALLEL(
        delayed(_graph_to_edge_df)(syn_g, feat_attr, struct_attrs) for syn_g in syn_gs))

    if control_g is None:
        control_df = None
    else:
        control_df = _graph_to_edge_df(control_g, feat_attr, struct_attrs)
        control_df = control_df.astype(ori_df.dtypes)

    if add_syn_id:
        for i, syn in enumerate(syn_dfs):
            syn["graph_id"] = i

    syn_df = pd.concat(syn_dfs, ignore_index=True).astype(ori_df.dtypes)

    return ori_df, syn_df, control_df


class EdgeInferenceEvaluator(InferenceEvaluator):
    def __init__(
        self,
        ori_g: nx.Graph,
        syn_gs: list[nx.Graph],
        aux_cols: AUX_COLS_INFERENCE_TYPE,
        secret: str,
        regression: bool = False,
        n_attacks: int = 500,
        control_g: nx.Graph | None = None,
        struct_attrs: STRUCT_ATTRS_TYPE = "all",
    ):
        ori, syn, control = build_edge_dfs(
            ori_g, syn_gs, control_g, True, struct_attrs)

        if aux_cols == "feat":
            aux_cols_ = [col for col in syn.columns if not col.startswith('_')]
        elif aux_cols == "struct":
            aux_cols_ = [col for col in syn.columns if col.startswith('_')]
        elif aux_cols == "all":
            aux_cols_ = list(syn.columns)
        else:
            aux_cols_ = aux_cols

        try:
            secret_aux_cols_ind = aux_cols_.index(secret)
            aux_cols_ = aux_cols_[:secret_aux_cols_ind] + aux_cols_[secret_aux_cols_ind + 1:]
        except ValueError:
            pass

        super().__init__(ori, syn, aux_cols_, secret, regression, n_attacks, control)


class EdgeLinkabilityEvaluator(LinkabilityEvaluator):
    def __init__(
        self,
        ori_g: nx.Graph,
        syn_gs: list[nx.Graph],
        aux_cols: AUX_COLS_LINKABILITY_TYPE,
        n_attacks: int = 500,
        n_neighbors: int = 1,
        control_g: nx.Graph | None = None,
        struct_attrs: STRUCT_ATTRS_TYPE = "all",
    ):
        ori, syn, control = build_edge_dfs(
            ori_g, syn_gs, control_g, aux_cols != "struct-struct", struct_attrs)

        if aux_cols == "feat-struct":
            aux_cols_l, aux_cols_r = [], []

            for c in syn.columns:
                (aux_cols_r if c.startswith('_') else aux_cols_l).append(c)
            aux_cols_ = (aux_cols_l, aux_cols_r)
        elif isinstance(aux_cols, str):
            if aux_cols == "feat-feat":
                cols = [c for c in syn.columns if not c.startswith('_')]
            elif aux_cols == "struct-struct":
                cols = [c for c in syn.columns if c.startswith('_')]
            else:
                cols = list(syn.columns)

            rng.shuffle(cols)
            cutoff = len(cols) // 2
            aux_cols_l = cols[cutoff:]
            aux_cols_r = cols[:cutoff]
            aux_cols_ = (aux_cols_l, aux_cols_r)
        else:
            aux_cols_ = aux_cols

        super().__init__(ori, syn, aux_cols_, n_attacks, n_neighbors, control)


class EdgeSinglingOutEvaluator(NodeSinglingOutEvaluator):
    def __init__(
        self,
        ori_g: nx.Graph,
        syn_gs: list[nx.Graph],
        n_attacks: int = 500,
        control_g: nx.Graph | None = None,
        feat_attr=True,
        struct_attrs: STRUCT_ATTRS_TYPE = "all",
    ):
        ori, syn, control = build_edge_dfs(
            ori_g, syn_gs, control_g, feat_attr, struct_attrs, add_syn_id=True)

        SinglingOutEvaluator.__init__(self, ori, syn, n_attacks, 0, control)
