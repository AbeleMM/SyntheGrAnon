import logging
from functools import partial
from typing import Callable, cast

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as nng
from joblib import delayed
from torch import Tensor
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from anonymeter.evaluators import (
    InferenceEvaluator,
    LinkabilityEvaluator,
    SinglingOutEvaluator,
)
from anonymeter.evaluators.singling_out_evaluator import (
    UniqueSinglingOutQueries,
    _convert_polars_dtype,
    _evaluate_queries_and_return_successful,
    _query_from_record,
    _random_query,
    fit_correction_term,
)
from anonymeter.graph_consts import (
    AUX_COLS_INFERENCE_TYPE,
    AUX_COLS_LINKABILITY_TYPE,
    MEMORY,
    PARALLEL,
    STRUCT_ATTRS_TYPE,
)
from anonymeter.stats.confidence import EvaluationResults

NODE_STRUCT_ATTR_FUNCS: dict[str, Callable] = {
    "avg_neigh_deg": nx.average_neighbor_degree,
    "clus_coef": nx.clustering,
    "deg_centrality": nx.degree_centrality,
    "square_clustering": nx.square_clustering,
    "eigenvec_centrality": partial(nx.eigenvector_centrality, max_iter=500),
    "betweenness_centrality": nx.betweenness_centrality,
}

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

rng = np.random.default_rng()
logger = logging.getLogger(__name__)


def _graph_df_to_data(g: nx.Graph, df: pd.DataFrame) -> Data:
    g_attr = g.copy(as_view=True)
    df_ = df.copy(deep=False)
    # TODO remove number normalization if redundant
    df_num = df_.select_dtypes(include="number")
    df_[df_num.columns] = (df_num - df_num.min()) / (df_num.max() - df_num.min())
    df_ = pd.get_dummies(df_)
    nx.set_node_attributes(g_attr, df_.to_dict(orient="index"))
    data: Data = from_networkx(g_attr, group_node_attrs="all").to(DEVICE)

    return data


@MEMORY.cache
def _graph_to_node_attr_df(g: nx.Graph) -> pd.DataFrame:
    node_attrs = nx.get_node_attributes(g, 'x')

    if not node_attrs:
        return pd.DataFrame()

    example_node_attr = next(iter(node_attrs.values()))

    if isinstance(example_node_attr, list):
        return pd.DataFrame.from_dict(
            node_attrs, orient="index",
            columns=[f"attr_{i}" for i in range(len(example_node_attr))],
            # dtype="boolean"
        )

    return pd.Series(node_attrs, dtype="category", name="attr_0").to_frame()


@MEMORY.cache
def _graph_to_node_struct_df(g: nx.Graph) -> pd.DataFrame:
    return pd.DataFrame({
        f"_{struct_attr}": struct_attr_func(g)
        for struct_attr, struct_attr_func in NODE_STRUCT_ATTR_FUNCS.items()})


def _graph_to_node_df(
        g: nx.Graph,
        feat_attr: bool,
        struct_attrs: STRUCT_ATTRS_TYPE) -> pd.DataFrame:
    attr_df = _graph_to_node_attr_df(g) if feat_attr else pd.DataFrame()
    struct_cols = [
        f"_{attr}" for attr in (
            NODE_STRUCT_ATTR_FUNCS.keys()
            if struct_attrs == "all" else struct_attrs)]

    if struct_cols:
        struct_df = _graph_to_node_struct_df(g)[struct_cols]
    else:
        struct_df = pd.DataFrame()

    # avoid join containing an empty DataFrame
    if attr_df.empty:
        return struct_df

    if struct_df.empty:
        return attr_df

    return attr_df.join(struct_df, how="outer")


class EmbeddingModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int, out_channels: int):
        super().__init__()
        self.conv1 = nng.GATConv(in_channels, hidden_channels, heads=heads, dropout=0.5)
        self.conv2 = nng.GATConv(
            hidden_channels * heads, hidden_channels, heads=heads, dropout=0.5)
        self.conv3 = nng.GATConv(hidden_channels * heads, out_channels, heads=1, dropout=0.5)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x, inplace=True)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x, inplace=True)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.normalize(x, p=2, dim=1)

        return x


def train_embedding_model(
        gs_dfs: list[tuple[nx.Graph, pd.DataFrame]],
        bs=64, n_epochs=1_000, lr=1e-3) -> EmbeddingModel:
    dl = DataLoader(
        [_graph_df_to_data(g, df) for g, df in gs_dfs],
        batch_size=bs,
        shuffle=True
    )
    data: Data
    model = EmbeddingModel(next(iter(dl)).num_node_features, 16, 8, 4).to(DEVICE)
    opt = AdamW(model.parameters(), lr=lr)
    model.train()
    loss_fn = nn.TripletMarginLoss(margin=1., p=2)
    losses = [0.] * n_epochs

    for i in range(n_epochs):
        for data in dl:
            data.edge_index = cast(Tensor, data.edge_index)
            opt.zero_grad()
            z = model(data.x, data.edge_index)
            pairs = torch.cdist(z, z, p=2)
            pairs[*data.edge_index] = torch.inf
            pairs.fill_diagonal_(torch.inf)
            anchor, pos = data.edge_index
            neg = torch.argmin(pairs[anchor], dim=-1)
            loss = loss_fn(z[anchor], z[pos], z[neg])
            loss.backward()
            opt.step()
            losses[i] += loss.item()

        losses[i] /= len(dl)

    # print([round(l, 3) for l in losses])

    return model


def build_embed_df(g: nx.Graph, df: pd.DataFrame, embedding_model: EmbeddingModel) -> pd.DataFrame:
    d: dict[str, list[float]] = {}
    data = _graph_df_to_data(g, df)
    embedding: npt.NDArray = embedding_model(data.x, data.edge_index).numpy(force=True)

    for i, dim in enumerate(embedding.T):
        d[f"dim_{i}"] = dim

    return pd.DataFrame(data=d, index=list(g.nodes))


def build_node_dfs(
        ori_g: nx.Graph,
        syn_gs: list[nx.Graph],
        control_g: nx.Graph | None,
        feat_attr: bool,
        struct_attrs: STRUCT_ATTRS_TYPE,
        embed: bool,
        add_syn_id=False) -> tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    ori_df = _graph_to_node_df(ori_g, feat_attr, struct_attrs)
    syn_dfs = cast(list[pd.DataFrame], PARALLEL(
        delayed(_graph_to_node_df)(syn_g, feat_attr, struct_attrs) for syn_g in syn_gs))

    if embed:
        syn_gds_syns = [(syn_g, syn) for syn_g, syn in zip(syn_gs, syn_dfs)]
        embedding_model = train_embedding_model(syn_gds_syns)
        ori_df = build_embed_df(ori_g, ori_df, embedding_model)
        syn_dfs = [build_embed_df(syn_g, syn, embedding_model) for syn_g, syn in syn_gds_syns]

    # if ori.duplicated().any():
    #     raise AssertionError("Original DataFrame has duplicate rows.")

    # if any(syn.duplicated().any() for syn in syns):
    #     raise AssertionError("A synthetic DataFrame has duplicate rows.")

    if control_g is None:
        control_df = None
    else:
        control_df = _graph_to_node_df(control_g, feat_attr, struct_attrs)
        control_df = control_df.astype(ori_df.dtypes)

        if embed:
            control_df = build_embed_df(control_g, control_df, embedding_model)

        # if control.duplicated().any():
            # raise AssertionError("Control DataFrame has duplicate rows.")

    if add_syn_id:
        for i, syn in enumerate(syn_dfs):
            syn["graph_id"] = i

    syn_df = pd.concat(syn_dfs, ignore_index=True).astype(ori_df.dtypes)

    return ori_df, syn_df, control_df


class NodeInferenceEvaluator(InferenceEvaluator):
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
        embed=False,
    ):
        ori, syn, control = build_node_dfs(
            ori_g, syn_gs, control_g, True, struct_attrs, embed)

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


class NodeLinkabilityEvaluator(LinkabilityEvaluator):
    def __init__(
        self,
        ori_g: nx.Graph,
        syn_gs: list[nx.Graph],
        aux_cols: AUX_COLS_LINKABILITY_TYPE,
        n_attacks: int = 500,
        n_neighbors: int = 1,
        control_g: nx.Graph | None = None,
        struct_attrs: STRUCT_ATTRS_TYPE = "all",
        embed=False,
    ):
        ori, syn, control = build_node_dfs(
            ori_g, syn_gs, control_g, aux_cols != "struct-struct", struct_attrs, embed)

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


def _evaluate_node_queries(df: pl.DataFrame, queries: list[pl.Expr]) -> tuple[int, ...]:
    if len(queries) == 0:
        return ()

    result_df = df.select([
        pl.col("graph_id").filter(q).is_unique().sum().alias(f"count_{i}")
        for i, q in enumerate(queries)])
    counts = result_df.row(0)
    return counts


class UniqueNodeSinglingOutQueries(UniqueSinglingOutQueries):
    def check_and_extend(
            self, queries: list[pl.Expr], df: pl.DataFrame,
            min_matching_syns: int) -> None:
        if self._max_size and len(self._list) >= self._max_size:
            return

        counts = _evaluate_node_queries(df=df, queries=queries)

        for query, count in zip(queries, counts):
            if count >= min_matching_syns:
                query_str = str(query)
                if query_str not in self._set:
                    self._set.add(query_str)
                    self._list.append(query)
                    if self._max_size and len(self._list) >= self._max_size:
                        return


def univariate_node_singling_out_queries(
        df: pl.DataFrame, n_queries: int,
        rng: np.random.Generator,
        min_matching_syns: int) -> list[pl.Expr]:
    queries = []

    schema = df.schema

    for df_ in df.partition_by("graph_id", include_key=False):
        for col in df_.columns:
            # Exactly one null
            null_count = df_.select(pl.col(col).is_null().sum()).item()
            if null_count == 1:
                queries.append(pl.col(col).is_null())

            # Numeric columns
            if schema[col].is_numeric():
                non_null_count = df_[col].drop_nulls().len()

                if non_null_count > 0:
                    col_min = df_[col].min()
                    col_max = df_[col].max()
                    queries.extend(
                        [
                            pl.col(col) <= col_min,
                            pl.col(col) >= col_max,
                        ]
                    )

            # Rare values
            counts_df = df_.group_by(col).len()
            rare_values_df = counts_df.filter(pl.col("len") == 1)
            rare_values = rare_values_df.select(col).to_series().to_list()
            if len(rare_values) > 0:
                queries.extend([pl.col(col) == val for val in rare_values])

    rng.shuffle(queries)

    unique_so_queries = UniqueNodeSinglingOutQueries(max_size=n_queries)
    unique_so_queries.check_and_extend(queries, df, min_matching_syns)

    return unique_so_queries.queries


def multivariate_node_singling_out_queries(
        df: pl.DataFrame, n_queries: int, n_cols: int,
        rng: np.random.Generator, min_matching_syns: int,
        batch_size: int = 1_000, max_patience=100) -> list[pl.Expr]:
    unique_so_queries = UniqueNodeSinglingOutQueries(max_size=n_queries)

    columns = [col for col in df.columns if col != "graph_id"]

    unique_graph_ids = df["graph_id"].unique().to_list()

    medians_dicts = df.group_by("graph_id").agg(pl.selectors.numeric().median()).to_dicts()

    dtypes_dict = df.schema

    last_len = 0
    patience = 0

    while len(unique_so_queries) < n_queries:
        if patience >= max_patience:
            logger.warning(
                f"Reached maximum patience {max_patience} when generating singling out queries. "
                f"Returning {len(unique_so_queries)} instead of the requested {n_queries}."
            )
            return unique_so_queries.queries

        # Generate a batch of queries

        # Pre-sample all random row indices
        random_indices = rng.integers(low=0, high=df.shape[0], size=batch_size)

        # Extract records
        records_iter = df[random_indices].iter_rows(named=True)

        # Pre-sample all column choices
        selected_columns = [
            rng.choice(columns, size=n_cols, replace=False).tolist()
            for _ in range(batch_size)]

        graph_ids = rng.choice(unique_graph_ids, size=batch_size, replace=True).tolist()

        queries_batch = [
            _query_from_record(
                record=record,
                dtypes=dtypes_dict,
                columns=columns,
                medians=medians_dicts[graph_id],
                rng=rng,
            )
            for record, columns, graph_id in zip(records_iter, selected_columns, graph_ids)
        ]

        # Store queries that single out and that haven't been seen before
        unique_so_queries.check_and_extend(queries_batch, df, min_matching_syns)

        if len(unique_so_queries) == last_len:
            patience += 1
        else:
            last_len = len(unique_so_queries)
            patience = 0

        if len(unique_so_queries) >= n_queries:
            break

    return unique_so_queries.queries


def _generate_node_singling_out_queries(
        df: pl.DataFrame, n_attacks: int,
        n_cols: int, rng: np.random.Generator,
        min_matching_syns: int) -> list[pl.Expr]:
    if n_cols == 1:
        queries = univariate_node_singling_out_queries(
            df=df,
            n_queries=n_attacks,
            rng=rng,
            min_matching_syns=min_matching_syns
        )
        mode = "univariate"
    elif n_cols > 1:
        queries = multivariate_node_singling_out_queries(
            df=df,
            n_queries=n_attacks,
            n_cols=n_cols,
            rng=rng,
            min_matching_syns=min_matching_syns
        )
        mode = "multivariate"
    else:
        raise RuntimeError(f"Parameter `n_cols` must be a positive integer.")

    if len(queries) < n_attacks:
        logger.warning(
            f"Attack `{mode}` could generate only {len(queries)} "
            f"singling out queries out of the requested {n_attacks}. "
        )
    return queries


def _random_node_queries(
    df: pl.DataFrame,
    n_queries: int,
    n_cols: int,
    rng: np.random.Generator,
) -> list[pl.Expr]:
    columns = [col for col in df.columns if col != "graph_id"]
    unique_values = {col: df[col].unique().to_list() for col in columns}
    column_types = {col: _convert_polars_dtype(df[col].dtype) for col in columns}

    queries = []
    for _ in range(n_queries):
        selected_cols = rng.choice(columns, size=n_cols, replace=False).tolist()

        queries.append(_random_query(
            unique_values=unique_values, cols=selected_cols, column_types=column_types, rng=rng))

    return queries


class NodeSinglingOutEvaluator(SinglingOutEvaluator):
    def __init__(
        self,
        ori_g: nx.Graph,
        syn_gs: list[nx.Graph],
        n_attacks: int = 500,
        control_g: nx.Graph | None = None,
        feat_attr=True,
        struct_attrs: STRUCT_ATTRS_TYPE = "all",
        embed=False,
    ):
        ori, syn, control = build_node_dfs(
            ori_g, syn_gs, control_g, feat_attr, struct_attrs, embed, add_syn_id=True)

        super().__init__(ori, syn, n_attacks, 0, control)

    def evaluate(self, n_cols: int, min_matching_syns: int) -> "NodeSinglingOutEvaluator":
        _n_cols = len(self._syn.columns) - 1 if n_cols == -1 else n_cols
        if min_matching_syns == -1:
            _min_matching_syns = self._syn.select(pl.col("graph_id").n_unique()).item()
        else:
            _min_matching_syns = min_matching_syns

        queries = _generate_node_singling_out_queries(
            df=self._syn,
            n_attacks=self._n_attacks,
            n_cols=_n_cols,
            rng=self._rng,
            min_matching_syns=_min_matching_syns
        )
        self._n_attacks_ori = max(len(queries), 1)
        self._queries = _evaluate_queries_and_return_successful(df=self._ori, queries=queries)
        self._n_success = len(self._queries)

        baseline_queries = _random_node_queries(
            df=self._syn,
            n_queries=self._n_attacks,
            n_cols=_n_cols,
            rng=self._rng,
        )
        self._n_attacks_baseline = len(baseline_queries)
        self._baseline_queries = _evaluate_queries_and_return_successful(
            df=self._ori, queries=baseline_queries)
        self._n_baseline = len(self._baseline_queries)

        if self._control is None:
            self._n_control = None
        else:
            self._n_control = len(_evaluate_queries_and_return_successful(
                df=self._control, queries=queries))
            self._n_attacks_control = self._n_attacks_ori

            # correct the number of success against the control set
            # to account for different dataset sizes.
            if len(self._control) != len(self._ori):
                # fit the model to the data:
                fitted_model = fit_correction_term(df=self._control, queries=queries)

                correction = fitted_model(len(self._ori)) / fitted_model(len(self._control))
                self._n_control = min(self. _n_attacks, self._n_control * correction)

        self._evaluated = True
        return self

    def results(self, confidence_level: float = 0.95) -> EvaluationResults:
        if not self._evaluated:
            raise RuntimeError(
                "The singling out evaluator wasn't evaluated yet. Please, run `evaluate()` first.")

        return EvaluationResults(
            n_attacks=(self._n_attacks_ori, self._n_attacks_baseline, self._n_attacks_control),
            n_success=self._n_success,
            n_baseline=self._n_baseline,
            n_control=self._n_control,
            confidence_level=confidence_level,
        )
