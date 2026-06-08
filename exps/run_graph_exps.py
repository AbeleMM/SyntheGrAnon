import pickle
from argparse import ArgumentParser
from pathlib import Path

import networkx as nx
from graph_funcs import (
    GRAPH_STRUCT_ATTR_FUNCS,
    EdgeInferenceExpConf,
    EdgeLinkabilityExpConf,
    EdgeSinglingOutExpConf,
    GraphInferenceExpConf,
    GraphLinkabilityExpConf,
    GraphSinglingOutExpConf,
    NodeInferenceExpConf,
    NodeLinkabilityExpConf,
    NodeSinglingOutExpConf,
    attack_graph_dataset,
)

from anonymeter.evaluators.node_evaluators import NODE_STRUCT_ATTR_FUNCS

DS_DIR = Path(__file__).parent / "datasets"


def nr_tab_attrs(ds_name: str) -> int:
    with open(DS_DIR / ds_name / f"{ds_name}_attr.pickle", "rb") as f:
        g: nx.Graph = pickle.load(f)["train"]

    example_node_attr = next(iter(nx.get_node_attributes(g, 'x').values()))

    if isinstance(example_node_attr, list):
        return len(example_node_attr)

    return 1


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    rerun: bool = args.rerun
    ds_to_nr_tab_attrs = {
        ds: nr_tab_attrs(ds) for ds in ["ego-facebook", "ego-twitter", "elliptic"]}
    models = ["edge", "ggsd", "grum", "gran", "spectre"]
    sweep_nr_neigh_and_min_syn = list(range(1, 6))
    single_node_struct_attrs = [[x] for x in NODE_STRUCT_ATTR_FUNCS.keys()]
    nr_node_struct_attrs = len(single_node_struct_attrs)
    sorted_graph_struct_attrs = sorted(GRAPH_STRUCT_ATTR_FUNCS.keys())
    nr_graph_struct_attrs = len(sorted_graph_struct_attrs)

    for ds, nr_ds_tab_attrs in ds_to_nr_tab_attrs.items():
        node_attr_secrets = [f"attr_{i}" for i in range(nr_ds_tab_attrs)]

        attack_graph_dataset(
            ds, NodeInferenceExpConf, models,
            {
                "n_attacks": [-1],
                "embed": [False],
                "struct_attrs": ["all"],
                "aux_cols": ["feat"] * (nr_ds_tab_attrs > 1) + ["struct", "all"],
                "secret": node_attr_secrets
            },
            rerun
        )

        attack_graph_dataset(
            ds, NodeLinkabilityExpConf, models,
            {
                "n_attacks": [-1],
                "embed": [False],
                "struct_attrs": ["all"],
                "aux_cols": (
                    ["feat-feat"] * (nr_ds_tab_attrs > 1) +
                    ["struct-struct", "feat-struct", "random"]),
                "n_neighbors": sweep_nr_neigh_and_min_syn
            },
            rerun
        )

        attack_graph_dataset(
            ds, NodeSinglingOutExpConf, models,
            {
                "n_attacks": [500],
                "embed": [False],
                "feat_attr": [True],
                "struct_attrs": [[]],
                "n_cols": list(range(1, nr_ds_tab_attrs + 1)),
                "min_matching_syns": sweep_nr_neigh_and_min_syn
            },
            rerun
        )

        attack_graph_dataset(
            ds, NodeSinglingOutExpConf, models,
            {
                "n_attacks": [500],
                "embed": [False],
                "feat_attr": [False],
                "struct_attrs": ["all"],
                "n_cols": list(range(1, nr_node_struct_attrs + 1)),
                "min_matching_syns": sweep_nr_neigh_and_min_syn
            },
            rerun
        )

        attack_graph_dataset(
            ds, NodeSinglingOutExpConf, models,
            {
                "n_attacks": [500],
                "embed": [False],
                "feat_attr": [True],
                "struct_attrs": ["all"],
                "n_cols": list(range(1, nr_ds_tab_attrs + nr_node_struct_attrs + 1)),
                "min_matching_syns": sweep_nr_neigh_and_min_syn
            },
            rerun
        )

        attack_graph_dataset(
            ds, NodeInferenceExpConf, models,
            {
                "n_attacks": [-1],
                "embed": [False],
                "aux_cols": ["all"],
                "secret": node_attr_secrets,
                "struct_attrs": single_node_struct_attrs
            },
            rerun
        )

        attack_graph_dataset(
            ds, NodeLinkabilityExpConf, models,
            {
                "n_attacks": [-1],
                "embed": [False],
                "aux_cols": ["random"],
                "n_neighbors": sweep_nr_neigh_and_min_syn,
                "struct_attrs": single_node_struct_attrs
            },
            rerun
        )

        attack_graph_dataset(
            ds, NodeSinglingOutExpConf, models,
            {
                "n_attacks": [500],
                "embed": [False],
                "feat_attr": [True],
                "n_cols": [nr_ds_tab_attrs + 1],
                "min_matching_syns": sweep_nr_neigh_and_min_syn,
                "struct_attrs": single_node_struct_attrs
            },
            rerun
        )

        attack_graph_dataset(
            ds, EdgeInferenceExpConf, models,
            {
                "n_attacks": [2_000],
                "struct_attrs": ["all"],
                "aux_cols": ["all"],
                "secret": [
                    f"attr_{attr_idx}-node_{node_idx}"
                    for attr_idx in range(nr_ds_tab_attrs)
                    for node_idx in range(2)
                ]
            },
            rerun
        )

        attack_graph_dataset(
            ds, EdgeLinkabilityExpConf, models,
            {
                "n_attacks": [2_000],
                "struct_attrs": ["all"],
                "aux_cols": ["random"],
                "n_neighbors": sweep_nr_neigh_and_min_syn
            },
            rerun
        )

        attack_graph_dataset(
            ds, EdgeSinglingOutExpConf, models,
            {
                "n_attacks": [500],
                "feat_attr": [True],
                "struct_attrs": ["all"],
                "n_cols": sweep_nr_neigh_and_min_syn,
                "min_matching_syns": sweep_nr_neigh_and_min_syn
            },
            rerun
        )

    for ds in ds_to_nr_tab_attrs:
        for i, a in enumerate(sorted_graph_struct_attrs):
            leftover_struct_attrs = sorted_graph_struct_attrs[:i] +\
                sorted_graph_struct_attrs[i + 1:]

            attack_graph_dataset(
                ds, GraphInferenceExpConf, models,
                {
                    "n_attacks": [-1],
                    "embed": [False],
                    "aux_cols": [
                        leftover_struct_attrs[:j]
                        for j in range(1, nr_graph_struct_attrs - 1)],
                    "secret": [a]
                },
                rerun
            )

        attack_graph_dataset(
            ds, GraphLinkabilityExpConf, models,
            {
                "n_attacks": [-1],
                "embed": [False],
                "aux_cols": [
                    (
                        sorted_graph_struct_attrs[:i],
                        sorted_graph_struct_attrs[i:]
                    )
                    for i in range(1, nr_graph_struct_attrs - 1)
                ],
                "n_neighbors": sweep_nr_neigh_and_min_syn
            },
            rerun
        )

        attack_graph_dataset(
            ds, GraphSinglingOutExpConf, models,
            {
                "n_attacks": [500],
                "embed": [False],
                "n_cols": list(range(1, nr_graph_struct_attrs + 1))
            },
            rerun
        )


if __name__ == "__main__":
    main()
