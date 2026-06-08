import pickle
from dataclasses import dataclass
from pathlib import Path
from random import Random

import networkx as nx
import pandas as pd
from torch_geometric.data import download_url, extract_tar, extract_zip


@dataclass
class EgoData:
    graph: nx.Graph
    feats: pd.DataFrame


def read_ego_datas(ds_path: Path) -> dict[str, EgoData]:
    ego_datas: dict[str, EgoData] = {}

    for path in ds_path.glob("*.edges"):
        stem = path.stem
        parent = path.parent
        feats = [
            line.strip().split(' ')
            for line in (parent / f"{stem}.feat").read_text().splitlines()
        ]
        feats = {
            n: [int(f) for f in fs]
            for n, *fs in feats
        }
        feats[stem] = [
            int(f)
            for f in (parent / f"{stem}.egofeat").read_text().strip().split(' ')
        ]
        g: nx.Graph = nx.read_edgelist(parent / f"{stem}.edges")
        g.add_nodes_from(feats.keys())
        g.add_edges_from((stem, n) for n in feats.keys())
        g.remove_edges_from(nx.selfloop_edges(g))
        feat_names = [
            line.strip().split(' ', 1)[-1]
            for line in (parent / f"{stem}.featnames").read_text().splitlines()
        ]
        feats = pd.DataFrame.from_dict(feats, orient="index", columns=feat_names)
        ego_datas[stem] = EgoData(g, feats)

    return ego_datas


def ego_data_to_graph(ego_data: EgoData, feat_names: list[str]) -> nx.Graph:
    graph = ego_data.graph.copy()
    feats = ego_data.feats
    feat_lists = ego_data.feats[feat_names].to_numpy().tolist()
    nx.set_node_attributes(graph, dict(zip(feats.index, feat_lists)), name='x')

    return nx.convert_node_labels_to_integers(graph)


def write_attr_data(ego_datas: dict[str, EgoData], write_dir: Path):
    train_ego = max(ego_datas.values(), key=lambda x: len(x.graph))
    train_nodes = set(train_ego.graph.nodes)
    control_ego = EgoData(nx.Graph(), pd.DataFrame())
    control_size = 0
    train_feats = set(train_ego.feats.columns)
    common_feats = train_feats

    for ego_data in ego_datas.values():
        if len(set(ego_data.graph.nodes) & train_nodes) > 0:
            continue

        size = len(ego_data.graph)

        if size < control_size:
            continue

        new_common_feats = train_feats & set(ego_data.feats.columns)

        if size == control_size and len(new_common_feats) > len(common_feats):
            continue

        control_ego = ego_data
        control_size = size
        common_feats = new_common_feats


    common_feats = sorted(common_feats)
    data = {
        "train": ego_data_to_graph(train_ego, common_feats),
        "control": ego_data_to_graph(control_ego, common_feats)
    }

    with open(write_dir / f"{write_dir.stem}_attr.pickle", "wb") as f:
        pickle.dump(data, f)


def write_unattr(
        graphs: list[nx.Graph], write_dir: Path,
        seed=0, n_max_graphs=100, ratio_train=0.8):
    gs = sorted(graphs, key=lambda x: len(x))[-n_max_graphs:]
    Random(seed).shuffle(gs)
    threshold = round(ratio_train * len(gs))
    data = {
        "train": gs[:threshold],
        "control": gs[threshold:]
    }

    with open(write_dir / f"{write_dir.stem}_unattr.pickle", "wb") as f:
        pickle.dump(data, f)


def main():
    tmp_dir = Path(__file__).parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_dir_str = tmp_dir.absolute().as_posix()
    datasets_dir = Path(__file__).parent / "datasets"

    for ds_name in ("facebook", "twitter"):
        data_dir_str = download_url(
            f"https://snap.stanford.edu/data/{ds_name}.tar.gz",
            tmp_dir_str, filename=f"{ds_name}.tar.gz")
        extract_tar(data_dir_str, tmp_dir_str)
        ego_datas = read_ego_datas(tmp_dir / ds_name)
        ds_dir = datasets_dir / f"ego-{ds_name}"
        ds_dir.mkdir(exist_ok=True)
        write_attr_data(ego_datas, ds_dir)
        write_unattr(
            [ego_data_to_graph(ego_data, []) for ego_data in ego_datas.values()],
            ds_dir
        )
        del ego_datas

    ds_name = "elliptic"
    download_url_prefix = "https://data.pyg.org/datasets/elliptic/"
    suffix = f"{ds_name}_txs_edgelist.csv"
    data_dir_str = download_url(f"{download_url_prefix}{suffix}.zip", tmp_dir_str)
    extract_zip(data_dir_str, tmp_dir_str)
    graph: nx.Graph = nx.read_edgelist(tmp_dir / suffix, comments="txId1,txId2", delimiter=',')
    suffix = f"{ds_name}_txs_classes.csv"
    data_dir_str = download_url(f"{download_url_prefix}{suffix}.zip", tmp_dir_str)
    extract_zip(data_dir_str, tmp_dir_str)
    classes = pd.read_csv(tmp_dir / suffix).set_index('txId')['class'].to_dict()
    classes = {str(k): 0 if v == "unknown" else int(v) for k, v in classes.items()}
    graphs: list[nx.Graph] = [
        graph.subgraph(sorted(cc))
        for cc in sorted(
            nx.connected_components(graph),
            key=lambda x: min(x)
        )
    ]
    ds_dir = datasets_dir / ds_name
    ds_dir.mkdir(exist_ok=True)
    #
    train_graph = max(graphs, key=lambda x: len(x)).copy()
    nx.set_node_attributes(train_graph, classes, name='x')
    train_graph_len = len(train_graph)
    control_graph = max(graphs, key=lambda x: len(x) * (len(x) != train_graph_len)).copy()
    nx.set_node_attributes(control_graph, classes, name='x')
    data = {
        "train": nx.convert_node_labels_to_integers(train_graph),
        "control": nx.convert_node_labels_to_integers(control_graph)
    }

    with open(ds_dir / f"{ds_name}_attr.pickle", "wb") as f:
        pickle.dump(data, f)
    #
    write_unattr([nx.convert_node_labels_to_integers(g) for g in graphs], ds_dir)


if __name__ == "__main__":
    main()
