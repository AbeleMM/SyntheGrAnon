import argparse
import copy
import os
import pickle
import subprocess as sp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from random import Random
from string import ascii_uppercase, digits
from typing import Literal, cast

import networkx as nx
import numpy as np
import pandas as pd
import pyemd
import pygsp as pg
import torch
import torch.nn.functional as F
import torch_geometric.data as pyg_data
import torch_geometric.loader as pyg_loader
import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as pyg_transforms
import torch_geometric.utils as pyg_utils
from scipy.linalg import eigvalsh, toeplitz
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

motif_to_indices = {
    "3path": [1, 2],
    "4cycle": [8],
}
COUNT_START_STR = "orbit counts:"
RAND = Random(-1)


def emd(x, y, distance_scaling=1.0):
    """EMD
    Args:
        x, y: 1D pmf of two distributions with the same support
    """
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    return np.abs(pyemd.emd(x, y, distance_mat))


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """Gaussian kernel with squared distance in exponential term replaced by EMD
    Args:
        x, y: 1D pmf of two distributions with the same support
        sigma: standard deviation
    """
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd * emd / (2 * sigma * sigma))


def gaussian(x, y, sigma=1.0):
    support_size = max(len(x), len(y))
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian_tv(x, y, sigma=1.0):
    support_size = max(len(x), len(y))
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, parallel=True, *args, **kwargs):
    """Discrepancy between 2 samples"""
    d = 0

    if not parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with ThreadPoolExecutor() as executor:
            for dist in executor.map(
                kernel_parallel_worker,
                [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1],
            ):
                d += dist
    if len(samples1) * len(samples2) > 0:
        d /= len(samples1) * len(samples2)
    else:
        d = 1e6
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """MMD between two samples"""
    # normalize histograms into pmf
    if is_hist:
        samples1 = [s1 / (np.sum(s1) + 1e-6) for s1 in samples1]
        samples2 = [s2 / (np.sum(s2) + 1e-6) for s2 in samples2]
    mmd = (
        disc(samples1, samples1, kernel, *args, **kwargs)
        + disc(samples2, samples2, kernel, *args, **kwargs)
        - 2 * disc(samples1, samples2, kernel, *args, **kwargs)
    )

    mmd = np.abs(mmd)

    if mmd < 0:
        import pdb

        pdb.set_trace()

    return mmd


def compute_emd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """EMD between average of two samples"""
    # normalize histograms into pmf
    if is_hist:
        samples1 = [np.mean(samples1)]
        samples2 = [np.mean(samples2)]
    return disc(samples1, samples2, kernel, *args, **kwargs), [samples1[0], samples2[0]]


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(
        true_graphs: list[nx.Graph],
        pred_graphs: list[nx.Graph],
        parallel=True, emd=False) -> float:
    sample_true = []
    sample_pred = []

    if parallel:
        with ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, true_graphs):
                sample_true.append(deg_hist)
        with ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, pred_graphs):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(true_graphs)):
            degree_temp = np.array(nx.degree_histogram(true_graphs[i]))
            sample_true.append(degree_temp)
        for i in range(len(pred_graphs)):
            degree_temp = np.array(nx.degree_histogram(pred_graphs[i]))
            sample_pred.append(degree_temp)

    if emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(sample_true, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_true, sample_pred, kernel=gaussian_tv)

    return mmd_dist


def eigh_worker(G):
    L = nx.normalized_laplacian_matrix(G).todense()
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        eigvals = np.zeros(L[0, :].shape)
        eigvecs = np.zeros(L.shape)
    return (eigvals, eigvecs)


def compute_list_eigh(graph_list, parallel=False):
    eigval_list = []
    eigvec_list = []
    if parallel:
        with ThreadPoolExecutor() as executor:
            for e_U in executor.map(eigh_worker, graph_list):
                eigval_list.append(e_U[0])
                eigvec_list.append(e_U[1])
    else:
        for i in range(len(graph_list)):
            e_U = eigh_worker(graph_list[i])
            eigval_list.append(e_U[0])
            eigvec_list.append(e_U[1])
    return eigval_list, eigvec_list


class DMG(object):
    """Dummy Normalized Graph"""
    lmax = 2


def get_spectral_filter_worker(eigvec, eigval, filters, bound=1.4):
    ges = filters.evaluate(eigval)
    linop = []
    for ge in ges:
        linop.append(eigvec @ np.diag(ge) @ eigvec.T)
    linop = np.array(linop)
    norm_filt = np.sum(linop**2, axis=2)
    hist_range = (0, bound)
    hist = np.array(
        [np.histogram(x, range=hist_range, bins=100)[0] for x in norm_filt]
    )  # NOTE: change number of bins
    return hist.flatten()


def spectral_filter_stats(
        true_eigvecs: list,
        true_eigvals: list,
        pred_eigvecs: list,
        pred_eigvals: list,
        parallel=False, emd=False) -> float:
    """Compute the distance between the eigvector sets."""

    n_filters = 12
    filters = pg.filters.Abspline(DMG, n_filters)
    bound = np.max(filters.evaluate(np.arange(0, 2, 0.01)))
    sample_true = []
    sample_pred = []
    if parallel:
        with ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_filter_worker,
                true_eigvecs,
                true_eigvals,
                [filters for _ in range(len(true_eigvals))],
                [bound for _ in range(len(true_eigvals))],
            ):
                sample_true.append(spectral_density)
        with ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_filter_worker,
                pred_eigvecs,
                pred_eigvals,
                [filters for _ in range(len(pred_eigvals))],
                [bound for _ in range(len(pred_eigvals))],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(true_eigvals)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    true_eigvecs[i], true_eigvals[i], filters, bound
                )
                sample_true.append(spectral_temp)
            except:
                pass
        for i in range(len(pred_eigvals)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    pred_eigvecs[i], pred_eigvals[i], filters, bound
                )
                sample_pred.append(spectral_temp)
            except:
                pass

    if emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(sample_true, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_true, sample_pred, kernel=gaussian_tv)

    return mmd_dist


def spectral_worker(G, n_eigvals=-1):
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    except:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1: n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def spectral_stats(
        graph_ref_list: list[nx.Graph],
        graph_pred_list: list[nx.Graph],
        n_eigvals=-1,
        parallel=True, emd=False) -> float:
    """Compute the distance between the degree distributions of two unordered sets of graphs."""
    sample_true = []
    sample_pred = []

    if parallel:
        with ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                spectral_worker, graph_ref_list, [n_eigvals for _ in graph_ref_list]
            ):
                sample_true.append(spectral_density)
        with ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                spectral_worker,
                graph_pred_list,
                [n_eigvals for _ in graph_pred_list],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i], n_eigvals)
            sample_true.append(spectral_temp)
        for i in range(len(graph_pred_list)):
            spectral_temp = spectral_worker(graph_pred_list[i], n_eigvals)
            sample_pred.append(spectral_temp)

    if emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(sample_true, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_true, sample_pred, kernel=gaussian_tv)

    return mmd_dist


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(cast(dict, nx.clustering(G)).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
    )
    return hist


def clustering_stats(
        true_graphs: list[nx.Graph],
        pred_graphs: list[nx.Graph],
        bins=100,
        parallel=True, emd=False) -> float:
    sample_true = []
    sample_pred = []

    if parallel:
        with ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in true_graphs]
            ):
                sample_true.append(clustering_hist)
        with ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in pred_graphs]
            ):
                sample_pred.append(clustering_hist)
    else:
        for i in range(len(true_graphs)):
            clustering_coeffs_list = list(cast(dict, nx.clustering(true_graphs[i])).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_true.append(hist)

        for i in range(len(pred_graphs)):
            clustering_coeffs_list = list(cast(dict, nx.clustering(pred_graphs[i])).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)

    if emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(
            sample_true,
            sample_pred,
            kernel=gaussian_emd,
            sigma=1.0 / 10,
            distance_scaling=bins,
        )
    else:
        mmd_dist = compute_mmd(
            sample_true, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10
        )

    return mmd_dist


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for u, v in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    tmp_fname = f'orca/tmp_{"".join(RAND.choice(ascii_uppercase + digits) for _ in range(8))}.txt'
    tmp_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), tmp_fname)
    f = open(tmp_fname, "w")
    f.write(str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n")
    for u, v in edge_list_reindexed(graph):
        f.write(str(u) + " " + str(v) + "\n")
    f.close()
    output = sp.check_output(
        [
            str(os.path.join(os.path.dirname(os.path.realpath(__file__)), "orca/orca")),
            "node",
            "4",
            tmp_fname,
            "std",
        ]
    )
    output = output.decode("utf8").strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array(
        [
            list(map(int, node_cnts.strip().split(" ")))
            for node_cnts in output.strip("\n").split("\n")
        ]
    )

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def motif_stats(
        true_graphs,
        pred_graphs,
        motif_type="4cycle",
        ground_truth_match=None) -> float:
    # graph motif counts (int for each graph)
    # normalized by graph size
    total_counts_real = []
    total_counts_pred = []

    num_matches_real = []
    num_matches_pred = []

    pred_graphs = [
        G for G in pred_graphs if not G.number_of_nodes() == 0
    ]
    indices = motif_to_indices[motif_type]

    for G in true_graphs:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_real.append(match_cnt / G.number_of_nodes())

        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_real.append(motif_temp)

    for G in pred_graphs:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_pred.append(match_cnt / G.number_of_nodes())

        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)

    total_counts_real = np.array(total_counts_real)[:, None]
    total_counts_pred = np.array(total_counts_pred)[:, None]

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(
            total_counts_real, total_counts_pred, kernel=gaussian, is_hist=False
        )
    else:
        mmd_dist = compute_mmd(
            total_counts_real, total_counts_pred, kernel=gaussian, is_hist=False
        )
    return mmd_dist


def orbit_stats(
        true_graphs: list[nx.Graph],
        pred_graphs: list[nx.Graph],
        compute_emd=False) -> float:
    total_counts_true = []
    total_counts_pred = []

    for G in true_graphs:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_true.append(orbit_counts_graph)

    for G in pred_graphs:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_true = np.array(total_counts_true)
    total_counts_pred = np.array(total_counts_pred)

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(
            total_counts_true,
            total_counts_pred,
            kernel=gaussian,
            is_hist=False,
            sigma=30.0,
        )
    else:
        mmd_dist = compute_mmd(
            total_counts_true,
            total_counts_pred,
            kernel=gaussian_tv,
            is_hist=False,
            sigma=30.0,
        )

    return mmd_dist


def compute_stats(
        true_graphs: list[nx.Graph],
        pred_graphs: list[nx.Graph],
        true_eigvecs: list,
        true_eigvals: list) -> dict[str, float]:
    mmd_data: dict[str, float] = {}

    degree_val = degree_stats(
        true_graphs,
        pred_graphs,
        parallel=False
    )
    mmd_data["degree"] = degree_val

    pred_eigvals, pred_eigvecs = compute_list_eigh(pred_graphs)
    wavelet_val = spectral_filter_stats(
        true_eigvecs=true_eigvecs,
        true_eigvals=true_eigvals,
        pred_eigvecs=pred_eigvecs,
        pred_eigvals=pred_eigvals,
    )
    mmd_data["wavelet"] = wavelet_val

    spectre_val = spectral_stats(
        true_graphs,
        pred_graphs,
        parallel=False
    )
    mmd_data["spectre"] = spectre_val

    clustering_val = clustering_stats(
        true_graphs,
        pred_graphs,
        parallel=False
    )
    mmd_data["clustering"] = clustering_val

    motif_val = motif_stats(
        true_graphs,
        pred_graphs,
        motif_type="4cycle",
        ground_truth_match=None
    )
    mmd_data["motif"] = motif_val

    orbit_val = orbit_stats(
        true_graphs, pred_graphs
    )
    mmd_data["orbit"] = orbit_val

    return mmd_data


def compute_ratios(
        gen_metrics: dict[str, float],
        ref_metrics: dict[str, float]) -> dict[str, float]:
    ratios = {}

    for key, ref_val in ref_metrics.items():
        if round(ref_val, 4) != 0.:
            ratios[key + "_ratio"] = gen_metrics[key] / ref_val
        else:
            print(f"WARNING: Reference {key} is 0. Skipping its ratio.")

    ratios["average_ratio"] = sum(ratios.values()) / len(ratios)

    return ratios


class PrepNodeFeats(pyg_transforms.BaseTransform):
    max_size = 32

    def forward(self, data):
        if data.x is None or data.x.numel() == 0:
            x = pyg_utils.degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
            x = x.clamp(max=self.max_size - 1)
            data.x = F.one_hot(x, num_classes=self.max_size).float()
        else:
            data.x = data.x.float()

        del data.weight

        return data


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = pyg_nn.SAGEConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = pyg_nn.SAGEConv(hidden_channels, hidden_channels)

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, 1)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        return x

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        x = z[edge_label_index[0]] * z[edge_label_index[1]]

        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = self.lin3(x)

        return x.squeeze(-1)


def train_model(
        model: LinkPredictor, dl: pyg_loader.DataLoader,
        opt: torch.optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0.

    for batch in dl:
        batch: pyg_data.Data = batch.to(device)
        edge_index = cast(torch.Tensor, batch.edge_index)
        pos_edge_index = batch.edge_label_index
        n_pos_edges: int = pos_edge_index.size(1)
        opt.zero_grad()

        z = model.encode(cast(torch.Tensor, batch.x), edge_index)

        neg_edge_index = pyg_utils.negative_sampling(
            edge_index=torch.cat([edge_index, pos_edge_index], dim=1),
            num_nodes=batch.num_nodes,
            num_neg_samples=n_pos_edges,
            force_undirected=True
        )

        all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        all_edge_label = torch.cat([
            torch.ones(n_pos_edges, device=device),
            torch.zeros(neg_edge_index.size(1), device=device)
        ], dim=0)

        out = model.decode(z, all_edge_index)
        loss = F.binary_cross_entropy_with_logits(out, all_edge_label)

        loss.backward()
        opt.step()
        total_loss += loss.item()

    return total_loss / len(dl)


def eval_model(model: LinkPredictor, ref_loader: pyg_loader.DataLoader, device: str) -> float:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for b in ref_loader:
        batch = b.to(device)
        z = model.encode(batch.x, batch.edge_index)

        out = model.decode(z, batch.edge_label_index).sigmoid()

        all_labels.append(batch.edge_label.numpy(force=True))
        all_preds.append(out.numpy(force=True))

    score = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_preds))

    return cast(float, score)


def compute_downstream(
        train_graphs: list[nx.Graph], group_node_attrs: Literal["all", None],
        val_loader: pyg_loader.DataLoader, test_loader: pyg_loader.DataLoader,
        lr=1e-3, patience=10, check_every=10) -> dict[str, float]:
    transform = pyg_transforms.Compose([
        PrepNodeFeats(),
        pyg_transforms.RandomLinkSplit(
            num_val=0,
            num_test=0,
            is_undirected=True,
            add_negative_train_samples=False,
            disjoint_train_ratio=0.25
        )
    ])
    train_pygs = [
        transform(pyg_utils.from_networkx(g, group_node_attrs=group_node_attrs))[0]
        for g in train_graphs
    ]

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = LinkPredictor(
        in_channels=train_pygs[-1].num_node_features, hidden_channels=16)
    model = model.to(device)
    train_loader = pyg_loader.DataLoader(train_pygs, batch_size=100, shuffle=True, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_roc_auc_val = -1.
    best_model = model
    cur_patience = 0
    cur_epoch = 0

    while True:
        cur_epoch += 1
        loss = train_model(model, train_loader, opt, device)

        if cur_epoch % check_every == 0:
            roc_auc_val = eval_model(model, val_loader, device)
            print(
                f"Epoch {cur_epoch:03d} | "
                f"Train Loss (Syn): {loss:.4f} | "
                f"Val AUC (Real): {roc_auc_val:.4f}")
            if roc_auc_val > best_roc_auc_val:
                best_model = copy.deepcopy(model)  # .to("cpu")
                cur_patience = 0
                best_roc_auc_val = roc_auc_val
            else:
                cur_patience += 1

                if cur_patience > patience:
                    break

    best_model.to(device)
    roc_auc_test = eval_model(best_model, test_loader, device)

    return {"roc_auc": roc_auc_test}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument('--type')
    parser.add_argument("--models", nargs='+')
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    dataset = args.dataset
    ds_type = args.type
    base_fname = f"{dataset}_{ds_type}.pickle"
    data_dir = base_dir / "datasets" / dataset

    with open(data_dir / base_fname, "rb") as f:
        ref_data: dict[str, nx.Graph | list[nx.Graph]] = pickle.load(f)

    train_graphs = ref_data["train"]

    if isinstance(train_graphs, nx.Graph):
        train_graphs = [train_graphs]

    control_graphs = ref_data["control"]

    if isinstance(control_graphs, nx.Graph):
        control_graphs = [control_graphs]

    real_graphs = [*train_graphs, *control_graphs]
    real_eigvals, real_eigvecs = compute_list_eigh(real_graphs)

    exp_configs = [
        {"dataset": dataset, "type": ds_type, "model": model}
        for model in args.models
    ]
    exp_configs_df = pd.DataFrame(exp_configs)
    res_path = base_dir / "utility.csv"

    if res_path.exists() and not args.rerun:
        res_df = pd.read_csv(res_path)
        merge_df = exp_configs_df.merge(
            res_df,
            how="left",
            on=list(exp_configs_df.columns),
            indicator=True
        )
        exp_configs_df = exp_configs_df[merge_df["_merge"] == "left_only"]

    exp_configs = exp_configs_df.to_dict(orient="records")

    group_node_attrs = "all" if ds_type == "attr" else None
    transform = pyg_transforms.Compose([
        PrepNodeFeats(),
        pyg_transforms.RandomLinkSplit(
            num_val=0.1,
            num_test=0.2,
            is_undirected=True,
            add_negative_train_samples=False,
            disjoint_train_ratio=0.
        )
    ])
    val_pygs: tuple[pyg_data.data.BaseData]
    test_pygs: tuple[pyg_data.data.BaseData]
    val_pygs, test_pygs, *_ = zip(*(
        transform(pyg_utils.from_networkx(g, group_node_attrs=group_node_attrs))[1:]
        for g in real_graphs
    ))
    val_loader = pyg_loader.DataLoader(
        val_pygs, batch_size=len(val_pygs), shuffle=True, pin_memory=True)
    test_loader = pyg_loader.DataLoader(
        test_pygs, batch_size=len(test_pygs), shuffle=True, pin_memory=True)

    for config in tqdm(exp_configs):
        with open(data_dir / f"{config['model']}_{base_fname}", "rb") as f:
            pred_graphs: list[nx.Graph] = pickle.load(f)

        pred_mmd = compute_stats(
            real_graphs, pred_graphs,
            real_eigvecs, real_eigvals)
        downstream = compute_downstream(pred_graphs, group_node_attrs, val_loader, test_loader)

        res_df_row = pd.DataFrame([config | pred_mmd | downstream]).round(4)
        res_df_row.to_csv(
            res_path,
            mode='w' if args.rerun else 'a',
            header=not res_path.exists(),
            index=False
        )


if __name__ == "__main__":
    main()
