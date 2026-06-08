# SyntheGrAnon

Install environment with `conda env create -p .env -f environment.yml` or `pip install -r requirements.txt`.

Run the experiments with `python exps/run_graph_exps.py`.

Compute MMD measurements with `python exps/compute_utility.py --dataset $D --type $T --models $M1 [$M2 ...]`, where:

- `$D` is one of `ego-facebook`, `ego-twitter`, or `elliptic`;
- `$T` is one of `attr` (node/edge-level) or `unattr` (community-level);
- `$M1` / `$M2` / ... are each one of `edge`, `ggsd`, `gran`, `grum`, `spectre`.

Selected dataset, type, and model combinations should correspond to an available checkpoint.

Process experiment results into tables and plots with `python exps/proc_graph_exps.py`.

The codebase is based upon that of [Anonymeter](https://github.com/statice/anonymeter).

# Artifact Appendix

Paper title: **Measuring Legislature-Aligned Privacy Risks in Synthetic Graphs**

## Artifact Review

Requested Badge(s):
  - [x] **Available**
  - [x] **Functional**
  - [x] **Reproduced**

## Description

This repository is the artifact for the paper "Measuring Legislature-Aligned Privacy Risks in Synthetic Graphs", authored by Abele Malan, Ahmad Al Kurdi, Stefanie Roos, and Lydia Chen, and accepted at PoPETs 2026.
The paper evaluates singling-out, linkability, and inference privacy risks in synthetic graphs at node, community, and edge levels.
The repository includes the attack implementation, processed real graph data, fixed synthetic graph samples, raw experiment measurements, and scripts that regenerate Tables 3, 4, 6, and 7, as well as Figures 5-7 from the paper.

### Security/Privacy Issues and Ethical Concerns

The artifact does not disable any security mechanisms or run vulnerable code.
The code loads the processed datasets and the synthetic graph samples via the provided `.pickle` files.
Its attacks are offline statistical evaluations against bundled datasets.
The Facebook and Twitter graph data are anonymized and relabeled processed subsets from the Stanford SNAP repository.
The Elliptic data consist of public Bitcoin transactions and their class labels.

## Basic Requirements

### Hardware Requirements

1. Can run on a laptop (No special hardware requirements).
However, for a full evaluation (recomputing attack/utility measurements instead of the quick path, which recomputes tables/figures from pre-existing measurements), the minimum recommended practical configuration is a 16-core x86-64 CPU and 32 GB RAM.
Moreover, a CUDA-capable GPU is recommended specifically for recomputing the downstream link-prediction task, although Apple MPS and CPU execution are also supported.
2. All experiments reported in the paper were run on one of the following three machines: (i) a 28-thread Intel Core i7-14700KF CPU, 64 GB RAM, Nvidia RTX 4090 GPU; (ii) 30-threads from an Intel Xeon Platinum 8562Y+, 124 GB RAM, Nvidia H100 NVL GPU; (iii) a 32-thread Intel Xeon Gold 6326 CPU, 64 GB RAM.
The paper reports privacy and utility metrics rather than timing or throughput, so hardware differences should mainly affect runtime rather than the overall results.

### Software Requirements

1. Host OS: validated on Ubuntu 24.04.4 LTS.
2. OS packages: conda (including Miniconda and Miniforge variants; validated on Miniforge 26.3); g++ (validated on 13.3)
3. Artifact packaging: no container runtime is used, only the conda package manager alongside the pip package installer.
4. Interpreter: Python 3.11.
5. Packages: Python packages specified in `requirements.txt` (which also references `pyproject.toml`).
6. Machine Learning Models: no pretrained model required.
7. Datasets: the preprocessed versions of all datasets (`Ego-Facebook`, `Ego-Twitter`, `Elliptic`) are included with the artifact.

### Estimated Time and Storage Consumption

- Overall time to run the artifact: human - 2 hours; compute time - 0.5 hours for quick evaluation (including setup), 1.5 days for full reproduction.
- Overall disk space consumed by the artifact: approximately 10 GB.

## Environment

### Accessibility

The artifact is available at: https://github.com/AbeleMM/SyntheGrAnon/tree/main

### Set up the environment


```sh
git clone git@github.com:AbeleMM/SyntheGrAnon.git
cd SyntheGrAnon
conda env create -p .env -f environment.yml
conda activate .env/
```

Finally, with your working directory temporarily set to `exps/orca`, compile `orca`; for instance, on Linux:

```sh
(cd exps/orca && g++ -O2 -std=c++11 -o orca orca.cpp)
```

### Testing the Environment

With the environment activated and the working directory at the root of the repository, run the smoke test (while simultaneously recomputing the paper tables & figures from the pre-existing raw measurements) via:

```sh
python run.py quick
```

The expected output should be similar to:

```text
Smoke test passed: community singling-out risk=0.1369, 95% CI=[0.0000, 0.3207]
Completed 'quick'. Outputs are under exps/.
```

## Artifact Evaluation

### Main Results and Claims

## Main Results and Claims

#### Main Result 1: Node-Level Privacy Risk

Table 3 shows that node-level privacy risk tends to be high for singling-out and inference, while for linkability it is comparatively low.
Figure 5 shows how the different components supported by our node-level attacks, especially singling out, influence risk.
The result is supported by Experiments 1 and 2 below.

#### Main Result 2: Community-Level Privacy Risk

Table 4 shows that community-level privacy risk tends to be quite high for linkability and inference, while singling out is lower.
Figure 6 shows that, generally, attack risk increases when moving from one to two predicates, then usually flattens out with further increases.
The result is supported by Experiments 1 and 2 below.

#### Main Result 3: Edge-Level Privacy Risk

Table 6 shows that, similar to the node case, edge-level privacy risk tends to be high for singling-out and inference, while for linkability it is comparatively low.
Figure 7, again similar to the node case, shows the benefits of increasing the minimum number of synthetic graphs or predicate variables (up to a point) for singling out attacks.
The experiment is supported by Experiments 1 and 2 below.

#### Main Result 4: Utility and Privacy

The ROC AUC downstream utility values in Tables 3 and 4, and the MMD values in Table 7, show that no synthesizer dominates every utility metric, and that higher quality does not uniformly imply higher privacy risk.
The result is supported by Experiments 1 and 3 below.

### Experiments

#### Experiment 1: Reproduce Paper Outputs

- Execution steps:

```bash
python run.py reproduce
```

- Expected result: the plots in `exps/plots` and tables in `exps/tables` are recomputed and overwritten to disk based on the raw measurements from `exps/results` and `exps/utility.csv`.
- Time: 5 human minutes and <1 compute minute.
- Supported results and claims: partly 1-4 by reproducing the exact figures/tables from the paper from the pre-existing raw measurements.

#### Experiment 2: Recompute Privacy Attacks

- Execution steps:

```bash
python run.py attacks
```

- Expected result: the raw measurements in `exps/results` are recomputed and overwritten to disk, also the plots in `exps/plots` and tables in `exps/tables` are recomputed and overwritten to disk based on the newly-computed attack measurements (using pre-existing raw measurements for utility).
- Time: 15 human minutes and approximately 1 compute day on 32 CPU threads.
- Supported results and claims: 1-3 by reproducing the exact figures/tables from the paper from the included preprocessed datasets and synthetic graph samples (excluding utility measurements).

#### Experiment 3: Recompute Utility and MMD

- Execution steps:

```bash
python run.py utility
```

- Expected result: the raw measurements in `exps/utility.csv` are recomputed and overwritten to disk, also the tables in `exps/tables` are recomputed and overwritten to disk based on the newly-computed utility/MMD measurements (using pre-existing raw measurements for attacks).
- Time: 15 human minutes and approximately 0.5 compute-days on 32 CPU threads and a modern CUDA-capable GPU.
- Supported results and claims: 4 by reproducing the utility and MMD measurements from Tables 3, 4, 6, and 7.

## Limitations

The artifact directly uses the included sets of synthetic graphs sampled from the evaluated graph generation models for the different datasets.
Thus, the code/process for retraining the models or resampling graphs is not included.
SyntheGrAnon's primary contribution is the graph-specific privacy evaluation rather than the creation of the synthetic graphs used as input.

Full attack reruns contain randomized attack sampling and are not bitwise deterministic.
Exact tables and figures are reproducible from the released raw experiment measurements; full reruns validate functionality and the paper's qualitative and quantitative trends.

Some dataset/model combinations are absent for the reasons reported in Section 5.1.2: model capability, invalid samples, or excessive memory use.

## Notes on Reusability

The introduced graph-specific evaluator classes (`src/anonymeter/evaluators/node_evaluators.py`, `exps/graph_funcs.py`, `src/anonymeter/evaluators/edge_evaluators.py`) can be applied to new synthetic graph generators or graph datasets, similar to the tabular evaluator classes from the original Anonymeter repository, which the artifact uses as a base.
New graph families can be added by following the pickle structure documented by `exps/create_dataset.py`, plus placing real and synthetic samples under `exps/datasets/<dataset>`.
Structural node and graph attributes are centralized in `node_evaluators.py` and `graph_funcs.py`, enabling the addition of domain-specific properties.
The experiment scripts use their original fixed paths under `exps`, while the evaluators can be imported into other experiment pipelines.
The graph experiments cache node-, community-, and edge-level structural attributes to disk for reuse across experiments on the same graphs.
