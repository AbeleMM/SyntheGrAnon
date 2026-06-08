from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

BASE_PATH = Path(__file__).parent
DATASET_VAL_MAPPING = {
    "ego-facebook": "Ego-Facebook",
    "ego-twitter": "Ego-Twitter",
    "elliptic": "Elliptic"
}
MODEL_VAL_MAPPING = {
    "edge": "EDGE",
    "ggsd": "GGSD",
    "grum": "GruM",
    "spectre": "SPECTRE",
    "gran": "GRAN"
}
MODEL_COLORS = dict(zip(
    ["EDGE", "GGSD", "GruM", "SPECTRE", "GRAN"], sns.color_palette()))
PLOT_KWARGS = {"bbox_inches": "tight", "metadata": {"CreationDate": None}}

sns.set_context("paper", font_scale=1.75)


def read_results() -> dict[tuple[str, ...], pd.DataFrame]:
    results = {
        tuple(res_path.stem.split('_')): pd
        .read_csv(res_path)
        .rename(columns={
                "dataset": "Dataset",
                "model": "Model",
                "risk_value": "Risk",
                "min_matching_syns": "Min Singled-out Syn Graphs",
                "struct_attrs": "Structural Attributes"})
        for res_path in (BASE_PATH / "results").glob(f"*.csv")
    }

    for df in results.values():
        df["Dataset"] = df["Dataset"].replace(DATASET_VAL_MAPPING)
        df["Model"] = df["Model"].replace(MODEL_VAL_MAPPING)

    return results


def tabulate_results(results: dict[tuple[str, ...], pd.DataFrame]) -> None:
    group_cols = ["lvl", "Dataset", "Model", "atk"]
    num_cols = ["mean", "std", "Risk", "ci"]

    results_ = [
        res.assign(**dict(zip(["lvl", "atk"], lvl_atk)))
        for lvl_atk, res in results.items()
    ]
    mean_results = [
        res
        .groupby(group_cols)["Risk"]
        .agg(["mean", "std"])
        .reset_index()
        for res in results_
    ]
    max_results = [
        res
        .sort_values(by=["Risk", "risk_ci_lwr", "risk_ci_upr"], ascending=False)
        .drop_duplicates(subset=group_cols, keep="first")
        for res in results_
    ]
    max_results = [
        res
        .assign(ci=np.maximum(
            res["Risk"] - res["risk_ci_lwr"],
            res["risk_ci_upr"] - res["Risk"],
        ))
        [[*group_cols, "Risk", "ci"]]
        .reset_index(drop=True)
        for res in max_results
    ]
    merged_results = [
        mean_r.merge(max_r, how="inner", on=group_cols)
        for mean_r, max_r in zip(mean_results, max_results)
    ]
    final_res = pd.concat(merged_results)
    final_res[num_cols] = final_res[num_cols].map(lambda x: f"{x:.2f}")
    final_res["mean"] = '$' + final_res["mean"] + " \\pm " + final_res["std"] + '$'
    final_res["max"] = '$' + final_res["Risk"] + " \\pm " + final_res["ci"] + '$'
    final_res = final_res.drop(["std", "Risk", "ci"], axis=1).reset_index(drop=True)
    final_res["atk"] = pd.Categorical(
        final_res["atk"],
        categories=["singling-out", "linkability", "inference"],
        ordered=True,
    )
    final_res = (
        final_res
        .pivot(
            index=["lvl", "Dataset", "Model"],
            columns="atk",
            values=["mean", "max"],
        )
        .reset_index()
    )
    lvl_col = ("lvl", "")

    for lvl in ["node", "graph", "edge"]:
        df = final_res.loc[final_res[lvl_col].eq(lvl)]
        df = df.drop(columns=[lvl_col])
        df.to_latex(BASE_PATH / "tables" / f"{lvl}.txt", index=False)


def postproc_fig(fg: sns.FacetGrid, fname: str, title_template: str | None = None) -> None:
    sns.move_legend(fg, "center right", frameon=True)
    fig = fg.figure
    fg.set_titles(title_template, size="large", fontweight="bold")
    fig.tight_layout()
    fig.savefig(
        BASE_PATH / "plots" / f"{fname}.pdf",
        bbox_inches="tight", metadata={"CreationDate": None})


def plot_so_min_matching_syns(
        results: dict[tuple[str, ...], pd.DataFrame], lvl: str) -> None:
    df = results[(lvl, "singling-out")]
    df = df[
        # (df["feat_attr"] == True) &
        # (df["Structural Attributes"] != "[]")
        (df["Structural Attributes"] == "all")
    ]

    fg = sns.relplot(
        data=df,
        x="Min Singled-out Syn Graphs",
        y="Risk",
        hue="Model",
        col="Dataset",
        kind="line",
        marker='o',
        palette=MODEL_COLORS,
        seed=0
    )
    fg.set_axis_labels("Min Singled-out Syn Graphs", "Risk")
    fg.set(xticks=sorted(df["Min Singled-out Syn Graphs"].unique()))
    postproc_fig(fg, f"so_min_matching_syns_{lvl}", "Dataset: {col_name}")


def plot_tab_struct_attrs(node_results: dict[str, pd.DataFrame]) -> None:
    attrs_choices = ["Tab only", "Struct only", "Both"]

    df_so = node_results["singling-out"]
    df_so["Attributes"] = np.select(
        [
            (df_so["feat_attr"] == True) & (df_so["Structural Attributes"] == "[]"),
            (df_so["feat_attr"] == False) & (df_so["Structural Attributes"] != "[]"),
            (df_so["feat_attr"] == True) & (df_so["Structural Attributes"] != "[]"),
        ],
        attrs_choices,
        default="-")

    df_link = node_results["linkability"]
    df_link["Attributes"] = np.select(
        [
            df_link["aux_cols"] == "feat-feat",
            df_link["aux_cols"] == "struct-struct",
            df_link["aux_cols"].isin(["feat-struct", "random"]),
        ],
        attrs_choices,
        default="-")

    df_inf = node_results["inference"]
    df_inf["Attributes"] = np.select(
        [
            df_inf["aux_cols"] == "feat",
            df_inf["aux_cols"] == "struct",
            df_inf["aux_cols"] == "all",
        ],
        attrs_choices,
        default="-")

    node_results_ = {"singling-out": df_so, "linkability": df_link, "inference": df_inf}
    df = pd.concat([
        res[["Dataset", "Model", "Attributes", "gt_baseline", "Risk"]]
        .assign(attack=atk)
        for atk, res in node_results_.items()
    ])
    df = df[df["Attributes"] != '-']

    fg = sns.catplot(
        data=df,
        x="Dataset",
        y="Risk",
        hue="Attributes",
        col="attack",
        kind="bar",
        palette=dict(zip(attrs_choices, sns.color_palette())),
        seed=0
    )
    fg.set_axis_labels("Dataset", "Risk")
    postproc_fig(fg, "tab_struct_attrs", "Attack: {col_name}")


def plot_so_n_pred_vars(so_results: pd.DataFrame, lvl: str) -> None:
    df = so_results

    fg = sns.relplot(
        data=df,
        x="n_cols",
        y="Risk",
        hue="Model",
        col="Dataset",
        kind="line",
        marker='o',
        palette=MODEL_COLORS,
        facet_kws={"sharex": False},
        seed=0
    )
    fg.set_axis_labels("Predicate Variables", "Risk")

    for ax in fg.axes.flat:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    postproc_fig(fg, f"so_n_pred_vars_{lvl}", "Dataset: {col_name}")


def plot_so_struct_vars(so_results: pd.DataFrame, lvl: str) -> None:
    df = so_results
    df = df[~df["Structural Attributes"].isin(["[]", "all"])].copy()
    trans = str.maketrans('_', ' ', "[']")
    df["Structural Attributes"] = df["Structural Attributes"]\
        .apply(lambda x: x.translate(trans)).str.title()

    fg = sns.catplot(
        data=df,
        x="Structural Attributes",
        y="Risk",
        hue="Model",
        col="Dataset",
        kind="bar",
        seed=0
    )
    fg.set_axis_labels("Structural Variables", "Risk")
    fg.set_xticklabels(rotation=15, horizontalalignment="right")
    postproc_fig(fg, f"so_struct_vars_{lvl}", "Dataset: {col_name}")


def main() -> None:
    results = read_results()

    tabulate_results(results)

    plot_tab_struct_attrs({atk: df for (lvl, atk), df in results.items() if lvl == "node"})

    plot_so_min_matching_syns(results, "node")

    plot_so_n_pred_vars(results[("node", "singling-out")], "node")

    plot_so_struct_vars(results[("node", "singling-out")], "node")

    plot_so_n_pred_vars(results[("graph", "singling-out")], "graph")

    plot_so_n_pred_vars(results[("edge", "singling-out")], "edge")

    plot_so_min_matching_syns(results, "edge")


if __name__ == "__main__":
    main()
