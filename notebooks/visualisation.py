import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Visualisation of results
    In this notebook, let's visualise the results of our relevance experiments, covering (1) decay-based feature transformation and (2) continuous relevance modelling.
    """)
    return


@app.cell
def _():
    import os
    import sys
    import matplotlib
    import marimo as mo
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    # define and add project root to path
    PROJECT_ROOT = os.path.abspath(os.path.dirname("__file__"))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    DATA_PATH: str = os.path.join(PROJECT_ROOT, 'data')
    print(DATA_PATH)

    # Set Arial as the font
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"

    itu_colors = [
        "#7BDBCD",  # Turquoise
        "#E47655",  # Coral
        "#7438BB",  # Purple
        "#6E7B7B",  # Dark Gray
        "#AEC4C4",  # Medium Gray
        "#D2DCDC",  # Light Gray
        "#E1E6E6",  # Ultralight Gray
        "#000000"   # Deep Black
    ]
    return DATA_PATH, Line2D, itu_colors, mo, np, os, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Decay-based features
    First, let's visualise the fitted curves. This is already implemented in the experiment script. Second, we take a look at classification performance with the decay-based features.
    """)
    return


@app.cell
def _(DATA_PATH: str, os, pd):
    classif_metrics: pd.DataFrame = pd.read_csv(os.path.join(DATA_PATH, 'output', 'decay_classif_eval.csv'))

    # take only the gradient boosting subset
    # classif_metrics = classif_metrics[classif_metrics['model_name'] == 'gradient_boosting']
    classif_metrics = classif_metrics[
        # ((classif_metrics['non_text_features'] == 'all') & (classif_metrics['encoding_features'] == 'all') & (classif_metrics['decay_features'] == 'spatio-temporal')) |
        ((classif_metrics['non_text_features'] == 'all') & (classif_metrics['encoding_features'] == 'all') & (classif_metrics['decay_features'] == 'none')) |
        ((classif_metrics['non_text_features'] == 'none') & (classif_metrics['encoding_features'] == 'none') & (classif_metrics['decay_features'] != 'none')) |
        ((classif_metrics['non_text_features'] == 'none') & (classif_metrics['encoding_features'] == 'all') & (classif_metrics['decay_features'] == 'spatio-temporal'))
    ]
    classif_metrics
    return (classif_metrics,)


@app.cell
def _(classif_metrics: "pd.DataFrame", pd):
    def print_decay_table(df: pd.DataFrame):
        df = df.copy()

        grouped_df: pd.DataFrame = (
            df.groupby(["non_text_features", "encoding_features", "decay_features"])
            .agg({
                "test_macro_f1": ["mean", "min", "max"],
                "test_roc_auc": ["mean", "min", "max"],
                "test_acc": ["mean", "min", "max"],
                "test_rmse": ["mean", "min", "max"],
                "test_mae": ["mean", "min", "max"],
                "test_r2": ["mean", "min", "max"]
            })
            .reset_index()
        )

        # Optional: flatten multi-level columns for cleaner access
        grouped_df.columns = [
            "_".join(col).strip("_") for col in grouped_df.columns.to_flat_index()
        ]
        print(grouped_df.to_latex())
        return grouped_df

    print_decay_table(df=classif_metrics)
    return


@app.cell
def _(
    DATA_PATH: str,
    Line2D,
    classif_metrics: "pd.DataFrame",
    itu_colors,
    np,
    os,
    pd,
    plt,
):
    def print_decay_feature_boxplot(df: pd.DataFrame):
        df = df.copy()

        # Map configurations to short descriptive labels
        config_labels = {
            ("all", "all", "none"): "Manual baseline",
            ("none", "none", "spatial"): "Spatial decay",
            ("none", "none", "temporal"): "Temporal decay",
            ("none", "none", "spatio-temporal"): "Spatio-temporal decay",
            ("none", "all", "spatio-temporal"): "Spatio-temporal decay\n+ event/loc encodings"
        }

        # Filter only relevant configurations
        df = df[df.apply(
            lambda r: (r["non_text_features"], r["encoding_features"], r["decay_features"]) in config_labels,
            axis=1
        )].copy()

        # Assign readable configuration labels
        df["Configuration"] = df.apply(
            lambda r: config_labels[(r["non_text_features"], r["encoding_features"], r["decay_features"])],
            axis=1
        )

        configs = list(config_labels.values())

        # Collect data for plotting
        f1_data = [df.loc[df["Configuration"] == c, "test_macro_f1"].values for c in configs]
        acc_data = [df.loc[df["Configuration"] == c, "test_acc"].values for c in configs]

        # --- Find overall best model based on Macro-F1 ---
        best_row = df.loc[df["test_macro_f1"].idxmax()]
        best_model = best_row["model_name"]
        best_config = best_row["Configuration"]
        best_value = best_row["test_macro_f1"]

        # --- Plot setup ---
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(9, 5))

        # Colors (fall back if itu_colors not defined)
        try:
            color_f1 = itu_colors[0]
            color_acc = itu_colors[1]
        except NameError:
            color_f1 = "#5DADE2"
            color_acc = "#58D68D"

        medianprops = dict(color="black", linewidth=1.2)
        meanprops = dict(marker="o", markerfacecolor="black", markeredgecolor="black", markersize=3)
        flierprops = dict(marker="x", markerfacecolor="black", markeredgecolor="black", markersize=5)

        # X positions for paired boxes
        x = np.arange(len(configs))
        width = 0.35
        pos_f1 = x - width/2
        pos_acc = x + width/2

        # --- Boxplots ---
        bp_f1 = ax.boxplot(
            f1_data,
            positions=pos_f1,
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor=color_f1, alpha=0.7),
            medianprops=medianprops,
            meanprops=meanprops,
            flierprops=flierprops,
            showmeans=True
        )

        bp_acc = ax.boxplot(
            acc_data,
            positions=pos_acc,
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor=color_acc, alpha=0.7),
            medianprops=medianprops,
            meanprops=meanprops,
            flierprops=flierprops,
            showmeans=True
        )

        # Annotate top score for each metric per configuration
        for i, cfg in enumerate(configs):
            top_f1 = df.loc[df["Configuration"] == cfg, "test_macro_f1"].max()
            top_acc = df.loc[df["Configuration"] == cfg, "test_acc"].max()
            ax.text(pos_f1[i], top_f1 + 0.005, f"{top_f1:.3f}", ha="center", va="bottom",
                    fontsize=8, color="black", fontweight="medium")
            ax.text(pos_acc[i], top_acc + 0.005, f"{top_acc:.3f}", ha="center", va="bottom",
                    fontsize=8, color="black", fontweight="medium")

        # --- Axis settings ---
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=0, ha="center", fontsize=9)
        ax.set_ylabel("Score")
        ax.set_title("Macro F1 score and accuracy across feature configurations and models")
        ax.spines[['right', 'top']].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend([bp_f1["boxes"][0], bp_acc["boxes"][0]], ["Macro F1", "Accuracy"], loc="lower right")

        # --- Custom legend including mean and outlier markers ---
        legend_elements = [
            bp_f1["boxes"][0],
            bp_acc["boxes"][0],
            Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=4, label='Mean'),
            Line2D([0], [0], color='black', marker='x', linestyle='None', markersize=5, label='Outlier')
        ]

        ax.legend(
            legend_elements,
            ["Macro F1", "Accuracy", "Mean", "Outlier"],
            loc="lower right"
        )

        fig.tight_layout()
        fig.savefig(os.path.join(DATA_PATH, 'figures', 'decay_feature_performance.pdf'), dpi=300)

    print_decay_feature_boxplot(df=classif_metrics)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Ordinal regression
    Second, let's have a look on the performance of regression instead of classification.
    """)
    return


@app.cell
def _(classif_metrics: "pd.DataFrame", pd):
    # baseline classification methods
    classif_baseline: pd.DataFrame = classif_metrics[
        ((classif_metrics['model_name'] == 'gradient_boosting') & (classif_metrics['non_text_features'] == 'all') &
         (classif_metrics['encoding_features'] == 'all') & (classif_metrics['decay_features'] == 'none')) |
        ((classif_metrics['model_name'] == 'random_forest') & (classif_metrics['non_text_features'] == 'none') &
         (classif_metrics['encoding_features'] == 'all') & (classif_metrics['decay_features'] == 'spatio-temporal'))
    ]
    classif_baseline
    return (classif_baseline,)


@app.cell
def _(DATA_PATH: str, os, pd):
    # load regression results
    regression_metrics: pd.DataFrame = pd.read_csv(os.path.join(DATA_PATH, 'output', 'regression_eval.csv'))
    regression_metrics = regression_metrics[
        ((regression_metrics['non_text_features'] == 'none') & (regression_metrics['encoding_features'] == 'all') & (regression_metrics['decay_features'] == 'spatio-temporal'))
        # ((regression_metrics['non_text_features'] == 'all') & (regression_metrics['encoding_features'] == 'all') & (regression_metrics['decay_features'] == 'none'))
    ]
    regression_metrics
    print(regression_metrics.to_latex())

    return (regression_metrics,)


@app.cell
def _(
    classif_baseline: "pd.DataFrame",
    pd,
    regression_metrics: "pd.DataFrame",
):
    def print_regression_metric_table(classif_df: pd.DataFrame, reg_df: pd.DataFrame):
        # Map readable names
        classif_df = classif_df.copy()
        reg_df = reg_df.copy()

        classif_df['name'] = classif_df['model_name'].map({
            'gradient_boosting': 'Gradient boosting',
            'random_forest': 'Random forest'
        })
        classif_df['inference'] = 'Classification'

        reg_df['name'] = reg_df['model_name'].map({
            'ridge_regression': 'Ridge regression',
            'svr': '\\gls{SVR}',
            'random_forest': 'Random forest',
            'gradient_boosting': 'Gradient boosting',
            'knn': '\\gls{kNN}',
        })
        reg_df['inference'] = 'Regression'

        # Select and rename columns
        COLUMNS = ['name', 'inference', 'test_rmse', 'test_mae', 'test_r2', 'test_corr']
        COLNAMES = {
            'name': 'Model',
            'inference': 'Configuration',
            'test_rmse': 'RMSE',
            'test_mae': 'MAE',
            'test_r2': '$R^2$',
            'test_corr': '$\\rho$'
        }

        df = pd.concat([classif_df[COLUMNS], reg_df[COLUMNS]], ignore_index=True)
        df = df.rename(columns=COLNAMES)

        # Round metrics for readability
        metric_cols = ['RMSE', 'MAE', '$R^2$', '$\\rho$']
        df[metric_cols] = df[metric_cols].map(lambda x: f"{x:.3f}")

        # Sort by correlation, highest first
        df = df.sort_values(by='$\\rho$', ascending=False)

        # Format LaTeX table
        latex_table = df.to_latex(
            index=False,
            escape=False,
            column_format='llcccc',
            caption="Evaluation metrics for classification and regression models.",
            label="tab:regression-metrics",
            bold_rows=False
        )
        print(latex_table)

    print_regression_metric_table(classif_df=classif_baseline, reg_df=regression_metrics)
    return


if __name__ == "__main__":
    app.run()
