import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Spatio-temporal, decay-based relevance regression
    This notebook contains an investigation on how spatiotemporal is applicable to disaster relevance. The work is adapted from Andreas Kramer's notebook `06_decay_feature_impact_rq2.ipynb`.

    Author: David Hanny, November 2025
    """)
    return


@app.cell
def _():
    import os
    import sys
    import pickle
    import json
    import warnings
    import matplotlib
    import marimo as mo
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    # import statsmodels.api as sm
    from typing import Optional, Tuple
    from scipy.optimize import curve_fit
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from scipy.stats import spearmanr
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.metrics import (accuracy_score,
                                 precision_recall_fscore_support,
                                 root_mean_squared_error,
                                 mean_absolute_error,
                                 roc_auc_score,
                                 r2_score,
                                 mean_squared_log_error,
                                 mean_absolute_percentage_error)
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.svm import SVC, SVR
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                  RandomForestRegressor, GradientBoostingRegressor)
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from tqdm import tqdm

    warnings.filterwarnings('ignore')  # suppress all warnings

    # define and add project root to path
    PROJECT_ROOT = os.path.abspath(os.path.dirname("__file__"))
    print(PROJECT_ROOT)
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    DATA_PATH: str = os.path.join(PROJECT_ROOT, 'data')
    print(DATA_PATH)
    from src.model_training.classification_head import optimise_model  # noqa
    from src.model_training.regression_head import optimise_regression_model  # noqa

    # plotting styles
    # Set Arial as the font
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"

    itu_colors = [
        "#7BDBCD",  # Turquoise
        "#E47655",  # Coral
        "#6E7B7B",  # Dark Gray
        "#7438BB",  # Purple
        "#AEC4C4",  # Medium Gray
        "#D2DCDC",  # Light Gray
        "#E1E6E6",  # Ultralight Gray
        "#000000"   # Deep Black
    ]


    # Encode np.ndarrays to lists for JSON storage
    class NumpyEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    return (
        Axes,
        DATA_PATH,
        Figure,
        GaussianNB,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        KNeighborsClassifier,
        KNeighborsRegressor,
        LogisticRegression,
        NumpyEncoder,
        Optional,
        RandomForestClassifier,
        RandomForestRegressor,
        Ridge,
        SVC,
        SVR,
        Tuple,
        accuracy_score,
        curve_fit,
        itu_colors,
        json,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_log_error,
        mo,
        np,
        optimise_model,
        optimise_regression_model,
        os,
        pd,
        plt,
        precision_recall_fscore_support,
        r2_score,
        roc_auc_score,
        root_mean_squared_error,
        spearmanr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load data
    Next up, we load the anonymised data.
    """)
    return


@app.cell
def _(DATA_PATH: str, os, pd):
    df_train: pd.DataFrame = pd.read_parquet(os.path.join(DATA_PATH, 'input', 'train_data_public.parquet'))
    df_test: pd.DataFrame = pd.read_parquet(os.path.join(DATA_PATH, 'input', 'test_data_public.parquet'))
    print(df_train.shape, df_test.shape)
    df_train
    return df_test, df_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Decay functions
    We can now define and fit the decay functions.
    """)
    return


@app.cell
def _(
    Axes,
    Figure,
    Optional,
    Tuple,
    curve_fit,
    itu_colors,
    np,
    pd,
    plt,
    r2_score,
):
    def linear_decay(d, a, b): return a - b * d
    def exp_decay(d, a, b): return a * np.exp(-b * d)
    def inverse_decay(d, a, b): return a / (b + d)
    def stretched_exp_decay(d, a, b, c): return a * np.exp(- (b * d)**c)


    def fit_decay_function(
        df: pd.DataFrame,
        distance_col: str,
        label_col: Optional[str] = "label_score",
        bin_size: Optional[str] = 50,
        ax: Optional[Axes] = None
    ) -> Tuple[dict, Figure, Axes]:
        """Fit multiple decay functions (linear, exponential, inverse) to binned average relevance scores
        as a function of spatial distance.

        Args:
            df (pd.DataFrame): Input DataFrame containing spatial distance and label score columns.
            distance_col (str): Column name representing distance from the event in kilometers.
            label_col (str, optional): Column name representing the relevance or label score. Defaults to "label_score".
            bin_size (int, optional): Number of samples within each bin. Defaults to 50.
            ax (Axes, optional): Matplotlib axis to use for plotting. Defaults to None.

        Returns:
            Tuple[dict, Figure, Axes]:
                - Dictionary containing fitted parameters and R² scores for each decay model.
                - Matplotlib Figure and Axes objects containing the plot.
        """
        # Filter valid distances
        df_valid = df[df[distance_col] >= 0].copy()
        print(f'Found {len(df_valid)} to fit the decay function.')

        # Bin and aggregate
        n_bins: int = len(df_valid) // bin_size
        df_valid["bin"], bins = pd.qcut(df_valid[distance_col], q=n_bins, retbins=True, duplicates="drop")
        # df_valid["bin"], bins = pd.cut(df_valid[distance_col], bins=n_bins, retbins=True)

        grouped: pd.DataFrame = (
            df_valid.groupby("bin", observed=False)
            .agg(
                avg_relevance=(label_col, "mean"),
                count=(label_col, "count"),
                mid_point=(distance_col, "mean"),
            )
        )

        x = grouped['mid_point'].values
        y = grouped['avg_relevance'].values

        # define decay models
        models: dict = {
            "Linear": (linear_decay, None),
            "Exponential": (exp_decay, [1.0, 0.01]),
            "Inverse": (inverse_decay, [1.0, 1.0]),
            "Stretched exponential": (stretched_exp_decay, [1.0, 0.01, 0.5])
        }

        # LOESS fit
        # smoothed: np.ndarray = sm.nonparametric.lowess(df_valid[label_col], df_valid[distance_col], return_sorted=True)

        # placeholder for results
        fit_results = {}
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.figure

        # Plot the aggregated data based on bins
        ax.scatter(x, y, color="black", label=f"Data bin (n={bin_size})", s=20, zorder=3)
        plot_colors = itu_colors[:len(models)] if "itu_colors" in globals() else ["#5DADE2", "#58D68D", "#F5B041", "#AF7AC5"]
        # ax.plot(smoothed[:, 0], smoothed[:, 1], label="LOESS Smoother", color="red")

        # iterate over each decay function, fit the curve, and evaluate fit
        for i, (name, (func, p0)) in enumerate(models.items()):
            color = plot_colors[i % len(plot_colors)]
            try:
                params, _ = curve_fit(func, df_valid[distance_col], df_valid[label_col], p0=p0, maxfev=10000)  # non-linear least squares to fit a function  # noqa
                y_pred = func(x, *params)
                r2 = r2_score(y, y_pred)
                fit_results[name] = {"params": params, "r2": r2}

                # Plot decay curve
                label: str = f'{name} (R²={r2:.3f})'
                label = label.replace('Stretched exponential', 'Stretched exp.')
                ax.plot(x, y_pred, label=label, color=color, linewidth=2.0, zorder=2)
            except Exception as e:
                print(f"{name} fitting failed:", e)

        ax.set_xlabel(f"{distance_col.replace('_', ' ').title()}")
        ax.set_ylabel("Mean relevance score")
        ax.set_title("Decay function fits across distance bins")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=True, fontsize=8, loc="upper right")
        # ax.grid(axis="y", linestyle="--", alpha=0.6)
        # ax.legend(frameon=False, fontsize=8, loc="upper right")

        return fit_results, fig, ax
    return fit_decay_function, stretched_exp_decay


@app.cell
def _(
    DATA_PATH: str,
    NumpyEncoder,
    df_test: "pd.DataFrame",
    df_train: "pd.DataFrame",
    fit_decay_function,
    json,
    os,
    plt,
):
    decay_fig, decay_axes = plt.subplots(2, 1, figsize=(6, 8), sharey=False)
    spatial_fit_results, _, _ = fit_decay_function(
        df=df_train, distance_col='event_distance_km', bin_size=70, ax=decay_axes[0]
    )
    print(spatial_fit_results)
    decay_axes[0].set_title('Spatial decay')
    decay_axes[0].set_xlabel('Geographic distance to impact site (km)')
    decay_axes[0].set_ylim([0, 0.65])

    df_train['event_distance_h_abs'] = df_train['event_distance_h'].abs()
    df_test['event_distance_h_abs'] = df_test['event_distance_h'].abs()
    temp_fit_results, _, _ = fit_decay_function(
        df=df_train, distance_col='event_distance_h_abs', bin_size=70, ax=decay_axes[1]
    )
    decay_axes[1].set_title('Temporal decay')
    decay_axes[1].set_xlabel('Temporal distance to disaster event (h)')
    decay_axes[1].set_ylabel('')
    decay_axes[1].set_ylim([0, 0.65])
    print(temp_fit_results)

    decay_fig.tight_layout()
    decay_fig.savefig(os.path.join(DATA_PATH, 'figures', 'decay_curves.pdf'), dpi=300)
    decay_fig.show()

    # Save fitted parameters as JSON
    decay_params = {
        "spatial_decay": spatial_fit_results,
        "temporal_decay": temp_fit_results
    }

    output_path = os.path.join(DATA_PATH, 'output', 'decay_fit_results.json')
    with open(output_path, 'w') as f:
        json.dump(decay_params, f, indent=4, cls=NumpyEncoder)

    print(f"Saved fitted decay parameters to {output_path}")
    return spatial_fit_results, temp_fit_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can go ahead and apply the best-fitting decay functions.
    """)
    return


@app.cell
def _(
    df_test: "pd.DataFrame",
    df_train: "pd.DataFrame",
    np,
    pd,
    spatial_fit_results,
    stretched_exp_decay,
    temp_fit_results,
):
    def apply_distance_decay(df: pd.DataFrame) -> pd.DataFrame:
        df['W_event_distance_km'] = df['event_distance_km'].apply(
            lambda x: stretched_exp_decay(x, *spatial_fit_results['Stretched exponential']['params'])
        )
        df['W_event_distance_h'] = df['event_distance_h'].apply(
            lambda x: stretched_exp_decay(np.abs(x), *temp_fit_results['Stretched exponential']['params'])
        )
        return df


    df_train_decay = apply_distance_decay(df=df_train)
    df_test_decay = apply_distance_decay(df=df_test)
    return df_test_decay, df_train_decay


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Model evaluation with decay features
    With the features fitted, we can now first fit classification models based on decay-based features.
    """)
    return


@app.cell
def _(
    DATA_PATH: str,
    GaussianNB,
    GradientBoostingClassifier,
    KNeighborsClassifier,
    LogisticRegression,
    RandomForestClassifier,
    SVC,
    accuracy_score,
    df_test_decay,
    df_train_decay,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
    np,
    optimise_model,
    os,
    pd,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
    spearmanr,
):
    def evaluate_decay_features(train_df: pd.DataFrame, test_df: pd.DataFrame, models: dict):
        RESULTS_CSV_PATH: str = os.path.join(DATA_PATH, 'output', 'decay_classif_eval.csv')

        if not os.path.exists(RESULTS_CSV_PATH):
            print('\n🤖 EVALUATING CLASSIFICATION MODELS\n--')

            DECAY_FEATURES: dict = {
                'None': [],
                'Spatial': ['W_event_distance_km'],
                'Temporal': ['W_event_distance_h'],
                'Spatio-temporal': ['W_event_distance_km', 'W_event_distance_h'],
            }

            NON_TEXT_FEATURES: dict = {
                'All': ['event_distance_km',
                        'event_distance_h',
                        'n_disaster_tweets_1km',
                        'n_disaster_tweets_10km',
                        'n_disaster_tweets_50km',
                        'n_disaster_tweets_10000km'],
                'Distance': ['event_distance_km', 'event_distance_h'],
                'None': []
            }

            # Define the event and location encoding options
            ENCODING_OPTIONS: dict = {
                "None": lambda df: np.empty((df.shape[0], 0)),  # returns an empty array so hstack works
                "All": lambda df: np.hstack([
                    np.vstack(df['event_type_encoding'].values),
                    df[['sphere_x', 'sphere_y', 'sphere_z']].values
                ])
            }

            # Feature preparation independent of sets
            y_train: np.ndarray = train_df['int_label'].values

            # Load OOF predictions for training the meta-learner
            oof_preds_text = np.load(os.path.join(DATA_PATH, 'input', 'oof_preds_text_train.npy'))
            print(f'Out-of-of fold predictions: {oof_preds_text[:5]}')

            # Load softmax predictions on test data
            test_preds_text = np.load(os.path.join(DATA_PATH, 'input', 'preds_text_test.npy'))
            print(f"Loaded test softmax probabilities from file: {test_preds_text}")

            # placeholders
            meta_training_results: list[dict] = []

            # Iterate over all feature combinations
            for non_text_name, non_text_feature_set in NON_TEXT_FEATURES.items():
                for decay_name, decay_feature_set in DECAY_FEATURES.items():
                    for encoding_name, encoding_option in ENCODING_OPTIONS.items():

                        # --- skip all undesired combinations ---
                        if not (
                            (non_text_name.lower() == 'all' and encoding_name.lower() == 'all')
                            or (non_text_name.lower() == 'none' and encoding_name.lower() == 'none' and decay_name.lower() != 'none')
                            or (non_text_name.lower() == 'none' and encoding_name.lower() == 'all' and decay_name.lower() != 'none')
                        ):
                            continue

                        print(f'Evaluating feature combination: {non_text_name}+{encoding_name}+{decay_name} ...')

                        # Construct the feature matrix for the training data
                        X_base_train: np.ndarray = train_df[non_text_feature_set + decay_feature_set].values  # base features
                        X_event_train: np.ndarray = encoding_option(train_df)  # event encoding features
                        X_train_non_text: np.ndarray = np.hstack([X_base_train, X_event_train])

                        # Construct the feature matrix for the test data
                        X_base_test: np.ndarray = test_df[non_text_feature_set + decay_feature_set].values  # base features
                        X_event_test: np.ndarray = encoding_option(test_df)  # event encoding features
                        X_test_non_text: np.ndarray = np.hstack([X_base_test, X_event_test])

                        # Features based on text probabilities and non-text features
                        # shape: [n_train, n_classes+9]
                        meta_part_features_train: np.ndarray = np.concatenate([X_train_non_text, oof_preds_text], axis=1)  # noqa
                        print(f'Feature vector size (train): {meta_part_features_train.shape}')

                        # shape: [n_train, n_classes+12]  # noqa
                        meta_part_features_test: np.ndarray = np.concatenate([X_test_non_text, test_preds_text], axis=1)
                        print(f'Feature vector size (test): {meta_part_features_test.shape}')

                        # Fit and store all models
                        for model_name, model in models.items():
                            # Train a model on the OOF probabilities of the text model and the non-text features
                            part_meta_model, part_meta_params, part_meta_f1 = optimise_model(model, meta_part_features_train, y_train)

                            # Make test predictions
                            predictions: np.ndarray = part_meta_model.predict(meta_part_features_test)
                            probs: np.ndarray = part_meta_model.predict_proba(meta_part_features_test)
                            test_df[
                                f'pred_{model_name}_{non_text_name.lower()}_{encoding_name.lower()}_{decay_name.lower()}'
                            ] = predictions  # noqa

                            # -- Classification evaluation --
                            prec, rec, f1, support = precision_recall_fscore_support(test_df['int_label'], predictions, average='macro')
                            roc_auc = roc_auc_score(y_true=test_df['int_label'], y_score=probs, multi_class='ovr')
                            acc = accuracy_score(y_true=test_df['int_label'], y_pred=predictions)

                            # -- Regression evaluation --
                            scale_mapping = np.array([0.0, 0.5, 1.0])
                            scaled_predictions: np.ndarray = scale_mapping[predictions]
                            rmse: float = root_mean_squared_error(y_true=test_df['label_score'], y_pred=scaled_predictions)
                            mae: float = mean_absolute_error(y_true=test_df['label_score'], y_pred=scaled_predictions)
                            r2: float = r2_score(y_true=test_df['label_score'], y_pred=scaled_predictions)
                            corr, corr_pval = spearmanr(a=test_df['label_score'], b=scaled_predictions)
                            msle: float = mean_squared_log_error(y_true=test_df['label_score'], y_pred=scaled_predictions)
                            mape: float = mean_absolute_percentage_error(y_true=test_df['label_score'], y_pred=scaled_predictions)

                            meta_training_results.append({
                                'model_name': model_name,
                                'method': 'partial',
                                'non_text_features': non_text_name.lower(),
                                'encoding_features': encoding_name.lower(),
                                'decay_features': decay_name.lower(),
                                'params': part_meta_params,
                                'cv_macro_f1': part_meta_f1,
                                'model': part_meta_model,
                                "test_macro_prec": prec,
                                "test_macro_rec": rec,
                                "test_macro_f1": f1,
                                'test_acc': acc,
                                "test_roc_auc": roc_auc,
                                "test_rmse": rmse,
                                "test_mae": mae,
                                "test_r2": r2,
                                'test_corr': corr,
                                'test_corr_p': corr_pval,
                                'test_msle': msle,
                                'test_mape': mape
                            })

                            print(f'- {model_name}: M-F1={f1:.2f}, ROC-AUC={roc_auc:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}')

            # text-only baseline evaluation
            print("\nEvaluating text-only classifier...")

            # predicted label = argmax of softmax
            text_only_pred = np.argmax(test_preds_text, axis=1)

            # classification metrics
            prec, rec, f1, support = precision_recall_fscore_support(test_df['int_label'], text_only_pred, average='macro')
            acc = accuracy_score(y_true=test_df['int_label'], y_pred=text_only_pred)
            roc_auc = roc_auc_score(y_true=test_df['int_label'], y_score=test_preds_text, multi_class='ovr')

            # regression metrics (again using label_score)
            scale_mapping = np.array([0.0, 0.5, 1.0])
            scaled_text_only_pred: np.ndarray = scale_mapping[text_only_pred]
            rmse = root_mean_squared_error(y_true=test_df['label_score'], y_pred=scaled_text_only_pred)
            mae = mean_absolute_error(y_true=test_df['label_score'], y_pred=scaled_text_only_pred)
            r2 = r2_score(y_true=test_df['label_score'], y_pred=scaled_text_only_pred)
            corr, corr_pval = spearmanr(a=test_df['label_score'], b=scaled_text_only_pred)
            msle = mean_squared_log_error(y_true=test_df['label_score'], y_pred=scaled_text_only_pred)
            mape = mean_absolute_percentage_error(y_true=test_df['label_score'], y_pred=scaled_text_only_pred)

            meta_training_results.append({
                'model_name': 'text_only_softmax',
                'method': 'text',
                'non_text_features': 'none',
                'encoding_features': 'none',
                'decay_features': 'none',
                'params': None,
                'cv_macro_f1': None,
                'model': None,
                "test_macro_prec": prec,
                "test_macro_rec": rec,
                "test_macro_f1": f1,
                'test_acc': acc,
                "test_roc_auc": roc_auc,
                "test_rmse": rmse,
                "test_mae": mae,
                "test_r2": r2,
                'test_corr': corr,
                'test_corr_p': corr_pval,
                'test_msle': msle,
                'test_mape': mape
            })

            print(f'Text-only: M-F1={f1:.2f}, ROC-AUC={roc_auc:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}')

            meta_val_result_df: pd.DataFrame = pd.DataFrame.from_dict(meta_training_results)
            meta_val_result_df.to_csv(RESULTS_CSV_PATH, index=False)
            # with open(os.path.join(DATA_PATH, 'output', 'decay_classif_models.pickle'), 'wb') as f:
            #     pickle.dump(meta_training_results, f)
            return meta_val_result_df
        else:
            print(f'Existing results for decay evaluation found. Reading from: {RESULTS_CSV_PATH}')
            return pd.read_csv(RESULTS_CSV_PATH)


    # Models to evaluate
    classif_models: dict = {
        "logistic_regression": LogisticRegression(random_state=1),
        "random_forest": RandomForestClassifier(random_state=2),
        "svm": SVC(probability=True, random_state=3),
        "gradient_boosting": GradientBoostingClassifier(random_state=5),
        "knn": KNeighborsClassifier(),
        "naive_bayes": GaussianNB()
    }

    test_results: pd.DataFrame = evaluate_decay_features(train_df=df_train_decay, test_df=df_test_decay,
                                                         models=classif_models)
    test_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Regression
    Lastly, we run the regression experiments.
    """)
    return


@app.cell
def _(
    DATA_PATH: str,
    GradientBoostingRegressor,
    KNeighborsRegressor,
    RandomForestRegressor,
    Ridge,
    SVR,
    accuracy_score,
    df_test_decay,
    df_train_decay,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
    np,
    optimise_regression_model,
    os,
    pd,
    precision_recall_fscore_support,
    r2_score,
    root_mean_squared_error,
    spearmanr,
):
    def evaluate_regression(train_df: pd.DataFrame, test_df: pd.DataFrame, models: dict):
        RESULTS_CSV_PATH: str = os.path.join(DATA_PATH, 'output', 'regression_eval.csv')

        if not os.path.exists(RESULTS_CSV_PATH):
            print('\n📈 EVALUATING REGRESSION MODELS\n--')

            DECAY_FEATURES: dict = {
                'None': [],
                'Spatial': ['W_event_distance_km'],
                'Temporal': ['W_event_distance_h'],
                'Spatio-temporal': ['W_event_distance_km', 'W_event_distance_h'],
            }

            NON_TEXT_FEATURES: dict = {
                'All': ['event_distance_km',
                        'event_distance_h',
                        'n_disaster_tweets_1km',
                        'n_disaster_tweets_10km',
                        'n_disaster_tweets_50km',
                        'n_disaster_tweets_10000km'],
                'Distance': ['event_distance_km', 'event_distance_h'],
                'None': []
            }

            # Define the event and location encoding options
            ENCODING_OPTIONS: dict = {
                "None": lambda df: np.empty((df.shape[0], 0)),  # returns an empty array so hstack works
                "All": lambda df: np.hstack([
                    np.vstack(df['event_type_encoding'].values),
                    df[['sphere_x', 'sphere_y', 'sphere_z']].values
                ])
            }

            # Feature preparation independent of sets
            y_train: np.ndarray = train_df['label_score'].values


            # Load OOF predictions for training the meta-learner
            oof_preds_text = np.load(os.path.join(DATA_PATH, 'input', 'oof_preds_text_train.npy'))
            print(f'Out-of-of fold predictions: {oof_preds_text[:5]}')

            # Load softmax predictions on test data
            test_preds_text = np.load(os.path.join(DATA_PATH, 'input', 'preds_text_test.npy'))
            print(f"Loaded test softmax probabilities from file: {test_preds_text}")

            # placeholders
            meta_training_results: list[dict] = []

            # Iterate over all feature combinations
            for non_text_name, non_text_feature_set in NON_TEXT_FEATURES.items():
                for decay_name, decay_feature_set in DECAY_FEATURES.items():
                    for encoding_name, encoding_option in ENCODING_OPTIONS.items():

                        # --- skip all undesired combinations ---
                        if not (
                            (non_text_name.lower() == 'all' and encoding_name.lower() == 'all')
                            or (non_text_name.lower() == 'none' and encoding_name.lower() == 'none' and decay_name.lower() != 'none')
                            or (non_text_name.lower() == 'none' and encoding_name.lower() == 'all' and decay_name.lower() != 'none')
                        ):
                            continue

                        print(f'Evaluating regression with combination: {non_text_name}+{encoding_name}+{decay_name} ...')

                        # Construct the feature matrix for the training data
                        X_base_train: np.ndarray = train_df[non_text_feature_set + decay_feature_set].values  # base features
                        X_event_train: np.ndarray = encoding_option(train_df)  # event encoding features
                        X_train_non_text: np.ndarray = np.hstack([X_base_train, X_event_train])

                        # Construct the feature matrix for the test data
                        X_base_test: np.ndarray = test_df[non_text_feature_set + decay_feature_set].values  # base features
                        X_event_test: np.ndarray = encoding_option(test_df)  # event encoding features
                        X_test_non_text: np.ndarray = np.hstack([X_base_test, X_event_test])

                        # Features based on text probabilities and non-text features
                        # shape: [n_train, n_classes+9]
                        meta_part_features_train: np.ndarray = np.concatenate([X_train_non_text, oof_preds_text], axis=1)
                        print(f'Feature vector size (train): {meta_part_features_train.shape}')

                         # shape: [n_train, n_classes+12]
                        meta_part_features_test: np.ndarray = np.concatenate([X_test_non_text, test_preds_text], axis=1)
                        print(f'Feature vector size (test): {meta_part_features_test.shape}')

                        # Fit and store all models
                        for model_name, model in models.items():
                            # Train a model on the OOF probabilities of the text model and the non-text features
                            part_meta_model, part_meta_params, part_meta_rmse = optimise_regression_model(
                                model, meta_part_features_train, y_train
                            )

                            # Make test predictions
                            predictions: np.ndarray = part_meta_model.predict(meta_part_features_test)
                            predictions = np.abs(predictions)
                            test_df[
                                f'reg_{model_name}_{non_text_name.lower()}_{encoding_name.lower()}_{decay_name.lower()}'
                            ] = predictions

                            class_levels: np.ndarray = np.array([0.0, 0.5, 1.0])
                            pred_class: np.ndarray = (
                                class_levels[np.abs(predictions[:, None] - class_levels).argmin(axis=1)] * 2
                            ).astype(int)

                            # classification metrics
                            prec, rec, f1, support = precision_recall_fscore_support(
                                test_df['int_label'], pred_class, average='macro'
                            )
                            acc = accuracy_score(y_true=test_df['int_label'], y_pred=pred_class)

                            # -- regression metrics --
                            rmse: float = root_mean_squared_error(y_true=test_df['label_score'], y_pred=predictions)
                            mae: float = mean_absolute_error(y_true=test_df['label_score'], y_pred=predictions)
                            r2: float = r2_score(y_true=test_df['label_score'], y_pred=predictions)
                            corr, corr_pval = spearmanr(a=test_df['label_score'], b=predictions)
                            msle: float = mean_squared_log_error(y_true=test_df['label_score'], y_pred=predictions)
                            mape: float = mean_absolute_percentage_error(y_true=test_df['label_score'], y_pred=predictions)

                            meta_training_results.append({
                                'model_name': model_name,
                                'method': 'partial',
                                'non_text_features': non_text_name.lower(),
                                'encoding_features': encoding_name.lower(),
                                'decay_features': decay_name.lower(),
                                'params': part_meta_params,
                                'cv_rmse': part_meta_rmse,
                                'model': part_meta_model,
                                "test_macro_prec": prec,
                                "test_macro_rec": rec,
                                "test_macro_f1": f1,
                                'test_acc': acc,
                                "test_rmse": rmse,
                                "test_mae": mae,
                                "test_r2": r2,
                                'test_corr': corr,
                                'test_corr_p': corr_pval,
                                'test_msle': msle,
                                'test_mape': mape
                            })

                            print(f'- {model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}')

            # text-only baseline evaluation
            print("\nEvaluating text-only classifier for regression ...")

            # predicted score = 0*p0 + 0.5*p1 + 1*p3
            reg_weights: np.ndarray = np.array([0.0, 0.5, 1.0])
            text_only_reg_score: np.ndarray = test_preds_text @ reg_weights
            text_only_reg_class: np.ndarray = (
                reg_weights[np.abs(text_only_reg_score[:, None] - reg_weights).argmin(axis=1)] * 2
            ).astype(int)

            # classification metrics
            prec, rec, f1, support = precision_recall_fscore_support(test_df['int_label'], text_only_reg_class, average='macro')
            acc = accuracy_score(y_true=test_df['int_label'], y_pred=text_only_reg_class)

            # regression metrics (again using label_score)
            rmse = root_mean_squared_error(y_true=test_df['label_score'], y_pred=text_only_reg_score)
            mae = mean_absolute_error(y_true=test_df['label_score'], y_pred=text_only_reg_score)
            r2 = r2_score(y_true=test_df['label_score'], y_pred=text_only_reg_score)
            corr, corr_pval = spearmanr(a=test_df['label_score'], b=text_only_reg_score)
            msle = mean_squared_log_error(y_true=test_df['label_score'], y_pred=text_only_reg_score)
            mape = mean_absolute_percentage_error(y_true=test_df['label_score'], y_pred=text_only_reg_score)

            meta_training_results.append({
                'model_name': 'text_only_softmax',
                'method': 'text',
                'non_text_features': 'none',
                'encoding_features': 'none',
                'decay_features': 'none',
                'params': None,
                'cv_macro_f1': None,
                'model': None,
                "test_macro_prec": prec,
                "test_macro_rec": rec,
                "test_macro_f1": f1,
                'test_acc': acc,
                "test_rmse": rmse,
                "test_mae": mae,
                "test_r2": r2,
                'test_corr': corr,
                'test_corr_p': corr_pval,
                'test_msle': msle,
                'test_mape': mape
            })

            print(f'Text-only regression: RMSE={rmse:.2f}, MAE={mae:.2f}')

            meta_val_result_df: pd.DataFrame = pd.DataFrame.from_dict(meta_training_results)
            meta_val_result_df.to_csv(RESULTS_CSV_PATH, index=False)
            # with open(os.path.join(DATA_PATH, 'output', 'regression_models.pickle'), 'wb') as f:
            #     pickle.dump(meta_training_results, f)
            return meta_val_result_df
        else:
            print(f'Existing results for regression evaluation found. Reading from: {RESULTS_CSV_PATH}')
            return pd.read_csv(RESULTS_CSV_PATH)


    # Models to evaluate
    regression_models: dict = {
        "ridge_regression": Ridge(random_state=1),
        "random_forest": RandomForestRegressor(random_state=2),
        "svr": SVR(),
        "gradient_boosting": GradientBoostingRegressor(random_state=5),
        "knn": KNeighborsRegressor()
    }

    regression_results: pd.DataFrame = evaluate_regression(train_df=df_train_decay, test_df=df_test_decay,
                                                           models=regression_models)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At the end, we save all results for visualisation.
    """)
    return


@app.cell
def _(DATA_PATH: str, df_test_decay, df_train_decay, os):
    if not os.path.exists(os.path.join(DATA_PATH, 'output', 'decay_train_df.parquet')):
        df_train_decay.to_parquet(os.path.join(DATA_PATH, 'output', 'decay_train_df.parquet'))
    if not os.path.exists(os.path.join(DATA_PATH, 'output', 'decay_test_df.parquet')):
        df_test_decay.to_parquet(os.path.join(DATA_PATH, 'output', 'decay_test_df.parquet'))
    return


if __name__ == "__main__":
    app.run()
