import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from temp.model import (
    DEFAULT_DATA_PATH,
    PROJECT_ROOT,
    build_preprocessor,
    get_feature_importance_df,
    prepare_dataframe,
    regression_metrics,
    resolve_id_column,
    resolve_target_column,
    to_builtin,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "model_outputs_fixed"


def build_fixed_model_specs(random_state: int, n_jobs: int) -> list[dict]:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError("xgboost is not installed in the active environment.") from exc

    return [
        {
            "model_name": "linear_regression",
            "scale_numeric": True,
            "params": {"fit_intercept": True},
            "estimator": LinearRegression(fit_intercept=True),
        },
        {
            "model_name": "elastic_net",
            "scale_numeric": True,
            "params": {
                "alpha": 10.0,
                "l1_ratio": 0.8,
                "fit_intercept": True,
            },
            "estimator": ElasticNet(
                alpha=10.0,
                l1_ratio=0.8,
                fit_intercept=True,
                random_state=random_state,
                max_iter=100000,
                tol=1e-4,
            ),
        },
        {
            "model_name": "random_forest",
            "scale_numeric": False,
            "params": {
                "n_estimators": 250,
                "max_depth": 5,
                "min_samples_split": 6,
                "min_samples_leaf": 1,
                "max_features": 0.3,
                "bootstrap": True,
                "max_samples": None,
                "ccp_alpha": 0.001,
            },
            "estimator": RandomForestRegressor(
                n_estimators=250,
                max_depth=5,
                min_samples_split=6,
                min_samples_leaf=1,
                max_features=0.3,
                bootstrap=True,
                max_samples=None,
                ccp_alpha=0.001,
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        },
        {
            "model_name": "xgboost",
            "scale_numeric": False,
            "params": {
                "n_estimators": 200,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.65,
                "colsample_bytree": 0.4,
                "min_child_weight": 9,
                "gamma": 0.8,
                "reg_alpha": 12.0,
                "reg_lambda": 8.0,
            },
            "estimator": XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.65,
                colsample_bytree=0.4,
                min_child_weight=9,
                gamma=0.8,
                reg_alpha=12.0,
                reg_lambda=8.0,
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=random_state,
                n_jobs=n_jobs,
                tree_method="hist",
                verbosity=0,
            ),
        },
    ]


def build_fixed_pipeline(
    feature_names: list[str],
    estimator,
    scale_numeric: bool,
) -> Pipeline:
    preprocessor = build_preprocessor(feature_names, scale_numeric=scale_numeric)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )


def evaluate_fixed_pipeline_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int,
    random_state: int,
    n_jobs: int,
) -> tuple[float, float]:
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="neg_root_mean_squared_error",
            cv=cv_splitter,
            n_jobs=n_jobs,
        )
    rmse_scores = -scores
    return float(np.mean(rmse_scores)), float(np.std(rmse_scores))


def save_fixed_model_outputs(
    output_dir: Path,
    fitted_pipeline: Pipeline,
    params: dict,
    cv_rmse_mean: float,
    cv_rmse_std: float,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    pred_train: np.ndarray,
    pred_valid: np.ndarray,
    id_train: pd.Series,
    id_valid: pd.Series,
    random_state: int,
) -> tuple[dict[str, float], dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(fitted_pipeline, output_dir / "model.pkl")
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(to_builtin(params), f, ensure_ascii=False, indent=2)

    train_metrics = regression_metrics(y_train, pred_train)
    valid_metrics = regression_metrics(y_valid, pred_valid)
    pd.DataFrame(
        [
            {"split": "train", **train_metrics},
            {"split": "valid", **valid_metrics},
        ]
    ).to_csv(output_dir / "metrics_train_valid.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {"sample_id": id_train.values, "y_true": y_train.values, "y_pred": pred_train}
    ).to_csv(output_dir / "pred_train.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {"sample_id": id_valid.values, "y_true": y_valid.values, "y_pred": pred_valid}
    ).to_csv(output_dir / "pred_valid.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "mean_test_rmse": cv_rmse_mean,
                "std_test_rmse": cv_rmse_std,
                **to_builtin(params),
            }
        ]
    ).to_csv(output_dir / "cv_results.csv", index=False, encoding="utf-8-sig")

    get_feature_importance_df(
        fitted_pipeline=fitted_pipeline,
        X_valid=X_valid,
        y_valid=y_valid,
        random_state=random_state,
    ).to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    return train_metrics, valid_metrics


def run_all_models_fixed(
    data_path: Path,
    output_root: Path,
    test_size: float,
    random_state: int,
    cv: int,
    n_jobs: int,
    target_column: str | None = None,
    drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError(f"Input data is empty: {data_path}")
    target_column = resolve_target_column(df, requested_target=target_column)
    id_column = resolve_id_column(df)
    df = prepare_dataframe(df, target_column=target_column, id_column=id_column)

    reserved_columns = {target_column}
    if id_column:
        reserved_columns.add(id_column)

    drop_columns = drop_columns or []
    unknown_drop_columns = [col for col in drop_columns if col not in df.columns]
    if unknown_drop_columns:
        raise KeyError(f"Unknown drop columns: {unknown_drop_columns}")

    feature_columns = [
        col for col in df.columns if col not in reserved_columns and col not in set(drop_columns)
    ]
    if not feature_columns:
        raise ValueError("No feature columns remain after exclusions.")

    all_nan_features = [col for col in feature_columns if df[col].isna().all()]
    if all_nan_features:
        raise ValueError(f"Feature columns are entirely missing: {all_nan_features}")

    X = df[feature_columns].copy()
    y = df[target_column].copy()
    sample_id = (
        df[id_column].astype(str).reset_index(drop=True)
        if id_column
        else pd.Series(df.index.astype(str), name="sample_id")
    )

    effective_cv = min(cv, len(df))
    if effective_cv < 2:
        raise ValueError("At least two samples are required for cross-validation.")

    X_train, X_valid, y_train, y_valid, id_train, id_valid = train_test_split(
        X,
        y,
        sample_id,
        test_size=test_size,
        random_state=random_state,
    )

    model_specs = build_fixed_model_specs(random_state=random_state, n_jobs=n_jobs)

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feature_columns}).to_csv(
        output_root / "used_features.csv", index=False, encoding="utf-8-sig"
    )

    summary_rows = []
    for spec in model_specs:
        model_name = spec["model_name"]
        print(f"Training model: {model_name}")
        model_dir = output_root / model_name

        try:
            pipeline = build_fixed_pipeline(
                feature_names=X_train.columns.tolist(),
                estimator=spec["estimator"],
                scale_numeric=spec["scale_numeric"],
            )
            cv_rmse_mean, cv_rmse_std = evaluate_fixed_pipeline_cv(
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                cv=effective_cv,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                pipeline.fit(X_train, y_train)
        except Exception as exc:
            summary_rows.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "best_cv_rmse": np.nan,
                    "train_r2": np.nan,
                    "train_rmse": np.nan,
                    "train_mae": np.nan,
                    "valid_r2": np.nan,
                    "valid_rmse": np.nan,
                    "valid_mae": np.nan,
                    "best_params": "",
                    "overfit_gap_r2": np.nan,
                    "error": str(exc),
                }
            )
            print(f"{model_name} failed: {exc}")
            continue

        pred_train = pipeline.predict(X_train)
        pred_valid = pipeline.predict(X_valid)

        train_metrics, valid_metrics = save_fixed_model_outputs(
            output_dir=model_dir,
            fitted_pipeline=pipeline,
            params=spec["params"],
            cv_rmse_mean=cv_rmse_mean,
            cv_rmse_std=cv_rmse_std,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
            pred_train=pred_train,
            pred_valid=pred_valid,
            id_train=id_train,
            id_valid=id_valid,
            random_state=random_state,
        )

        summary_rows.append(
            {
                "model": model_name,
                "status": "ok",
                "best_cv_rmse": cv_rmse_mean,
                "train_r2": train_metrics["r2"],
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "valid_r2": valid_metrics["r2"],
                "valid_rmse": valid_metrics["rmse"],
                "valid_mae": valid_metrics["mae"],
                "best_params": json.dumps(to_builtin(spec["params"]), ensure_ascii=False),
                "overfit_gap_r2": train_metrics["r2"] - valid_metrics["r2"],
                "error": "",
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["status", "valid_rmse"], ascending=[True, True], na_position="last"
    )
    summary_df.to_csv(output_root / "summary_leaderboard.csv", index=False, encoding="utf-8-sig")
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train maize yield models with fixed single-parameter settings."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Optional target column. If omitted, the script tries Yield, yield_ha, and yield_per_mu.",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=[],
        help="Optional feature columns to exclude before model training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df = run_all_models_fixed(
        data_path=args.data_path,
        output_root=args.output_root,
        test_size=args.test_size,
        random_state=args.random_state,
        cv=args.cv,
        n_jobs=args.n_jobs,
        target_column=args.target_column,
        drop_columns=args.drop_columns,
    )
    print("\nTraining finished. Leaderboard:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
