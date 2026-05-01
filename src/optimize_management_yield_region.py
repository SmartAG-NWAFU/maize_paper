import argparse
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_DATA_PATH = PROJECT_ROOT / "data" / "model_vars.csv"
DEFAULT_COST_DATA_PATH = PROJECT_ROOT / "data" / "cost_profit_output.csv"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "outputs" / "model_outputs_fixed"
DEFAULT_FEATURES_PATH = DEFAULT_MODEL_ROOT / "used_features.csv"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "outputs" / "management_yield_region_optimized"
DEFAULT_COST_ESTIMATORS_CACHE_PATH = PROJECT_ROOT / "outputs" / "cost_profit_estimators_cache_ha.joblib"
DEFAULT_COST_ESTIMATORS_MERGED_PATH = PROJECT_ROOT / "outputs" / "cost_profit_estimators_merged_ha.csv"
DEFAULT_GRAIN_PRICE = 2.3
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAXITER = 100
DEFAULT_POPSIZE = 9
DEFAULT_LOWER_QUANTILE = 0.1
DEFAULT_UPPER_QUANTILE = 0.9
DEFAULT_DISTANCE_PENALTY_WEIGHT = 3000
DECISION_VALUE_STEPS = {
    "Sow_DOY": 1.0,
    "Density": 500.0,
    "Fer_N": 5.0,
    "Fer_P": 5.0,
    "Fer_K": 5.0,
    "Pest_Cost": 50.0,
    "Irr_Elec": 50.0,
}

MODEL_COLUMN_ALIASES = {
    "region_code": "Region",
    "sowing_date_doy": "Sow_DOY",
    "plant_density": "Density",
    "nitrogen_rate": "Fer_N",
    "phosphorus_rate": "Fer_P",
    "potassium_rate": "Fer_K",
    "fertilization_events": "Fer_Count",
    "pesticide_application_events": "Pest_Count",
    "pesticide_cost": "Pest_Cost",
    "irrigation_events": "Irr_Count",
    "irrigation_electricity": "Irr_Elec",
    "post_sowing_precipitation_total": "Precip",
    "post_sowing_radiation_total": "Rad",
    "post_sowing_relative_humidity_mean": "RH",
    "yield_per_mu": "Yield",
    "irrigation_mode_sprinkler": "Irr_Sprinkler",
    "irrigation_mode_none": "Irr_None",
    "irrigation_mode_drip": "Irr_Drip",
    "irrigation_mode_flood": "Irr_Flood",
}

FINAL_RESULT_COLUMNS = [
    "region_code",
    "region_name",
    "yield_value",
    "yield_mean",
    "yield_q25",
    "yield_q75",
    "profit_value",
    "input_cost",
    "objective_value",
    "distance_penalty",
    "Density",
    "Fer_N",
    "Fer_P",
    "Fer_K",
    "Fer_Count",
    "Pest_Count",
    "Sow_DOY",
    "Irr_Count",
    "Irr_Elec",
    "Pest_Cost",
    "result_type",
    "irrigation_mode",
    "irrigation_label",
]

IRRIGATION_MODE_SPECS = [
    ("NoIrrigation", "无灌溉", "Irr_None"),
    ("Flood", "漫灌", "Irr_Flood"),
    ("Sprinkler", "喷灌", "Irr_Sprinkler"),
    ("Drip", "滴灌", "Irr_Drip"),
]

BASELINE_RESULT_TYPE = "Baseline median"
OPTIMAL_YIELD_RESULT_TYPE = "Optimal yield"
NO_OBJECTIVE_IMPROVEMENT_RESULT_TYPE = "No objective improvement"
MODEL_NAME_PREFERENCE = ["xgboost", "random_forest", "linear_regression", "elastic_net"]

REGION_CODE_MAP = {
    0: "Hebei",
    1: "Shandong",
}


def configure_threads() -> None:
    total_cores = os.cpu_count() or 1
    half_cores = max(1, total_cores // 2)
    os.environ.setdefault("OMP_NUM_THREADS", str(half_cores))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(half_cores))
    os.environ.setdefault("MKL_NUM_THREADS", str(half_cores))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(half_cores))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(half_cores))
    os.environ.setdefault("BLIS_NUM_THREADS", str(half_cores))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize region-specific maize management for maximum yield. "
            "Management counts are inferred from related management variables."
        )
    )
    parser.add_argument("--model-data-path", type=Path, default=DEFAULT_MODEL_DATA_PATH)
    parser.add_argument("--cost-data-path", type=Path, default=DEFAULT_COST_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--cost-estimators-cache-path", type=Path, default=DEFAULT_COST_ESTIMATORS_CACHE_PATH)
    parser.add_argument("--cost-estimators-merged-path", type=Path, default=DEFAULT_COST_ESTIMATORS_MERGED_PATH)
    parser.add_argument("--refresh-cost-estimators-cache", action="store_true")
    parser.add_argument("--grain-price", type=float, default=DEFAULT_GRAIN_PRICE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--popsize", type=int, default=DEFAULT_POPSIZE)
    parser.add_argument("--aggregation", choices=["median", "mean"], default="median")
    parser.add_argument("--lower-quantile", type=float, default=DEFAULT_LOWER_QUANTILE)
    parser.add_argument("--upper-quantile", type=float, default=DEFAULT_UPPER_QUANTILE)
    parser.add_argument("--distance-penalty-weight", type=float, default=DEFAULT_DISTANCE_PENALTY_WEIGHT)
    parser.add_argument("--allow-unseen-irrigation-modes", action="store_true")
    return parser.parse_args()


def linear_formula_text(model: LinearRegression, variables: list[str], precision: int = 6) -> str:
    terms = [f"{float(model.intercept_):.{precision}f}"]
    for coef, variable in zip(model.coef_, variables):
        coef_value = float(coef)
        sign = "+" if coef_value >= 0 else "-"
        terms.append(f" {sign} {abs(coef_value):.{precision}f}*{variable}")
    return "".join(terms)


def get_irrigation_scenarios(available_columns: list[str]) -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    irrigation_dummy_cols = [spec[2] for spec in IRRIGATION_MODE_SPECS if spec[2] in available_columns]
    for mode_key, mode_label, active_col in IRRIGATION_MODE_SPECS:
        if active_col not in available_columns:
            continue
        mode_mapping = {col: int(col == active_col) for col in irrigation_dummy_cols}
        scenarios.append(
            {
                "mode_key": mode_key,
                "mode_label": mode_label,
                "mode_mapping": mode_mapping,
            }
        )
    if not scenarios:
        raise ValueError("No irrigation scenario columns were found in the model features.")
    return scenarios


def infer_irrigation_mode(row: pd.Series, irrigation_scenarios: list[dict[str, object]]) -> str:
    for scenario in irrigation_scenarios:
        if all(int(row[col]) == value for col, value in scenario["mode_mapping"].items()):
            return str(scenario["mode_key"])
    return "Unknown"


def load_model_with_fallback(model_path: Path | None):
    if model_path is not None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        return joblib.load(model_path), model_path

    errors: list[str] = []
    for candidate_name in MODEL_NAME_PREFERENCE:
        candidate_path = DEFAULT_MODEL_ROOT / candidate_name / "model.pkl"
        if not candidate_path.exists():
            continue
        try:
            return joblib.load(candidate_path), candidate_path
        except Exception as exc:
            errors.append(f"{candidate_path}: {exc}")

    error_text = "\n".join(errors) if errors else "No candidate models were found."
    raise RuntimeError(f"Unable to load any preferred model.\n{error_text}")


def load_inputs(
    model_data_path: Path,
    cost_data_path: Path,
    model_path: Path | None,
    features_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, object, list[str], Path]:
    model_df = pd.read_csv(model_data_path).rename(columns=MODEL_COLUMN_ALIASES)
    cost_df = pd.read_csv(cost_data_path, encoding="utf-8-sig")
    features = pd.read_csv(features_path)["feature"].tolist()
    for feature in features:
        if feature not in model_df.columns:
            model_df[feature] = np.nan
    model, resolved_model_path = load_model_with_fallback(model_path)
    return model_df, cost_df, model, features, resolved_model_path


def get_cost_columns(cost_df: pd.DataFrame) -> dict[str, str]:
    required_columns = {
        "sowing_cost": "sowing_cost_yuan_per_ha",
        "irrig_device_cost": "irrigation_device_cost_yuan_per_ha",
        "irrig_cost": "irrigation_running_cost_yuan_per_ha",
        "fert_cost": "fertilization_cost_yuan_per_ha",
        "total_cost": "total_cost_yuan_per_ha",
    }
    missing = [column_name for column_name in required_columns.values() if column_name not in cost_df.columns]
    if missing:
        raise KeyError(f"Missing required cost columns: {missing}")
    return required_columns


def fit_linear_cost_model(df: pd.DataFrame, x_cols: str | list[str], y_col: str) -> LinearRegression:
    model = LinearRegression()
    if isinstance(x_cols, str):
        x_cols = [x_cols]
    train_df = df[[*x_cols, y_col]].dropna()
    model.fit(train_df[x_cols], train_df[y_col])
    return model


def build_cost_estimator_inputs(
    model_df: pd.DataFrame,
    cost_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    cost_cols = get_cost_columns(cost_df)
    merged_df = model_df.merge(cost_df[["uuid", *cost_cols.values()]], on="uuid", how="inner").copy()
    return merged_df, cost_cols


def build_cost_estimators_from_merged(
    merged_df: pd.DataFrame,
    cost_cols: dict[str, str],
    irrigation_scenarios: list[dict[str, object]],
) -> dict[str, object]:
    merged_df = merged_df.copy()
    merged_df["observed_mode"] = merged_df.apply(
        infer_irrigation_mode,
        axis=1,
        irrigation_scenarios=irrigation_scenarios,
    )

    sowing_model = fit_linear_cost_model(merged_df, "Density", cost_cols["sowing_cost"])
    fert_model = fit_linear_cost_model(merged_df, ["Fer_N", "Fer_P", "Fer_K"], cost_cols["fert_cost"])
    irrigation_model = fit_linear_cost_model(merged_df, "Irr_Elec", cost_cols["irrig_cost"])

    device_cost_by_mode = (
        merged_df.groupby("observed_mode")[cost_cols["irrig_device_cost"]].median().to_dict()
    )
    for scenario in irrigation_scenarios:
        mode_key = str(scenario["mode_key"])
        value = device_cost_by_mode.get(mode_key, 0.0)
        device_cost_by_mode[mode_key] = 0.0 if pd.isna(value) else float(value)

    observed_sowing = sowing_model.predict(merged_df[["Density"]])
    observed_fert = fert_model.predict(merged_df[["Fer_N", "Fer_P", "Fer_K"]])
    observed_irrigation = irrigation_model.predict(merged_df[["Irr_Elec"]])
    observed_pesticide = merged_df["Pest_Cost"].to_numpy(dtype=float)
    observed_device = merged_df["observed_mode"].map(device_cost_by_mode).to_numpy(dtype=float)

    merged_df["fixed_cost"] = (
        merged_df[cost_cols["total_cost"]]
        - observed_sowing
        - observed_fert
        - observed_irrigation
        - observed_pesticide
        - observed_device
    )

    return {
        "fert_driver": ["Fer_N", "Fer_P", "Fer_K"],
        "sowing_model": sowing_model,
        "fert_model": fert_model,
        "irrigation_model": irrigation_model,
        "device_cost_by_mode": device_cost_by_mode,
        "fixed_cost": merged_df.set_index("uuid")["fixed_cost"],
    }


def load_or_build_cost_estimators(
    model_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    irrigation_scenarios: list[dict[str, object]],
    cache_path: Path,
    merged_path: Path,
    refresh_cache: bool = False,
) -> dict[str, object]:
    if cache_path.exists() and not refresh_cache:
        return joblib.load(cache_path)

    merged_df, cost_cols = build_cost_estimator_inputs(model_df, cost_df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(merged_path, index=False, encoding="utf-8-sig")

    estimators = build_cost_estimators_from_merged(merged_df, cost_cols, irrigation_scenarios)
    joblib.dump(estimators, cache_path)
    return estimators


def estimate_input_cost(
    density: float,
    fer_n: float,
    fer_p: float,
    fer_k: float,
    irr_elec: float,
    pest_cost: float,
    scenario_mode: str,
    estimators: dict[str, object],
    fixed_cost: np.ndarray,
) -> np.ndarray:
    sowing_model = estimators["sowing_model"]
    fert_model = estimators["fert_model"]
    irrigation_model = estimators["irrigation_model"]
    device_cost = float(estimators["device_cost_by_mode"].get(scenario_mode, 0.0))

    sowing_cost = float(sowing_model.intercept_ + sowing_model.coef_[0] * density)
    fert_cost = float(
        fert_model.intercept_
        + fert_model.coef_[0] * fer_n
        + fert_model.coef_[1] * fer_p
        + fert_model.coef_[2] * fer_k
    )
    irrigation_cost = float(irrigation_model.intercept_ + irrigation_model.coef_[0] * irr_elec)
    return fixed_cost + sowing_cost + fert_cost + pest_cost + irrigation_cost + device_cost


def region_label(region_code: object) -> str:
    if pd.isna(region_code):
        return "All"
    if isinstance(region_code, (int, np.integer)):
        return REGION_CODE_MAP.get(int(region_code), f"Region {int(region_code)}")
    if isinstance(region_code, float) and float(region_code).is_integer():
        return REGION_CODE_MAP.get(int(region_code), f"Region {int(region_code)}")
    return f"Region {region_code}"


def quantile_bounds(
    series: pd.Series,
    lower_quantile: float,
    upper_quantile: float,
    positive_only: bool = False,
) -> tuple[float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if positive_only:
        clean = clean.loc[clean > 0]
    if clean.empty:
        raise ValueError("Cannot build bounds from an empty series.")

    lower_value = float(clean.quantile(lower_quantile))
    upper_value = float(clean.quantile(upper_quantile))

    if lower_value > upper_value:
        lower_value, upper_value = upper_value, lower_value
    if np.isclose(lower_value, upper_value):
        lower_value = float(clean.min())
        upper_value = float(clean.max())
    if np.isclose(lower_value, upper_value):
        delta = max(abs(lower_value) * 0.05, 1.0)
        lower_value -= delta
        upper_value += delta
    return lower_value, upper_value


def build_decision_space(
    region_df: pd.DataFrame,
    irrigation_mode: str,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[list[str], list[tuple[float, float]]]:
    decision_columns = ["Sow_DOY", "Density", "Fer_N", "Fer_P", "Fer_K", "Pest_Cost"]
    bounds = [
        quantile_bounds(region_df["Sow_DOY"], lower_quantile, upper_quantile),
        quantile_bounds(region_df["Density"], lower_quantile, upper_quantile),
        quantile_bounds(region_df["Fer_N"], lower_quantile, upper_quantile),
        quantile_bounds(region_df["Fer_P"], lower_quantile, upper_quantile),
        quantile_bounds(region_df["Fer_K"], lower_quantile, upper_quantile),
        quantile_bounds(region_df["Pest_Cost"], lower_quantile, upper_quantile),
    ]

    if irrigation_mode != "NoIrrigation":
        reference_df = region_df.loc[
            (region_df["observed_irrigation_mode"] == irrigation_mode) & (region_df["Irr_Elec"] > 0)
        ]
        if reference_df.empty:
            reference_df = region_df.loc[region_df["Irr_Elec"] > 0]
        decision_columns.append("Irr_Elec")
        bounds.append(
            quantile_bounds(
                reference_df["Irr_Elec"],
                lower_quantile,
                upper_quantile,
                positive_only=True,
            )
        )

    return decision_columns, bounds


def decode_decision_vector(
    decision_columns: list[str],
    x: np.ndarray,
    irrigation_mode: str,
    bounds: list[tuple[float, float]],
) -> dict[str, float]:
    direct_values = {
        column: quantize_decision_value(column, float(value), bound)
        for column, value, bound in zip(decision_columns, x, bounds)
    }
    if irrigation_mode == "NoIrrigation":
        direct_values["Irr_Elec"] = 0.0
    return direct_values


def quantize_decision_value(
    column: str,
    value: float,
    bound: tuple[float, float],
) -> float:
    lower_bound, upper_bound = bound
    clipped_value = float(np.clip(value, lower_bound, upper_bound))
    step = DECISION_VALUE_STEPS.get(column, 1.0)
    if step <= 0:
        raise ValueError(f"Step size must be positive for {column}.")

    feasible_lower = np.ceil(lower_bound / step) * step
    feasible_upper = np.floor(upper_bound / step) * step
    if feasible_lower <= feasible_upper:
        quantized_value = round(clipped_value / step) * step
        quantized_value = float(np.clip(quantized_value, feasible_lower, feasible_upper))
    else:
        quantized_value = clipped_value

    if np.isclose(step, round(step)):
        return float(int(round(quantized_value)))
    return float(quantized_value)


def build_monotonic_count_model(
    driver_values: pd.Series,
    count_values: pd.Series,
) -> dict[str, object] | None:
    work_df = pd.DataFrame(
        {
            "driver": pd.to_numeric(driver_values, errors="coerce"),
            "count": pd.to_numeric(count_values, errors="coerce"),
        }
    ).dropna().sort_values("driver")
    if work_df.empty:
        return None

    x = work_df["driver"].to_numpy(dtype=float)
    y = work_df["count"].to_numpy(dtype=float)
    if len(np.unique(y.astype(int))) <= 1:
        return {
            "kind": "constant",
            "value": int(round(float(y[0]))),
        }

    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(x, y)
    return {
        "kind": "isotonic",
        "model": model,
        "min_count": int(np.min(y)),
        "max_count": int(np.max(y)),
    }


def predict_monotonic_count(
    count_model: dict[str, object] | None,
    driver_value: float,
) -> int:
    if count_model is None:
        raise ValueError("Count model is not available.")

    if count_model["kind"] == "constant":
        return int(count_model["value"])

    model: IsotonicRegression = count_model["model"]
    predicted_continuous = float(model.predict([float(driver_value)])[0])
    predicted_count = int(np.ceil(predicted_continuous - 1e-9))
    return int(
        np.clip(
            predicted_count,
            int(count_model["min_count"]),
            int(count_model["max_count"]),
        )
    )


def build_region_count_models(region_df: pd.DataFrame) -> dict[str, object]:
    fer_total = (
        pd.to_numeric(region_df["Fer_N"], errors="coerce")
        + pd.to_numeric(region_df["Fer_P"], errors="coerce")
        + pd.to_numeric(region_df["Fer_K"], errors="coerce")
    )
    fer_count_model = build_monotonic_count_model(fer_total, region_df["Fer_Count"])
    pest_count_model = build_monotonic_count_model(region_df["Pest_Cost"], region_df["Pest_Count"])

    irrigation_count_models: dict[str, dict[str, object] | None] = {}
    for irrigation_mode in sorted(region_df["observed_irrigation_mode"].dropna().unique().tolist()):
        if irrigation_mode == "NoIrrigation":
            continue
        mode_df = region_df.loc[
            (region_df["observed_irrigation_mode"] == irrigation_mode) & (region_df["Irr_Elec"] > 0)
        ]
        irrigation_count_models[irrigation_mode] = build_monotonic_count_model(
            mode_df["Irr_Elec"],
            mode_df["Irr_Count"],
        )

    irrigation_count_models["__fallback__"] = build_monotonic_count_model(
        region_df.loc[region_df["Irr_Elec"] > 0, "Irr_Elec"],
        region_df.loc[region_df["Irr_Elec"] > 0, "Irr_Count"],
    )
    return {
        "fer_count_model": fer_count_model,
        "pest_count_model": pest_count_model,
        "irrigation_count_models": irrigation_count_models,
    }


def infer_management_counts(
    irrigation_mode: str,
    direct_values: dict[str, float],
    fer_count_model: dict[str, object] | None,
    pest_count_model: dict[str, object] | None,
    irrigation_count_models: dict[str, dict[str, object] | None],
) -> dict[str, int]:
    fer_total = direct_values["Fer_N"] + direct_values["Fer_P"] + direct_values["Fer_K"]
    if fer_count_model is None:
        raise ValueError("Fer count model is not available.")
    fer_count = predict_monotonic_count(fer_count_model, fer_total)

    if pest_count_model is None:
        raise ValueError("Pest count model is not available.")
    pest_count = predict_monotonic_count(pest_count_model, direct_values["Pest_Cost"])

    if irrigation_mode == "NoIrrigation":
        irr_count = 0
    else:
        irrigation_count_model = irrigation_count_models.get(irrigation_mode)
        if irrigation_count_model is None:
            irrigation_count_model = irrigation_count_models.get("__fallback__")
        if irrigation_count_model is None:
            raise ValueError(f"Irrigation count model is not available for mode: {irrigation_mode}")
        irr_count = predict_monotonic_count(irrigation_count_model, direct_values["Irr_Elec"])
        irr_count = max(1, irr_count)

    return {
        "Fer_Count": int(fer_count),
        "Pest_Count": int(pest_count),
        "Irr_Count": int(irr_count),
    }


def nearest_management_distance(
    region_df: pd.DataFrame,
    irrigation_mode: str,
    direct_values: dict[str, float],
) -> float:
    columns = ["Sow_DOY", "Density", "Fer_N", "Fer_P", "Fer_K", "Pest_Cost"]
    if irrigation_mode != "NoIrrigation":
        columns.append("Irr_Elec")

    reference_df = region_df.copy()
    if irrigation_mode != "NoIrrigation":
        reference_df = reference_df.loc[reference_df["observed_irrigation_mode"] == irrigation_mode]
        if reference_df.empty:
            reference_df = region_df.copy()
    else:
        none_df = reference_df.loc[reference_df["observed_irrigation_mode"] == "NoIrrigation"]
        if not none_df.empty:
            reference_df = none_df

    reference_df = reference_df[columns].dropna()
    if reference_df.empty:
        return 0.0

    reference_matrix = reference_df.to_numpy(dtype=float)
    candidate_array = np.asarray([direct_values[column] for column in columns], dtype=float)
    scales = reference_matrix.std(axis=0, ddof=0)
    scales[~np.isfinite(scales) | (scales < 1e-9)] = 1.0
    distances = np.linalg.norm((reference_matrix - candidate_array) / scales, axis=1)
    return float(np.min(distances))


def aggregate_yield(values: np.ndarray, aggregation: str) -> float:
    if aggregation == "mean":
        return float(np.mean(values))
    return float(np.median(values))


def evaluate_region_management(
    region_df: pd.DataFrame,
    model,
    features: list[str],
    estimators: dict[str, object],
    fer_count_model: dict[str, object] | None,
    pest_count_model: dict[str, object] | None,
    irrigation_count_models: dict[str, dict[str, object] | None],
    fixed_cost: np.ndarray,
    irrigation_scenario: dict[str, object],
    direct_values: dict[str, float],
    grain_price: float,
    aggregation: str,
    distance_penalty_weight: float,
) -> dict[str, float | int | str]:
    counts = infer_management_counts(
        irrigation_mode=str(irrigation_scenario["mode_key"]),
        direct_values=direct_values,
        fer_count_model=fer_count_model,
        pest_count_model=pest_count_model,
        irrigation_count_models=irrigation_count_models,
    )
    x_predict_df = region_df[features].copy()

    for column, value in direct_values.items():
        if column in x_predict_df.columns:
            x_predict_df[column] = float(value)
    for column, value in counts.items():
        if column in x_predict_df.columns:
            x_predict_df[column] = int(value)
    for column, value in irrigation_scenario["mode_mapping"].items():
        if column in x_predict_df.columns:
            x_predict_df[column] = int(value)

    predicted_yield = np.asarray(model.predict(x_predict_df), dtype=float)
    estimated_cost = estimate_input_cost(
        density=float(direct_values["Density"]),
        fer_n=float(direct_values["Fer_N"]),
        fer_p=float(direct_values["Fer_P"]),
        fer_k=float(direct_values["Fer_K"]),
        irr_elec=float(direct_values["Irr_Elec"]),
        pest_cost=float(direct_values["Pest_Cost"]),
        scenario_mode=str(irrigation_scenario["mode_key"]),
        estimators=estimators,
        fixed_cost=fixed_cost,
    )
    predicted_profit = predicted_yield * grain_price - estimated_cost

    raw_objective = aggregate_yield(predicted_yield, aggregation)
    distance_penalty = nearest_management_distance(
        region_df=region_df,
        irrigation_mode=str(irrigation_scenario["mode_key"]),
        direct_values=direct_values,
    )
    objective_value = raw_objective - distance_penalty_weight * distance_penalty

    return {
        "yield_median": float(np.median(predicted_yield)),
        "yield_mean": float(np.mean(predicted_yield)),
        "yield_q25": float(np.quantile(predicted_yield, 0.25)),
        "yield_q75": float(np.quantile(predicted_yield, 0.75)),
        "profit_median": float(np.median(predicted_profit)),
        "input_cost_median": float(np.median(estimated_cost)),
        "raw_objective": raw_objective,
        "distance_penalty": float(distance_penalty),
        "objective_value": float(objective_value),
        "Density": float(direct_values["Density"]),
        "Fer_N": float(direct_values["Fer_N"]),
        "Fer_P": float(direct_values["Fer_P"]),
        "Fer_K": float(direct_values["Fer_K"]),
        "Fer_Count": int(counts["Fer_Count"]),
        "Pest_Count": int(counts["Pest_Count"]),
        "Sow_DOY": float(direct_values["Sow_DOY"]),
        "Irr_Count": int(counts["Irr_Count"]),
        "Irr_Elec": float(direct_values["Irr_Elec"]),
        "Pest_Cost": float(direct_values["Pest_Cost"]),
        "irrigation_mode": str(irrigation_scenario["mode_key"]),
        "irrigation_label": str(irrigation_scenario["mode_label"]),
    }


def build_baseline_summary(
    model_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    grain_price: float,
    aggregation: str,
) -> dict[str, float]:
    cost_cols = get_cost_columns(cost_df)
    baseline_df = (
        model_df[
            [
                "uuid",
                "Yield",
                "Density",
                "Fer_N",
                "Fer_P",
                "Fer_K",
                "Fer_Count",
                "Pest_Count",
                "Sow_DOY",
                "Irr_Count",
                "Irr_Elec",
                "Pest_Cost",
            ]
        ]
        .merge(cost_df[["uuid", cost_cols["total_cost"]]], on="uuid", how="inner")
        .copy()
    )
    baseline_df["profit"] = baseline_df["Yield"] * grain_price - baseline_df[cost_cols["total_cost"]]
    return {
        "yield_median": float(baseline_df["Yield"].median()),
        "yield_mean": float(baseline_df["Yield"].mean()),
        "yield_q25": float(baseline_df["Yield"].quantile(0.25)),
        "yield_q75": float(baseline_df["Yield"].quantile(0.75)),
        "profit_median": float(baseline_df["profit"].median()),
        "input_cost_median": float(baseline_df[cost_cols["total_cost"]].median()),
        "objective_value": aggregate_yield(baseline_df["Yield"].to_numpy(dtype=float), aggregation),
        "distance_penalty": 0.0,
        "Density": float(baseline_df["Density"].median()),
        "Fer_N": float(baseline_df["Fer_N"].median()),
        "Fer_P": float(baseline_df["Fer_P"].median()),
        "Fer_K": float(baseline_df["Fer_K"].median()),
        "Fer_Count": float(baseline_df["Fer_Count"].median()),
        "Pest_Count": float(baseline_df["Pest_Count"].median()),
        "Sow_DOY": float(baseline_df["Sow_DOY"].median()),
        "Irr_Count": float(baseline_df["Irr_Count"].median()),
        "Irr_Elec": float(baseline_df["Irr_Elec"].median()),
        "Pest_Cost": float(baseline_df["Pest_Cost"].median()),
    }


def optimize_region(
    region_df: pd.DataFrame,
    region_cost_df: pd.DataFrame,
    model,
    features: list[str],
    estimators: dict[str, object],
    irrigation_scenarios: list[dict[str, object]],
    grain_price: float,
    aggregation: str,
    random_state: int,
    maxiter: int,
    popsize: int,
    lower_quantile: float,
    upper_quantile: float,
    distance_penalty_weight: float,
    allow_unseen_irrigation_modes: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    region_code = region_df["Region"].iloc[0]
    region_name = region_label(region_code)
    count_models = build_region_count_models(region_df)
    fer_count_model = count_models["fer_count_model"]
    pest_count_model = count_models["pest_count_model"]
    irrigation_count_models = count_models["irrigation_count_models"]

    fixed_cost_series = estimators["fixed_cost"].reindex(pd.Index(region_df["uuid"].values))
    fixed_cost_series = fixed_cost_series.fillna(float(estimators["fixed_cost"].median()))
    fixed_cost = fixed_cost_series.to_numpy(dtype=float)

    observed_modes = set(region_df["observed_irrigation_mode"].dropna().tolist())
    candidate_scenarios = irrigation_scenarios
    if not allow_unseen_irrigation_modes:
        candidate_scenarios = [
            scenario for scenario in irrigation_scenarios if str(scenario["mode_key"]) in observed_modes
        ]
    if not candidate_scenarios:
        raise ValueError(f"No irrigation scenarios are available for region {region_code}.")

    mode_rows: list[dict[str, object]] = []
    for scenario_index, irrigation_scenario in enumerate(candidate_scenarios):
        irrigation_mode = str(irrigation_scenario["mode_key"])
        decision_columns, bounds = build_decision_space(
            region_df=region_df,
            irrigation_mode=irrigation_mode,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

        def objective(x: np.ndarray) -> float:
            direct_values = decode_decision_vector(
                decision_columns=decision_columns,
                x=np.asarray(x, dtype=float),
                irrigation_mode=irrigation_mode,
                bounds=bounds,
            )
            evaluation = evaluate_region_management(
                region_df=region_df,
                model=model,
                features=features,
                estimators=estimators,
                fer_count_model=fer_count_model,
                pest_count_model=pest_count_model,
                irrigation_count_models=irrigation_count_models,
                fixed_cost=fixed_cost,
                irrigation_scenario=irrigation_scenario,
                direct_values=direct_values,
                grain_price=grain_price,
                aggregation=aggregation,
                distance_penalty_weight=distance_penalty_weight,
            )
            return -float(evaluation["objective_value"])

        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=random_state + int(region_code) * 100 + scenario_index,
            maxiter=maxiter,
            popsize=popsize,
            polish=True,
            updating="immediate",
        )

        best_direct_values = decode_decision_vector(
            decision_columns=decision_columns,
            x=np.asarray(result.x, dtype=float),
            irrigation_mode=irrigation_mode,
            bounds=bounds,
        )
        best_evaluation = evaluate_region_management(
            region_df=region_df,
            model=model,
            features=features,
            estimators=estimators,
            fer_count_model=fer_count_model,
            pest_count_model=pest_count_model,
            irrigation_count_models=irrigation_count_models,
            fixed_cost=fixed_cost,
            irrigation_scenario=irrigation_scenario,
            direct_values=best_direct_values,
            grain_price=grain_price,
            aggregation=aggregation,
            distance_penalty_weight=distance_penalty_weight,
        )
        best_evaluation.update(
            {
                "region_code": int(region_code),
                "region_name": region_name,
                "optimization_status": "ok" if bool(result.success) else "warn",
                "optimizer_message": str(result.message),
                "optimizer_nit": int(result.nit),
                "optimizer_nfev": int(result.nfev),
            }
        )
        mode_rows.append(best_evaluation)

    mode_results_df = pd.DataFrame(mode_rows).sort_values(
        ["objective_value", "yield_median"],
        ascending=[False, False],
    ).reset_index(drop=True)

    baseline_summary = build_baseline_summary(region_df, region_cost_df, grain_price, aggregation)
    best_row = mode_results_df.iloc[0]
    use_best_candidate = float(best_row["objective_value"]) > float(baseline_summary["objective_value"])

    baseline_row = {
        "region_code": int(region_code),
        "region_name": region_name,
        "yield_value": baseline_summary["yield_median"],
        "yield_mean": baseline_summary["yield_mean"],
        "yield_q25": baseline_summary["yield_q25"],
        "yield_q75": baseline_summary["yield_q75"],
        "profit_value": baseline_summary["profit_median"],
        "input_cost": baseline_summary["input_cost_median"],
        "objective_value": baseline_summary["objective_value"],
        "distance_penalty": baseline_summary["distance_penalty"],
        "Density": baseline_summary["Density"],
        "Fer_N": baseline_summary["Fer_N"],
        "Fer_P": baseline_summary["Fer_P"],
        "Fer_K": baseline_summary["Fer_K"],
        "Fer_Count": baseline_summary["Fer_Count"],
        "Pest_Count": baseline_summary["Pest_Count"],
        "Sow_DOY": baseline_summary["Sow_DOY"],
        "Irr_Count": baseline_summary["Irr_Count"],
        "Irr_Elec": baseline_summary["Irr_Elec"],
        "Pest_Cost": baseline_summary["Pest_Cost"],
        "result_type": BASELINE_RESULT_TYPE,
        "irrigation_mode": np.nan,
        "irrigation_label": BASELINE_RESULT_TYPE,
    }
    selected_candidate_row = (
        {
            "region_code": int(region_code),
            "region_name": region_name,
            "yield_value": best_row["yield_median"],
            "yield_mean": best_row["yield_mean"],
            "yield_q25": best_row["yield_q25"],
            "yield_q75": best_row["yield_q75"],
            "profit_value": best_row["profit_median"],
            "input_cost": best_row["input_cost_median"],
            "objective_value": best_row["objective_value"],
            "distance_penalty": best_row["distance_penalty"],
            "Density": best_row["Density"],
            "Fer_N": best_row["Fer_N"],
            "Fer_P": best_row["Fer_P"],
            "Fer_K": best_row["Fer_K"],
            "Fer_Count": best_row["Fer_Count"],
            "Pest_Count": best_row["Pest_Count"],
            "Sow_DOY": best_row["Sow_DOY"],
            "Irr_Count": best_row["Irr_Count"],
            "Irr_Elec": best_row["Irr_Elec"],
            "Pest_Cost": best_row["Pest_Cost"],
            "result_type": OPTIMAL_YIELD_RESULT_TYPE,
            "irrigation_mode": best_row["irrigation_mode"],
            "irrigation_label": best_row["irrigation_label"],
        }
        if use_best_candidate
        else {
            **baseline_row,
            "result_type": NO_OBJECTIVE_IMPROVEMENT_RESULT_TYPE,
            "irrigation_label": NO_OBJECTIVE_IMPROVEMENT_RESULT_TYPE,
        }
    )

    selected_rows_df = pd.DataFrame(
        [
            baseline_row,
            selected_candidate_row,
        ]
    )
    numeric_cols = [column for column in FINAL_RESULT_COLUMNS if column not in {
        "region_code",
        "region_name",
        "result_type",
        "irrigation_mode",
        "irrigation_label",
    }]
    selected_rows_df[numeric_cols] = selected_rows_df[numeric_cols].round(2)
    return selected_rows_df[FINAL_RESULT_COLUMNS], mode_results_df


def build_region_results(
    model_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    model,
    features: list[str],
    estimators: dict[str, object],
    irrigation_scenarios: list[dict[str, object]],
    grain_price: float,
    aggregation: str,
    random_state: int,
    maxiter: int,
    popsize: int,
    lower_quantile: float,
    upper_quantile: float,
    distance_penalty_weight: float,
    allow_unseen_irrigation_modes: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_region_frames: list[pd.DataFrame] = []
    mode_frames: list[pd.DataFrame] = []
    for region_code in sorted(model_df["Region"].dropna().unique().tolist()):
        region_df = model_df.loc[model_df["Region"] == region_code].copy()
        region_cost_df = cost_df.loc[cost_df["uuid"].isin(region_df["uuid"])].copy()
        if region_df.empty:
            continue
        selected_rows_df, mode_results_df = optimize_region(
            region_df=region_df,
            region_cost_df=region_cost_df,
            model=model,
            features=features,
            estimators=estimators,
            irrigation_scenarios=irrigation_scenarios,
            grain_price=grain_price,
            aggregation=aggregation,
            random_state=random_state,
            maxiter=maxiter,
            popsize=popsize,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            distance_penalty_weight=distance_penalty_weight,
            allow_unseen_irrigation_modes=allow_unseen_irrigation_modes,
        )
        selected_region_frames.append(selected_rows_df)
        mode_frames.append(mode_results_df)

    selected_results_df = (
        pd.concat(selected_region_frames, ignore_index=True)
        if selected_region_frames
        else pd.DataFrame(columns=FINAL_RESULT_COLUMNS)
    )
    mode_results_df = pd.concat(mode_frames, ignore_index=True) if mode_frames else pd.DataFrame()
    return selected_results_df, mode_results_df


def export_region_results(region_results_df: pd.DataFrame, export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    ordered_df = region_results_df.copy()
    ordered_df["sort_order"] = ordered_df["result_type"].map(
        {
            BASELINE_RESULT_TYPE: 0,
            OPTIMAL_YIELD_RESULT_TYPE: 1,
            NO_OBJECTIVE_IMPROVEMENT_RESULT_TYPE: 1,
        }
    )
    ordered_df = ordered_df.sort_values(["region_code", "sort_order"]).drop(columns=["sort_order"])
    ordered_df.to_csv(
        export_dir / "management_region_baseline_vs_optimal.csv",
        index=False,
        encoding="utf-8-sig",
    )


def export_mode_results(mode_results_df: pd.DataFrame, export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    if mode_results_df.empty:
        return
    mode_results_df = mode_results_df.copy()
    numeric_columns = mode_results_df.select_dtypes(include=[np.number]).columns.tolist()
    mode_results_df[numeric_columns] = mode_results_df[numeric_columns].round(4)
    mode_results_df.to_csv(
        export_dir / "management_region_mode_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )


def export_run_metadata(
    export_dir: Path,
    model_path: Path,
    aggregation: str,
    random_state: int,
    maxiter: int,
    popsize: int,
    lower_quantile: float,
    upper_quantile: float,
    distance_penalty_weight: float,
) -> None:
    metadata_df = pd.DataFrame(
        [
            {
                "resolved_model_path": str(model_path),
                "aggregation": aggregation,
                "random_state": random_state,
                "maxiter": maxiter,
                "popsize": popsize,
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
                "distance_penalty_weight": distance_penalty_weight,
            }
        ]
    )
    metadata_df.to_csv(export_dir / "run_metadata.csv", index=False, encoding="utf-8-sig")


def export_cost_formula_results(estimators: dict[str, object], export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    sowing_model: LinearRegression = estimators["sowing_model"]
    fert_model: LinearRegression = estimators["fert_model"]
    irrigation_model: LinearRegression = estimators["irrigation_model"]
    device_cost_by_mode: dict[str, float] = estimators["device_cost_by_mode"]
    fixed_cost_series: pd.Series = estimators["fixed_cost"]

    formula_rows = [
        {
            "formula_name": "sowing_cost",
            "target_column": "sowing_cost",
            "variables": "Density",
            "formula_text": linear_formula_text(sowing_model, ["Density"]),
            "intercept": float(sowing_model.intercept_),
            "coef_Density": float(sowing_model.coef_[0]),
            "coef_Fer_N": np.nan,
            "coef_Fer_P": np.nan,
            "coef_Fer_K": np.nan,
            "coef_Irr_Elec": np.nan,
            "fixed_cost_median": np.nan,
        },
        {
            "formula_name": "fert_cost",
            "target_column": "fert_cost",
            "variables": "Fer_N,Fer_P,Fer_K",
            "formula_text": linear_formula_text(fert_model, ["Fer_N", "Fer_P", "Fer_K"]),
            "intercept": float(fert_model.intercept_),
            "coef_Density": np.nan,
            "coef_Fer_N": float(fert_model.coef_[0]),
            "coef_Fer_P": float(fert_model.coef_[1]),
            "coef_Fer_K": float(fert_model.coef_[2]),
            "coef_Irr_Elec": np.nan,
            "fixed_cost_median": np.nan,
        },
        {
            "formula_name": "irrigation_running_cost",
            "target_column": "irrigation_cost",
            "variables": "Irr_Elec",
            "formula_text": linear_formula_text(irrigation_model, ["Irr_Elec"]),
            "intercept": float(irrigation_model.intercept_),
            "coef_Density": np.nan,
            "coef_Fer_N": np.nan,
            "coef_Fer_P": np.nan,
            "coef_Fer_K": np.nan,
            "coef_Irr_Elec": float(irrigation_model.coef_[0]),
            "fixed_cost_median": np.nan,
        },
        {
            "formula_name": "total_input_cost",
            "target_column": "input_cost",
            "variables": "fixed_cost,Density,Fer_N,Fer_P,Fer_K,Pest_Cost,Irr_Elec,device_cost_by_mode",
            "formula_text": (
                "fixed_cost + sowing_cost + fert_cost + Pest_Cost + "
                "irrigation_running_cost + irrigation_device_cost"
            ),
            "intercept": np.nan,
            "coef_Density": float(sowing_model.coef_[0]),
            "coef_Fer_N": float(fert_model.coef_[0]),
            "coef_Fer_P": float(fert_model.coef_[1]),
            "coef_Fer_K": float(fert_model.coef_[2]),
            "coef_Irr_Elec": float(irrigation_model.coef_[0]),
            "fixed_cost_median": float(fixed_cost_series.median()),
        },
    ]
    formula_df = pd.DataFrame(formula_rows).round(6)
    formula_df.to_csv(export_dir / "cost_formula_summary.csv", index=False, encoding="utf-8-sig")

    device_cost_df = pd.DataFrame(
        [
            {"irrigation_mode": mode_key, "irrigation_device_cost": float(device_cost)}
            for mode_key, device_cost in device_cost_by_mode.items()
        ]
    ).sort_values("irrigation_mode")
    device_cost_df["irrigation_device_cost"] = device_cost_df["irrigation_device_cost"].round(6)
    device_cost_df.to_csv(export_dir / "irrigation_device_cost_by_mode.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    configure_threads()
    args = parse_args()

    if not 0.0 <= args.lower_quantile < args.upper_quantile <= 1.0:
        raise ValueError("Quantile bounds must satisfy 0 <= lower < upper <= 1.")

    model_df, cost_df, model, features, resolved_model_path = load_inputs(
        model_data_path=args.model_data_path,
        cost_data_path=args.cost_data_path,
        model_path=args.model_path,
        features_path=args.features_path,
    )
    irrigation_scenarios = get_irrigation_scenarios(features)
    model_df["observed_irrigation_mode"] = model_df.apply(
        infer_irrigation_mode,
        axis=1,
        irrigation_scenarios=irrigation_scenarios,
    )
    estimators = load_or_build_cost_estimators(
        model_df=model_df,
        cost_df=cost_df,
        irrigation_scenarios=irrigation_scenarios,
        cache_path=args.cost_estimators_cache_path,
        merged_path=args.cost_estimators_merged_path,
        refresh_cache=args.refresh_cost_estimators_cache,
    )

    selected_results_df, mode_results_df = build_region_results(
        model_df=model_df,
        cost_df=cost_df,
        model=model,
        features=features,
        estimators=estimators,
        irrigation_scenarios=irrigation_scenarios,
        grain_price=args.grain_price,
        aggregation=args.aggregation,
        random_state=args.random_state,
        maxiter=args.maxiter,
        popsize=args.popsize,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        distance_penalty_weight=args.distance_penalty_weight,
        allow_unseen_irrigation_modes=args.allow_unseen_irrigation_modes,
    )

    args.export_dir.mkdir(parents=True, exist_ok=True)
    export_region_results(selected_results_df, args.export_dir)
    export_mode_results(mode_results_df, args.export_dir)
    export_cost_formula_results(estimators, args.export_dir)
    export_run_metadata(
        export_dir=args.export_dir,
        model_path=resolved_model_path,
        aggregation=args.aggregation,
        random_state=args.random_state,
        maxiter=args.maxiter,
        popsize=args.popsize,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        distance_penalty_weight=args.distance_penalty_weight,
    )

    print(f"Resolved model: {resolved_model_path}")
    print(f"Selected region results saved to: {args.export_dir / 'management_region_baseline_vs_optimal.csv'}")
    print(f"Mode candidate results saved to: {args.export_dir / 'management_region_mode_candidates.csv'}")


if __name__ == "__main__":
    main()
