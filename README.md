# Maize Precision-Agriculture Paper Workspace

This repository is a paper-analysis workspace for a maize management optimization study in the North China Plain. The central question is not only which factors are associated with maize yield, but how much yield and profit could improve if current management practices were optimized within the range supported by observed field data.

The current analysis follows a yield-first, profit-second structure:

1. Build leakage-controlled maize yield prediction models.
2. Select the best-performing model for scenario analysis.
3. Define current regional management as the baseline.
4. Search for region-specific maximum-yield management scenarios within empirical management bounds.
5. Estimate input cost and profit under baseline and optimized scenarios.
6. Generate manuscript figures and results text.

## Repository Layout

```text
data/
  model_vars.csv              Main model dataset, 81 field observations.
  cost_required_ha.csv        Cost data in hectare-scale units.
  trial_locations.csv         Field locations for the study-area map.
  china_shp/                  China/province shapefiles for Fig. 1.

src/
  model_fixed.py              Fixed-parameter model training script.
  optimize_management_yield_region.py
                              Region-level yield optimization and profit evaluation.
  figs/                       Figure-generation scripts for Figs. 1-9.

outputs/
  model_outputs_fixed/        Saved model artifacts, metrics, predictions, and feature importance.
  management_yield_region_optimized/
                              Optimization outputs and cost-estimator summaries.

fig/
  fig1-fig9 image outputs and figure summary CSVs.

docs/
  Manuscript drafts, methods/results text, variable tables, and converted paper files.
```

## Main Data

`data/model_vars.csv` is the main modeling table. It contains 81 field-level observations with:

- identifiers: `uuid`
- region: `Region`, where `0 = Hebei` and `1 = Shandong`
- management variables: `Sow_DOY`, `Density`, `Fer_N`, `Fer_P`, `Fer_K`, `Fer_Count`, `Pest_Count`, `Pest_Cost`, `Irr_Count`, `Irr_Elec`
- irrigation mode dummies: `Irr_Sprinkler`, `Irr_None`, `Irr_Drip`, `Irr_Flood`
- field/weather variables: `Lodging`, `Precip`, `Rad`, `RH`, `Wind`
- target: `Yield`, in kg/ha

`data/cost_required_ha.csv` contains cost fields used by the optimization script:

- `sowing_cost_yuan_per_ha`
- `irrigation_device_cost_yuan_per_ha`
- `irrigation_running_cost_yuan_per_ha`
- `fertilization_cost_yuan_per_ha`
- `total_cost_yuan_per_ha`

`data/trial_locations.csv` and `data/china_shp/` support the study-area map.

## Modeling Workflow

The intended model-training entrypoint is:

```powershell
python src\model_fixed.py
```

The script trains four candidate models with fixed hyperparameters:

- linear regression
- Elastic Net
- random forest
- XGBoost

Outputs are written under `outputs/model_outputs_fixed/`, including:

- `summary_leaderboard.csv`
- per-model `model.pkl`
- `metrics_train_valid.csv`
- `pred_train.csv`
- `pred_valid.csv`
- `feature_importance.csv`
- `best_params.json`
- `cv_results.csv`

The saved leaderboard currently selects XGBoost as the best model:

| model | valid R2 | valid RMSE kg/ha | valid MAE kg/ha |
| --- | ---: | ---: | ---: |
| XGBoost | 0.652 | 1168.06 | 932.71 |
| Random forest | 0.608 | 1238.41 | 881.71 |
| Linear regression | 0.540 | 1341.92 | 1129.51 |
| Elastic Net | 0.319 | 1632.71 | 1234.18 |

Top XGBoost features in the saved output include `Irr_Drip`, `Density`, `Sow_DOY`, `Irr_Count`, `Pest_Count`, `Wind`, `RH`, `Irr_Elec`, `Rad`, and `Precip`.

## Optimization Workflow

The region-level optimization entrypoint is:

```powershell
python src\optimize_management_yield_region.py
```

This script:

1. Loads the saved model, preferring XGBoost.
2. Loads the feature list from `outputs/model_outputs_fixed/used_features.csv`.
3. Builds or loads cost estimators from `data/cost_required_ha.csv`.
4. Optimizes management separately by region.
5. Evaluates candidate irrigation modes: no irrigation, flood, sprinkler, and drip, where available.
6. Uses differential evolution to search management values within regional empirical quantile bounds.
7. Calculates baseline and optimized yield, input cost, and profit.

The directly optimized management variables are:

- `Sow_DOY`
- `Density`
- `Fer_N`
- `Fer_P`
- `Fer_K`
- `Pest_Cost`
- `Irr_Elec`, except under the no-irrigation scenario
- irrigation mode

Operational counts are inferred rather than independently optimized:

- `Fer_Count` from total nutrient input
- `Pest_Count` from pesticide cost
- `Irr_Count` from irrigation electricity input

The current optimization metadata is:

- model: `outputs/model_outputs_fixed/xgboost/model.pkl`
- aggregation: median
- random state: 42
- lower/upper management quantiles: 0.05 / 0.95
- distance penalty weight: 150.0

## Current Main Results

The saved optimization results are in:

```text
outputs/management_yield_region_optimized/management_region_baseline_vs_optimal.csv
fig/fig9_region_optimization_summary.csv
```

Current regional summary:

| region | baseline yield | optimized yield | yield gain | baseline profit | optimized profit | profit gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hebei | 9585.00 kg/ha | 12209.45 kg/ha | 27.38% | 10764.39 CNY/ha | 16163.22 CNY/ha | 50.15% |
| Shandong | 10721.85 kg/ha | 12603.64 kg/ha | 17.55% | 12031.56 CNY/ha | 17615.88 CNY/ha | 46.41% |

The current optimized scenarios select drip irrigation for both regions. The main adjustment pattern is earlier sowing, higher density, increased or rebalanced nutrient inputs, stronger irrigation support, and modestly changed pesticide cost.

## Figure Scripts

Figure-generation scripts are under `src/figs/`:

- `fig1_study_area.py`: study-area map; requires a Tianditu API key from `.env` or environment variable `tianditu_api_key`.
- `fig2_sowing.py`: sowing date and plant density.
- `fig3_fertilization.py`: fertilization frequency and N/P/K inputs.
- `fig4_irrigation.py`: irrigation frequency, electricity input, and irrigation mode.
- `fig5_pesticide.py`: pesticide frequency and cost.
- `fig6_yield_cost_profit.py`: baseline yield, cost, and profit.
- `fig7_model_performance.py`: model performance comparison.
- `fig8_feature_importance.py`: XGBoost feature importance.
- `fig9_region_optimization_results.py`: baseline vs optimized yield and profit.

Example:

```powershell
python src\figs\fig7_model_performance.py
python src\figs\fig9_region_optimization_results.py
```

## Manuscript Files

The main paper files are in `docs/`.

Notable files include:

- `docs/project_brief.md`
- `docs/model_variable_table.md`
- `docs/method_optimal_yield_20260416.md`
- `docs/results_fig2_fig9_precision_agriculture_en_20260416.md`
- `docs/maize_paper.docx`
- `docs/maize_paper.tex`
- `docs/maize_paper.pdf`
- `docs/references.bib`

`docs/maize_paper.tex` was converted from `docs/maize_paper.docx` with Pandoc. The extracted media are in `docs/maize_paper_media/`.

`docs/references.bib` is the BibTeX bibliography file for the maize paper.

## Known Reproducibility Issues

Several outputs are already present and internally consistent, but the repository is not yet fully rerunnable from source without cleanup.

Known issues:

- `src/model_fixed.py` imports helpers from `temp.model`, but no `temp/` module is present in this workspace. Running `python src\model_fixed.py --help` currently fails with `ModuleNotFoundError: No module named 'temp'`.
- `src/figs/fig6_yield_cost_profit.py` defaults to `data/cost.csv`, which is not present. The available cost file is `data/cost_required_ha.csv`, but the script currently expects a Chinese column named `总成本(元/亩)`.
- `docs/project_brief.md` references older script names such as `src/model.py`, `src/optimize_management_yield.py`, and `src/plot_management_responses.py`; the current workspace uses `src/model_fixed.py` and `src/optimize_management_yield_region.py`.
- There is no dependency manifest such as `requirements.txt`, `pyproject.toml`, or `environment.yml`.
- There is no `.git/` directory in this workspace, so this appears to be a snapshot rather than an active git checkout.

## Suggested Cleanup

The highest-priority cleanup tasks are:

1. Restore or inline the missing helper functions currently imported from `temp.model`.
2. Update `fig6_yield_cost_profit.py` to use `data/cost_required_ha.csv` or restore the expected `data/cost.csv`.
3. Add an environment file listing dependencies such as `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `joblib`, `matplotlib`, `seaborn`, `geopandas`, `requests`, `Pillow`, `mercantile`, `python-dotenv`, and `pandoc` if document conversion is part of the workflow.
4. Sync `docs/project_brief.md` with the actual script names and current file layout.
5. Add a short command checklist for regenerating models, optimization outputs, figures, and manuscript exports.
