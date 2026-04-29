# Materials and Methods Draft for SCI Submission

This document is an English version of the methods draft prepared from `themes/paper0314_precision_agriculture.md`, `docs/project_brief.md`, `docs/model_variable_table.md`, and the current analytical scripts in `src/model.py`, `src/optimize_management_yield.py`, and `src/plot_management_responses.py`. The text is written in a journal-style form suitable for further polishing for SCI submission.

## 2. Materials and Methods

### 2.1 Study area, datasets, and analytical unit

This study focused on the summer maize production system in the Huang-Huai-Hai region of China, with the current sample set mainly collected from Hebei and Shandong provinces. The Huang-Huai-Hai Plain is one of the major maize-producing regions in China and is characterized by substantial variation in management intensity, irrigation conditions, and productivity levels, making it a relevant case for evaluating management optimization potential under real production conditions (Ren et al., 2022; Wu et al., 2021). After data cleaning and harmonization, the final dataset used in this study included 81 field-level observations, each identified by a unique `uuid`. Among them, 26 observations were from Hebei and 55 were from Shandong. The analytical unit was the field-level observation, for which yield, management records, irrigation mode, cost information, and post-sowing weather aggregates were jointly available.

The modeling dataset was stored in `data/model_vars.csv`, whereas cost data were stored in `data/cost.csv`. The former was used for yield modeling and scenario simulation, and the latter was used for cost decomposition and profit estimation. The target variable was maize yield (`Yield`), which was standardized in the current database as kg ha^-1; results can be converted to kg mu^-1 for reporting where needed. Profit was treated as an evaluation outcome rather than a direct machine learning target. Specifically, profit was calculated after yield prediction using scenario-specific costs, thereby avoiding the introduction of derived economic relationships into the predictive model itself.

### 2.2 Variable construction, preprocessing, and leakage control

Variables were classified into management variables, environmental variables, and outcome variables. Management variables included sowing date (`Sow_DOY`), plant density (`Density`), nitrogen application rate (`Fer_N`), phosphorus application rate (`Fer_P`), potassium application rate (`Fer_K`), number of fertilization events (`Fer_Count`), number of pesticide applications (`Pest_Count`), pesticide cost (`Pest_Cost`), number of irrigation events (`Irr_Count`), irrigation electricity use (`Irr_Elec`), and irrigation-mode dummy variables (`Irr_None`, `Irr_Flood`, `Irr_Sprinkler`, and `Irr_Drip`). Environmental variables included post-sowing total precipitation (`Precip`), post-sowing total radiation (`Rad`), mean post-sowing relative humidity (`RH`), and maximum wind speed during the growing season (`Wind`). These weather variables were used to characterize the environmental background of each field, but they were held constant during scenario optimization so that only management-related changes were evaluated.

Data preprocessing followed a conservative and unified workflow. First, all fields except the sample identifier were converted to numeric format, and observations with missing target values were removed. Second, missing feature values were imputed using the median. Additional standardization was applied to linear regression and Elastic Net models, whereas tree-based models were fitted using variables in their original scales. Irrigation mode had already been encoded as mutually exclusive dummy variables in the processed dataset and therefore required no further transformation. Given the limited sample size, no high-dimensional interaction terms or additional derived indices were introduced, in order to reduce overfitting risk.

Leakage control was a central consideration in the methodological design. As noted by Kaufman et al. (2012), predictive performance can be substantially inflated when model inputs contain information that is directly derived from, or strongly entangled with, the target variable. Following this principle, the yield model in the present study used `Yield` as the sole prediction target and excluded `profit`, `total_cost`, `land_rent_cost`, and any other variables directly derived from yield-cost accounting. Profit was computed only after scenario-based yield prediction and cost estimation. This design ensured that the model addressed a clear question: how yield responds to management changes under a given environmental background, rather than a hybrid target confounded by accounting identities.

### 2.3 Yield modeling and explainability analysis

To obtain a robust response model that could support subsequent scenario optimization, four regression algorithms were compared: ordinary linear regression, Elastic Net, random forest, and XGBoost. Tree-based methods are well suited for agricultural systems because they can capture nonlinear responses and interactions among management and environmental variables. In particular, XGBoost has been widely validated for its predictive performance and computational efficiency in structured regression problems (Chen and Guestrin, 2016), and recent agricultural studies have shown that machine learning models integrating field management and weather variables can substantially improve crop yield prediction (Dhaliwal and Williams, 2024; Baio et al., 2023).

In the implemented workflow, samples were first randomly split into a training set (80%) and an independent validation set (20%) using a fixed random seed of 42. Model selection was then performed within the training set using five-fold cross-validation, with negative RMSE as the optimization criterion. All candidate models shared the same preprocessing pipeline, differing only in the estimator itself and whether standardization was applied. After training, model performance was evaluated on both the training and validation sets using the coefficient of determination (R^2), root mean square error (RMSE), and mean absolute error (MAE), calculated as:

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i-\hat y_i)^2}{\sum_{i=1}^{n}(y_i-\bar y)^2}
$$

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
$$

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
$$

The model showing the best validation performance with an acceptable level of overfitting was selected as the basis for management scenario simulation. In the current dataset, XGBoost provided the best overall predictive performance and was therefore adopted as the final yield model.

To interpret the relationships between key management variables and predicted yield, SHAP (SHapley Additive exPlanations) values were computed for the selected tree-based model. SHAP decomposes each prediction into additive feature contributions and provides both global importance and local directional interpretation (Lundberg and Lee, 2017). In this study, SHAP values were calculated for all observations using TreeExplainer, and dependence plots were generated for major management variables, including sowing date, plant density, nutrient inputs, irrigation frequency, irrigation electricity use, and pesticide cost. The purpose of this explainability analysis was not to infer causality, but to identify which management dimensions should be prioritized in the subsequent optimization step and to detect potential nonlinear response ranges relevant for decision support.

### 2.4 Baseline definition, maximum-yield optimization, and profit evaluation

The study adopted a two-scenario framework consisting of a current management baseline and a maximum-yield scenario. The current management baseline was not defined by model-predicted values. Instead, it was constructed directly from the observed sample distribution by calculating the median observed yield, median observed profit, and median values of each management variable at either the whole-dataset level or the regional subset level. Median values were used rather than means in order to reduce the influence of outliers in a relatively small field dataset.

The maximum-yield scenario was generated by perturbing only controllable management variables, while keeping weather-related variables and other non-management background features fixed at their observed field values. This design allowed the estimated gains to be interpreted as the improvement achievable through management adjustment alone under the existing field background. To avoid extrapolation beyond empirical support, candidate scenario values were strictly constrained within the observed data range. Continuous management variables were represented by three candidate levels derived from the empirical distribution, corresponding to the 0.10, 0.55, and 1.00 quantiles. Count variables, including fertilization frequency, pesticide application frequency, and irrigation frequency, were enumerated over the observed integer range from minimum to maximum. Irrigation mode was explored by enumerating four mutually exclusive options: no irrigation, flood irrigation, sprinkler irrigation, and drip irrigation. Candidate levels across variables were then combined using a Cartesian product to generate all feasible management scenarios.

For a given candidate scenario $s$, scenario-specific management values were substituted into each field-level feature vector to obtain predicted yields $\hat Y_{is}$. The regional scenario yield was then summarized as the median predicted yield across all fields:

$$
\tilde Y_s = \mathrm{median}(\hat Y_{1s}, \hat Y_{2s}, \ldots, \hat Y_{ns})
$$

The scenario with the highest median predicted yield was defined as the maximum-yield strategy. When two or more scenarios produced the same median yield, the one with the higher median predicted profit was retained. This definition intentionally targeted a regionally representative management combination rather than an individualized prescription for each field, which is consistent with the objective of evaluating management optimization potential at the regional decision-support level.

Profit evaluation followed a yield-first, profit-second framework. After linking the modeling data and cost data by `uuid`, three linear sub-models were fitted to estimate scenario-dependent cost components: sowing cost as a function of plant density, fertilizer cost as a function of N, P, and K application rates, and irrigation operating cost as a function of irrigation electricity use. In addition, the median irrigation device cost was calculated for each observed irrigation mode. For each field, a fixed-cost component was also estimated as the difference between observed total cost and the sum of these variable cost components. Accordingly, the total input cost for sample $i$ under scenario $s$ was expressed as:

$$
C_{is} = C^{fixed}_i + C^{sow}_s + C^{fert}_s + C^{pest}_s + C^{irr}_s + C^{device}_s
$$

Given a grain price $p$, which was set to 2.4 yuan kg^-1 in the implemented script, scenario profit was calculated as:

$$
\Pi_{is} = p \cdot \hat Y_{is} - C_{is}
$$

The regional scenario profit was defined as the median profit across all fields:

$$
\tilde \Pi_s = \mathrm{median}(\Pi_{1s}, \Pi_{2s}, \ldots, \Pi_{ns})
$$

Yield gain and profit gain were calculated at both the overall and regional scales as:

$$
\Delta Y = Y_{opt} - Y_{base}
$$

$$
\Delta Y(\%) = \frac{Y_{opt} - Y_{base}}{Y_{base}} \times 100
$$

$$
\Delta \Pi = \Pi_{opt} - \Pi_{base}
$$

$$
\Delta \Pi(\%) = \frac{\Pi_{opt} - \Pi_{base}}{\Pi_{base}} \times 100
$$

where $Y_{base}$ and $\Pi_{base}$ denote the baseline median yield and median profit, respectively, and $Y_{opt}$ and $\Pi_{opt}$ denote the corresponding values under the maximum-yield scenario.

### 2.5 Regional comparison and uncertainty statement

To examine spatial heterogeneity in optimization potential, the same baseline construction, scenario search, and profit evaluation procedures were repeated separately for the Hebei and Shandong subsets in addition to the full dataset. The purpose of this comparison was to test whether different current management baselines were associated with different magnitudes of potential gain, rather than to derive highly localized field prescriptions under the current sample size.

The results should be interpreted as model-based scenario estimates rather than direct evidence from intervention experiments. Therefore, the maximum-yield strategy represents the potential improvement achievable within the support of the observed dataset under the existing environmental background, rather than a guaranteed field outcome. In addition, the current dataset contains only 81 field observations and is spatially concentrated in Hebei and Shandong, while information on soil properties, cultivar diversity, and finer-resolution environmental covariates remains limited. As a result, the external generalizability of the model and the universality of the recommended adjustments still require validation with more years, more locations, and field-scale experiments. Nevertheless, by combining strict leakage control, region-specific baseline comparison, and scenario search constrained by observed data support, the framework provides a defensible way to quantify how much yield and profit could still be improved from current maize management in the Huang-Huai-Hai region.

## References

1. Baio, F.H.R., Santana, D.C., Teodoro, L.P.R., Oliveira, I.C., Gava, R., Oliveira, J.L.G., Silva Junior, C.A., Teodoro, P.E., and Shiratsuchi, L.S. 2023. Maize yield prediction with machine learning, spectral variables and irrigation management. *Remote Sensing*, 15(1), 79. https://doi.org/10.3390/rs15010079
2. Chen, T., and Guestrin, C. 2016. XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785
3. Dhaliwal, D.S., and Williams, M.M. II. 2024. Sweet corn yield prediction using machine learning models and field-level data. *Precision Agriculture*, 25, 51-64. https://doi.org/10.1007/s11119-023-10057-1
4. Kaufman, S., Rosset, S., Perlich, C., and Stitelman, O. 2012. Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 15. https://doi.org/10.1145/2382577.2382579
5. Lundberg, S.M., and Lee, S.-I. 2017. A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems*, 30, 4765-4774. https://arxiv.org/abs/1705.07874
6. Ren, H., Liu, M., Zhang, J., Liu, P., and Liu, C. 2022. Effects of agronomic traits and climatic factors on yield and yield stability of summer maize (*Zea mays* L.) in the Huang-Huai-Hai Plain in China. *Frontiers in Plant Science*, 13, 1050064. https://doi.org/10.3389/fpls.2022.1050064
7. Sihi, D., Dari, B., Kuruvila, A.P., Jha, G., and Basu, K. 2022. Explainable machine learning approach quantified the long-term (1981-2015) impact of climate and soil properties on yields of major agricultural crops across CONUS. *Frontiers in Sustainable Food Systems*, 6, 847892. https://doi.org/10.3389/fsufs.2022.847892
8. Wu, D., Xie, R., Ming, B., Hou, P., Xue, J., Ren, H., Zhang, W., Wang, K., and Li, S. 2021. The priority of management factors for reducing the yield gap of summer maize in the north of Huang-Huai-Hai region, China. *Journal of Integrative Agriculture*, 20(2), 450-459. https://doi.org/10.1016/S2095-3119(20)63294-4

## Metadata to verify before submission

1. The current project files do not explicitly preserve the sample year range or the original weather data source. These details should be added to Section 2.1 or to the Supplementary Materials before submission.
2. If the manuscript reports results in kg mu^-1 and yuan mu^-1, the unit-conversion rules should be stated explicitly in the Methods so that model variables, economic calculations, and reported results remain fully consistent.
