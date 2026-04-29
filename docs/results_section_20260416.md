# Results 草稿

本文档根据 `fig/` 中的正式图件、`src/figs/` 中的绘图脚本以及对应汇总表整理，作为论文 Results 部分的中文正文底稿使用。

说明：

1. `Fig. 2` 的种植密度脚本当前将 `Density` 又乘以 15，导致图中绝对数值偏大。因此，下面关于密度的绝对数值采用 `data/model_vars.csv` 的原始口径，而不是 `fig2_region_management_difference_summary.csv` 中被重复换算后的数值。
2. `Fig. 9` 当前图中显示的是产量和利润两个面板，但其汇总表同时包含投入成本变化，因此结果段落中保留成本变化描述。

## 3. Results

### 3.1 Spatial distribution of observations and current management status

Field observations were located in the Huang-Huai-Hai region and were concentrated in Hebei and Shandong (Fig. 1). After data cleaning, the final dataset included 81 field-level observations, including 26 observations from Hebei and 55 from Shandong. This spatial distribution provided the basis for subsequent regional comparison of current management status and optimization potential.

Current management differed between the two regions in several aspects (Figs. 2-5). Sowing date was broadly similar between Hebei and Shandong, with median sowing dates of day 167.5 and day 169.0, respectively, indicating that the regional difference in planting time was relatively small. By contrast, plant density was clearly higher in Shandong than in Hebei. Based on the source data, the median density was 67,005 plants ha^-1 in Shandong and 55,275 plants ha^-1 in Hebei, suggesting a generally more intensive planting configuration in Shandong.

Regional differences were more pronounced for nutrient management (Fig. 3). Although the median fertilization frequency was 1 event in both regions, Shandong showed consistently higher nutrient input levels than Hebei. The median N, P, and K application rates in Shandong were 217.5, 75.0, and 85.93 kg ha^-1, respectively, compared with 189.0, 40.5, and 55.88 kg ha^-1 in Hebei. These results indicate that interregional differences in fertilization were driven more by application intensity than by the number of fertilization events.

Irrigation management also differed substantially between the two regions (Fig. 4). Hebei had a higher mean irrigation frequency than Shandong (1.27 vs. 0.95 events) and a slightly higher median irrigation electricity input (329.33 vs. 300.00 kWh ha^-1). In terms of irrigation mode, Shandong was dominated by drip irrigation, which accounted for 32 of 55 fields, whereas Hebei showed a more even distribution among flood irrigation, sprinkler irrigation, and drip irrigation, with no rainfed fields in the current sample. This pattern suggests that the irrigation structure in Shandong was more concentrated around drip-based systems, whereas Hebei retained greater heterogeneity in irrigation practice.

Pesticide management showed a different pattern (Fig. 5). The median number of pesticide applications was 2 in both regions, indicating similar operational frequency. However, pesticide cost was markedly higher in Hebei than in Shandong, with median values of 530.92 and 255.00 CNY ha^-1, respectively. Thus, regional differences in plant protection were reflected more strongly in expenditure level than in application frequency.

These management differences were accompanied by regional differences in baseline production performance (Fig. 6). Under current management, the median yield in Shandong reached 10,721.85 kg ha^-1, higher than the 9,585.00 kg ha^-1 observed in Hebei. Median production cost was similar between the two regions, at 12,150.00 CNY ha^-1 in Shandong and 12,171.68 CNY ha^-1 in Hebei. As a result, Shandong also achieved a higher baseline median profit than Hebei, at 12,031.56 versus 10,764.39 CNY ha^-1. Taken together, Figs. 2-6 indicate that Shandong generally operated under a more intensive input regime and achieved higher baseline productivity and profitability, whereas Hebei showed lower baseline yield and larger heterogeneity in several management dimensions.

### 3.2 Performance of candidate yield prediction models

The four candidate algorithms showed clear differences in predictive performance (Fig. 7). Among them, XGBoost achieved the best overall validation performance, with a validation R^2 of 0.652, RMSE of 1,168.06 kg ha^-1, and MAE of 932.71 kg ha^-1. Random forest ranked second, with a validation R^2 of 0.608 and RMSE of 1,238.41 kg ha^-1. Linear regression and Elastic Net performed less well, especially in terms of validation RMSE, indicating that purely linear structures were insufficient to capture the nonlinear relationships embedded in the field dataset.

The comparison between training and validation metrics further suggested that XGBoost provided a reasonable balance between predictive ability and overfitting control. Its train-validation R^2 gap was 0.077, smaller than that of random forest (0.135), indicating better generalization stability under the current sample size. Therefore, XGBoost was selected as the final yield prediction model for the subsequent optimization analysis.

### 3.3 Key predictors associated with yield variation

The XGBoost feature importance analysis revealed that yield variation was jointly associated with management and environmental factors (Fig. 8). When irrigation-mode dummy variables were aggregated, irrigation method was the most influential predictor, with a combined importance of 0.141. Among individual features, drip irrigation, plant density, sowing date, irrigation frequency, and pesticide application frequency ranked among the top predictors. Weather variables, including wind speed, relative humidity, radiation, and precipitation, also appeared in the top ten, indicating that background environmental conditions remained an important source of yield variation even after management variables were considered.

From a management perspective, the importance ranking suggests that water-related decisions and crop establishment variables were particularly relevant for explaining yield differences in the current dataset. In addition to irrigation mode and irrigation frequency, plant density and sowing date showed strong predictive contributions, supporting their prioritization in the maximum-yield optimization stage. Nutrient inputs, especially phosphorus and nitrogen, also contributed to model prediction, but their individual importance was lower than that of the leading irrigation and establishment variables.

### 3.4 Regional optimization gains under the maximum-yield strategy

The regional optimization analysis showed that both Hebei and Shandong retained substantial room for improvement under the maximum-yield strategy (Fig. 9). In Hebei, the optimized scenario increased regional median yield from 9,585.00 to 12,209.45 kg ha^-1, an increase of 2,624.45 kg ha^-1 or 27.38%. Over the same comparison, median profit increased from 10,764.39 to 16,163.22 CNY ha^-1, corresponding to a gain of 5,398.83 CNY ha^-1 or 50.15%. Input cost also increased, from 12,171.68 to 13,200.61 CNY ha^-1, but the proportional increase in cost (8.45%) was much smaller than the gains in yield and profit.

Shandong also showed clear but smaller relative gains than Hebei. Under the optimized scenario, regional median yield increased from 10,721.85 to 12,603.64 kg ha^-1, a gain of 1,881.79 kg ha^-1 or 17.55%. Median profit increased from 12,031.56 to 17,615.88 CNY ha^-1, corresponding to a gain of 5,584.32 CNY ha^-1 or 46.41%. Median input cost increased from 12,150.00 to 13,047.58 CNY ha^-1, representing a 7.39% increase. Thus, the optimization strategy improved both productivity and profitability in Shandong as well, but the relative yield gain was lower than that observed in Hebei.

The optimized management combinations also showed clear directional shifts compared with the regional baselines. In Hebei, the optimized scenario involved earlier sowing (DOY 164 vs. 167.5), higher density (64,000 vs. 55,275 plants ha^-1), increased N, P, and K inputs, lower pesticide cost, higher irrigation electricity input, and a shift toward drip irrigation. In Shandong, the optimized scenario likewise involved earlier sowing (DOY 163 vs. 169), slightly higher density (68,000 vs. 67,005 plants ha^-1), increased P input, a moderate increase in irrigation frequency and electricity input, and continued use of drip irrigation. These patterns suggest that the maximum-yield pathway in both regions relied mainly on earlier establishment, denser planting, and stronger irrigation support, while the magnitude of adjustment required was larger in Hebei.

Overall, Fig. 9 shows that the region with the lower current baseline, Hebei, exhibited the larger relative optimization gain, whereas Shandong, despite its higher current baseline, still retained meaningful room for coordinated yield and profit improvement. This result is consistent with the baseline comparisons shown in Figs. 2-6 and supports the interpretation that regional optimization potential is closely linked to current management level.

## 可直接压缩成论文正文的版本

如果你后续想把 Results 压缩成更接近期刊风格的正文，可以直接按下面四个小节组织：

1. `3.1 Current management status and baseline levels`
2. `3.2 Performance of the yield prediction model`
3. `3.3 Key drivers associated with yield variation`
4. `3.4 Optimization gains under the maximum-yield strategy`

## 一个需要你尽快处理的小问题

在当前版本中，`[src/figs/fig2_sowing.py](/D:/Workspace/Papers/maize_paper/src/figs/fig2_sowing.py)` 对 `Density` 做了重复面积换算，因此 `fig2` 的密度轴绝对值不适合直接引用。正文结果我已经绕开这个问题，但如果要投稿，建议先把这张图修正，否则图和文中的密度数值会对不上。
