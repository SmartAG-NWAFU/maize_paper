# Results for submission-style English manuscript (Precision Agriculture style)

## 3. Results

### 3.1 Current management status and baseline regional performance

Current management differed consistently between Hebei and Shandong, providing the basis for subsequent analysis of regional optimization potential. As shown in **Fig. 2**, sowing date was broadly similar between the two regions, with median values of day 167.5 in Hebei and day 169.0 in Shandong. In contrast, plant density was clearly higher in Shandong than in Hebei. Based on the source data, median density reached 67,005 plants ha^-1 in Shandong and 55,275 plants ha^-1 in Hebei, indicating a more intensive crop establishment strategy in Shandong.

**Fig. 2. Regional differences in sowing date and plant density under current maize management.** Regional variation in current maize management for (a) sowing date and (b) plant density in Hebei and Shandong. Boxplots show the median, interquartile range, and range excluding outliers, and the overlaid points represent individual field observations.

This regional contrast extended to nutrient management. In **Fig. 3**, the median fertilization frequency was 1 in both regions, but nutrient input rates were consistently higher in Shandong. Median N, P, and K application rates were 217.5, 75.0, and 85.93 kg ha^-1 in Shandong, compared with 189.0, 40.5, and 55.88 kg ha^-1 in Hebei. Thus, regional differences in fertilization were driven more by application intensity than by the number of fertilization events.

**Fig. 3. Regional differences in fertilization practices under current maize management.** Regional variation in fertilization management between Hebei and Shandong, including (a) fertilization frequency, (b) nitrogen application, (c) phosphorus application, and (d) potassium application. Boxplots show the median, interquartile range, and range excluding outliers, and the overlaid points represent individual field observations.

Irrigation management also differed substantially between regions (**Fig. 4**). Although the median irrigation frequency was 1 in both regions, Hebei showed a slightly higher mean irrigation frequency than Shandong (1.27 vs. 0.95) and a higher median irrigation electricity input (329.33 vs. 300.00 kWh ha^-1). More importantly, irrigation mode differed markedly: drip irrigation dominated in Shandong (32 of 55 fields), whereas Hebei showed a more even distribution among flood, sprinkler, and drip irrigation. This suggests that Shandong had already shifted toward a more concentrated drip-based irrigation structure, while Hebei retained greater heterogeneity in irrigation practice.

**Fig. 4. Regional differences in irrigation frequency, irrigation input, and irrigation mode under current maize management.** Regional variation in irrigation management between Hebei and Shandong, including (a) irrigation frequency, (b) irrigation electricity use, and (c) the distribution of irrigation modes. Boxplots show the median, interquartile range, and range excluding outliers, and bars indicate the number of fields under each irrigation mode.

By comparison, pesticide management showed less regional divergence in application frequency but a clearer difference in cost (**Fig. 5**). The median number of pesticide applications was 2 in both regions, whereas median pesticide cost was substantially higher in Hebei than in Shandong (530.92 vs. 255.00 CNY ha^-1). This indicates that plant protection differences were reflected more strongly in expenditure level than in operational frequency.

**Fig. 5. Regional differences in pesticide application frequency and pesticide cost under current maize management.** Regional variation in pesticide management between Hebei and Shandong, including (a) pesticide application frequency and (b) pesticide cost. Boxplots show the median, interquartile range, and range excluding outliers, and the overlaid points represent individual field observations.

The above management patterns were reflected in baseline production performance (**Fig. 6**). Under current management, median yield was higher in Shandong than in Hebei (10,721.85 vs. 9,585.00 kg ha^-1), whereas median production cost was similar between regions (12,150.00 vs. 12,171.68 CNY ha^-1). Consequently, Shandong also showed a higher median baseline profit than Hebei (12,031.56 vs. 10,764.39 CNY ha^-1). Taken together, Figs. 2-6 indicate that Shandong operated under a higher management baseline and achieved higher baseline productivity and profitability, whereas Hebei maintained lower baseline performance and therefore greater apparent room for improvement.

**Fig. 6. Regional differences in yield, production cost, and profit under current maize management.** Regional variation in (a) maize yield, (b) production cost, and (c) profit under current management in Hebei and Shandong. Panel (d) shows the joint distribution of field observations in the cost-yield space. Boxplots show the median, interquartile range, and range excluding outliers, and the overlaid points represent individual field observations.

### 3.2 Performance of candidate yield prediction models

The four candidate algorithms differed clearly in predictive performance (**Fig. 7**). XGBoost achieved the best validation performance, with a validation R^2 of 0.652, RMSE of 1,168.06 kg ha^-1, and MAE of 932.71 kg ha^-1. Random forest ranked second, with a validation R^2 of 0.608 and RMSE of 1,238.41 kg ha^-1, whereas linear regression and Elastic Net performed less well. These results indicate that nonlinear models were better suited to capture the yield responses embedded in the field dataset.

XGBoost also showed a relatively small train-validation R^2 gap (0.077), compared with 0.135 for random forest, suggesting a better balance between fit and generalization. Therefore, XGBoost was selected as the final model for feature interpretation and scenario optimization.

**Fig. 7. Performance comparison of candidate models for maize yield prediction.** Comparison of the predictive performance of four candidate models for maize yield prediction. Panel (a) shows the coefficient of determination (R^2) for the training and validation sets, and panel (b) shows the corresponding root mean square error (RMSE). Models include linear regression, Elastic Net, random forest, and XGBoost.

### 3.3 Major predictors associated with yield variation

The XGBoost model showed that yield variation was jointly associated with management and environmental factors, but the strongest predictors were management-related (**Fig. 8**). Among individual predictors, drip irrigation ranked first, followed by plant density, sowing date, irrigation frequency, and pesticide application frequency. Weather variables, including wind speed, relative humidity, radiation, and precipitation, also appeared among the leading predictors, indicating that background environmental conditions remained important in shaping yield variation.

From a management perspective, the feature ranking identifies irrigation-related decisions and crop establishment as the dominant levers associated with yield improvement in the current dataset. Nutrient inputs, particularly phosphorus and nitrogen, also contributed to model prediction, although their individual importance was lower than that of the leading irrigation and establishment variables. These results provide the mechanistic basis for the optimization patterns reported below.

**Fig. 8. Relative importance of the top predictors in the XGBoost yield model.** Relative importance of the top predictors in the XGBoost model for maize yield prediction. Irrigation-mode dummy variables were aggregated into a single irrigation-method category for display. Higher values indicate greater contribution to model prediction.

### 3.4 Regional optimization gains under the maximum-yield strategy

The scenario analysis showed that both regions retained substantial scope for improvement under the maximum-yield strategy (**Fig. 9**). In Hebei, the optimized scenario increased median yield from 9,585.00 to 12,209.45 kg ha^-1, corresponding to a gain of 2,624.45 kg ha^-1 or 27.38%. Median profit increased from 10,764.39 to 16,163.22 CNY ha^-1, corresponding to a gain of 5,398.83 CNY ha^-1 or 50.15%. Median input cost also increased, from 12,171.68 to 13,200.61 CNY ha^-1, but the proportional increase in cost (8.45%) was much smaller than the gains in yield and profit.

Shandong also showed clear gains, although the relative increase in yield was smaller than in Hebei. Under the optimized scenario, median yield increased from 10,721.85 to 12,603.64 kg ha^-1, a gain of 1,881.79 kg ha^-1 or 17.55%. Median profit increased from 12,031.56 to 17,615.88 CNY ha^-1, corresponding to a gain of 5,584.32 CNY ha^-1 or 46.41%. Median input cost increased from 12,150.00 to 13,047.58 CNY ha^-1, representing a 7.39% increase. Thus, the optimized scenario improved both productivity and profitability in both regions, with larger relative gains in Hebei.

The optimized management patterns were consistent with the variable-importance results. In Hebei, the optimized scenario involved earlier sowing, higher density, increased N, P, and K inputs, more intensive irrigation input, and a shift toward drip irrigation. In Shandong, the optimized scenario also involved earlier sowing, slightly higher density, increased phosphorus input, more frequent irrigation, and continued use of drip irrigation. The larger magnitude of change required in Hebei is consistent with its lower current management baseline.

Overall, the results from Figs. 2-9 support a consistent interpretation: regions with a higher current management baseline, such as Shandong, already achieved relatively higher yield and profit, whereas regions with a lower baseline, such as Hebei, retained greater optimization potential. After irrigation-related variables, plant density, and sowing date were identified as key predictors of yield variation, the scenario analysis further demonstrated that coordinated adjustment of these management factors could simultaneously improve regional yield and profit.

**Fig. 9. Yield and profit gains under the maximum-yield optimization strategy in Hebei and Shandong.** Comparison of the baseline and optimized scenarios for (a) yield and (b) profit in Hebei and Shandong. The baseline represents the regional median under current management, whereas the optimized scenario represents the regional maximum-yield strategy identified by the machine-learning-based optimization framework. In Hebei, the optimized scenario increased median yield from 9,585.00 to 12,209.45 kg ha^-1 and median profit from 10,764.39 to 16,163.22 CNY ha^-1. In Shandong, the optimized scenario increased median yield from 10,721.85 to 12,603.64 kg ha^-1 and median profit from 12,031.56 to 17,615.88 CNY ha^-1.

## Integration note

This version is intentionally more compact than the expanded draft and is closer to the paragraph density and tone typically used in *Precision Agriculture*. If needed, it can now be integrated directly with the existing Introduction, Materials and Methods, and Conclusions sections.
