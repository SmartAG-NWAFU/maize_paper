**Quantifying Yield and Profit Improvement Potential of Maize\
through Management Optimization under Current Farming\
Conditions in the North China Plain of China**

Zhiming Xia^1^, Bin Chen^1^, Qi Shen^2^, Zeyun Liang^2^, Ming Tian^2^, Dengke Cao^3^, Yan Zhao^4^, Qiang Yu^1^, Gang Zhao^1,\*^

1 College of Soil and Water Conservation Science and Engineering, Northwest A&F University, Yangling, Shaanxi, 712100, China

2 Longping Kaihong Agricultural Technology (Beijing) Co., Ltd., Beijing, 100020, China

3 Hebei Lingyou Landscaping Engineering Co., Ltd., Zhengding, Hebei, 050800, China

4 Hebei Zhengding Haolin Family Farm, Zhengding, Hebei, 050800, China

\* Corresponding authors: Gang Zhao (gang.zhao@nwafu.edu.cn)

# Abstract

**Purpose:** We quantified how much maize yield and profit could improve through region-specific management optimization relative to current farmer management in the North China Plain of China.

**Methods:** Field observations from 81 summer maize fields in Hebei and Shandong Provinces were used to build leakage-controlled yield prediction models from management, field-status, weather, yield, and input-cost variables. Ordinary least squares, Elastic Net, random forest, and XGBoost models were compared. The selected model was then used as a regional response function for constrained maximum-yield scenario optimization, with controllable management variables restricted to empirically supported ranges. Profit was evaluated after yield optimization using scenario-specific input-cost estimates.

**Results:** XGBoost showed the best validation performance, with an R² of 0.652 and an RMSE of 1168.06 kg ha⁻¹. Irrigation method, plant density, sowing date, and irrigation frequency were the leading management-related predictors. The optimized scenarios selected drip irrigation in both regions and involved earlier sowing, higher plant density, adjusted nutrient inputs, and stronger irrigation support. Median yield and profit increased by 27.38% and 50.15% in Hebei, and by 17.55% and 46.41% in Shandong, respectively.

**Conclusion:** A constrained baseline-to-optimization framework can translate field observations into region-specific estimates of agronomic and economic improvement potential for precision maize management.

**Keywords:** Maize; precision agriculture; management optimization; machine learning; yield prediction; profitability

# Introduction

Maize production in the North China Plain is important for regional grain supply, but further yield improvement increasingly depends on more precise and economically viable management under resource and cost constraints rather than simple input expansion. Summer maize production in this region is characterized by high input intensity, strong field-level management heterogeneity, water and nutrient constraints, and large differences in yield and profit among farms. Previous studies have shown that maize yield gaps in China and the North China Plain are closely related to crop establishment, nutrient input, irrigation, and farmer-to-farmer management differences ([Chen et al. 2019](#bm_ref_chen_2019); [Chen et al. 2014](#bm_ref_chen_2014); [Chen et al. 2011](#bm_ref_chen_2011); [Liang et al. 2011](#bm_ref_liang_2011); [Meng et al. 2013](#bm_ref_meng_2013); [Wang et al. 2023](#bm_ref_wang_2023)). At the same time, inefficient nitrogen and water management can reduce resource-use efficiency, increase environmental pressure, and weaken farm profitability ([B.-Y. Liu et al. 2021](#bm_ref_liu_by_2021); [W. Liu et al. 2021](#bm_ref_liu_w_2021); [Mo et al. 2017](#bm_ref_mo_2017); [Yang et al. 2015](#bm_ref_yang_2015)). The practical question for precision maize management is not only which factors are associated with yield variation, but how much yield and profit improvement potential remains relative to current management baselines.

Machine learning provides useful tools for analysing yield variation in heterogeneous production data. Crop yield is affected by nonlinear and interacting effects of management, weather, soil, and field conditions, and algorithms such as random forest, gradient boosting, and XGBoost can capture complex relationships that are difficult to represent with simple linear models ([Jeong et al. 2016](#bm_ref_jeong_2016); [Khaki and Wang 2019](#bm_ref_khaki_wang_2019); [Shahhosseini et al. 2019](#bm_ref_shahhosseini_2019), 2021; [Van Klompenburg et al. 2020](#bm_ref_van_klompenburg_2020)). Explainable machine-learning approaches can further identify important predictors and improve the interpretability of data-driven decision support ([Hu et al. 2023](#bm_ref_hu_2023); [Smith et al. 2026](#bm_ref_smith_2026)). For maize and other crops, these methods have been used to predict yield and evaluate the relative contribution of sowing date, plant density, nutrient input, irrigation, weather, soil, and remote-sensing variables ([Baio et al. 2023](#bm_ref_baio_2023); [Maseko et al. 2024](#bm_ref_maseko_2024); [Paudel et al. 2021](#bm_ref_paudel_2021)). However, prediction accuracy and variable importance alone do not establish causal effects, and model extrapolation outside the support of observed data can be unreliable. Scenario analysis based on machine learning needs explicit constraints and cautious interpretation.

Despite these advances, many yield-modelling studies still stop at model comparison or variable-importance ranking. Such outputs indicate which factors are associated with yield variation, but they do not directly quantify how controllable management variables should be combined under realistic constraints or how much improvement is possible relative to the current management baseline. The distinction is important for farmers and regional decision makers, because maize management is a season-long system involving sowing date, plant density, nutrient supply, irrigation, crop protection, harvesting, and cost structure. Yield improvement also needs economic evaluation: a high-yield scenario is not practically useful if the additional input cost offsets the yield benefit ([Adhikari et al. 2023](#bm_ref_adhikari_2023); [Zhang et al. 2022](#bm_ref_zhang_2022); [Zhang et al. 2026](#bm_ref_zhang_2026)). A transparent approach is to estimate agronomic improvement potential under a constrained maximum-yield strategy first and then evaluate whether the yield-oriented scenario also improves profit.

The North China Plain provides an appropriate setting for this baseline-dependent analysis because field-level maize management varies substantially across farms and sampled provinces. In the present study, field observations from Hebei and Shandong were used to integrate crop establishment, nutrient input, irrigation, crop protection, weather, yield, and cost information. A leakage-controlled yield model was developed using management and environmental variables while excluding cost and profit variables that are directly derived from yield-cost accounting. The trained model was then used as a response function in a constrained maximum-yield scenario search, where only controllable management variables were changed and candidate solutions were kept within empirically supported regional ranges. Profit was evaluated after yield optimization using scenario-specific input-cost changes. The optimized scenarios are best interpreted as model-estimated regional decision-support configurations, not as field-validated causal prescriptions for individual farms.

The objectives were to develop a leakage-controlled and interpretable machine-learning model for maize yield prediction, identify the key management and environmental predictors associated with yield variation, quantify model-estimated yield improvement potential under constrained maximum-yield scenarios relative to current management baselines, and evaluate whether the yield-oriented optimization scenarios also improve profit after accounting for input-cost changes. It was expected that irrigation-related variables, plant density, sowing date, and nutrient input would be major predictors of maize yield variation; that constrained maximum-yield scenarios would increase predicted yield relative to current baseline management; and that regions with lower current management baselines would show greater relative optimization potential. By linking current management baselines with constrained maximum-yield scenario optimization and post-optimization profit assessment, the study provides a practical framework for quantifying model-estimated management improvement potential in maize production in the North China Plain.

# Materials and methods

## Study area

The North China Plain (NCP), located approximately between 32°-40°N and 114°-121°E, is one of the most important grain-producing regions in China and contributes substantially to national food production ([Fang et al. 2010](#bm_ref_fang_2010); [W. Liu et al. 2021](#bm_ref_liu_w_2021)). Cropping systems in this region are highly intensive, and the winter wheat-summer maize rotation is widely practiced across the plain ([Fang et al. 2010](#bm_ref_fang_2010)). The NCP is a low-relief alluvial plain with an average elevation of approximately 20 m above sea level. It is characterized by a warm-temperate monsoon climate, with mean annual temperatures ranging from 8 to 15°C and annual precipitation of approximately 500-900 mm. Precipitation is unevenly distributed throughout the year, with most rainfall concentrated in summer. Although this seasonal pattern generally supports summer maize growth, irrigation remains important for stabilizing crop water supply and reducing the risk of drought stress during critical growth stages ([Fang et al. 2010](#bm_ref_fang_2010); [Mo et al. 2017](#bm_ref_mo_2017); [Yang et al. 2015](#bm_ref_yang_2015)).

![**Figure 1.** Distribution of maize cultivation in China and location of the surveyed summer maize fields in the North China Plain. Panel (a) shows national maize planting area, the North China Plain boundary, and the regional position of the study sites; panels (b) and (c) show the field locations in Hebei and Shandong Province, respectively, on satellite basemaps. Orange shading indicates maize planting area, the green outline indicates the North China Plain, and point markers indicate the 81 surveyed fields during the 2025 summer maize season, including 26 fields in Hebei and 55 fields in Shandong. Scale bars and geographic coordinates are shown for spatial reference.](media/image1.png){width="6.0in" height="5.795833333333333in"}

The overall study workflow is summarized in [Figure 2](#bm_fig_2). It links the two sampled regions, field-level management and outcome data collection, descriptive statistical analysis, machine-learning-based yield modelling, and regional optimization analysis for yield and profit evaluation.

![[]{#bm_fig_2 .anchor}**Figure 2.** Overall workflow for field data collection, yield modelling, and regional management optimization. The workflow links the Hebei and Shandong study fields, field-level records of weather, crop establishment, irrigation, fertilization, crop protection, other field-status variables, yield and cost, descriptive statistical analysis, machine-learning model comparison, XGBoost-based yield prediction, and constrained optimization simulation. Model evaluation used R² and RMSE, and the optimization stage evaluated yield improvement, input-cost response, profit response, and region-specific management levers.](media/image2.png){alt="image" width="5.833333333333333in" height="7.56in"}

## Data collection and variable construction

During the 2025 summer maize season (June to October), field surveys and field measurements were conducted to characterize farmer management practices under real production conditions. The final modelling dataset contained 81 field-level observations, comprising 55 fields in Shandong Province and 26 fields in Hebei Province. Each observation was identified by a unique field identifier and represented one field managed under the farmer's current production practice.

Management records included sowing date, plant density, nitrogen, phosphorus, and potassium application rates, the number of fertilization events, pesticide applications, and irrigations, irrigation electricity consumption, irrigation method, and the corresponding costs of these management operations. The irrigation method was encoded as four mutually exclusive dummy variables representing no irrigation, flood irrigation, sprinkler irrigation, and drip irrigation. Lodging rate was retained as a field-status variable. Weather variables included cumulative precipitation, cumulative solar radiation, mean relative humidity, and maximum wind speed during the maize growing season. These weather variables were obtained by querying Open-Meteo (https://open-meteo.com/) using the geographical coordinates of the surveyed fields.

The calendar distribution of major field operations is summarized in [Figure 3](#bm_fig_3). Sowing and the first fertilization events were concentrated in June in both regions, whereas irrigation and crop-protection operations extended into July and August, and harvest mainly occurred from late September to October. Compared with Hebei, Shandong showed a broader and generally later operation window for irrigation and some crop-protection activities.

![[]{#bm_fig_3 .anchor}**Figure 3.** Calendar distribution of major farming operations in Hebei and Shandong during the 2025 summer maize season. Each point represents one recorded field operation, colors distinguish regions, the x-axis shows operation date, and the y-axis groups management activities, including sowing, fertilization, herbicide application, insecticide application, irrigation, fungicide application, growth-regulator application, and harvest. The figure summarizes the timing window and regional overlap of field operations used to characterize current farmer management.](media/image3.png){alt="image" width="5.833333333333333in" height="3.1795833333333334in"}

At maize maturity, grain yield was estimated from field sampling. Grain yield was calculated as:

  ------------------------------------------------------------------------
         $$Y = E \times K \times W \times 10^{- 6}$$               \(1\)
  ------ --------------------------------------------------------- -------

  ------------------------------------------------------------------------

where $Y$ is grain yield (kg ha⁻¹), $E$ is harvested ear number per hectare, $K$ is average kernel number per ear, and $W$ is 1000-kernel weight (g). The number of sampling points was determined according to field size and within-field growth variation. Three sampling points were randomly selected for fields smaller than 0.67 ha with relatively small growth variation, five sampling points were selected for fields larger than 0.67 ha or with relatively large growth variation, and nine sampling points were selected for fields of 6.67 ha or larger. At each sampling point, the harvested ear number was calculated as:

  ------------------------------------------------------------------------
         $$E = \frac{10,000}{S_{p} \times S_{r}}$$                 \(2\)
  ------ --------------------------------------------------------- -------

  ------------------------------------------------------------------------

where $S_{p}$ and $S_{r}$ are plant spacing and row spacing (m), respectively. Average kernel number per ear was estimated from 20 consecutive ears, and 1000-kernel weight was measured after air-drying the grains to 14% moisture content.

All agronomic and economic variables used in the analysis were expressed on a per-hectare basis. Total input cost was calculated as the sum of sowing cost, irrigation equipment cost, irrigation operating cost, pesticide cost, fertilization cost, and land rent. Profit was calculated after yield prediction rather than being used as a model input:

  ------------------------------------------------------------------------
         $$P = Y \times p - C$$                                    \(3\)
  ------ --------------------------------------------------------- -------

  ------------------------------------------------------------------------

where is profit (CNY ha⁻¹), is grain yield (kg ha⁻¹), is maize grain price, and is total input cost (CNY ha⁻¹). In this study, was set to 2.4 CNY/kg. A detailed description of the variables used for yield modelling is provided in [Table 1](#bm_table_1).

  ------------------------------------------------------------------------------------------------
  **Type**     **Variable**    **Description**                                       **Unit**
  ------------ --------------- ----------------------------------------------------- -------------
  Management   Region          Trial region code, 0=Hebei, 1=Shandong                \-

               Sow_DOY         Sowing day of the year                                Day of year

               Density         Planting density per hectare                          Plants ha⁻¹

               Fer_N           Nitrogen applied per hectare                          Kg ha⁻¹

               Fer_P           Phosphorus applied per hectare                        Kg ha⁻¹

               Fer_K           Potassium applied per hectare                         Kg ha⁻¹

               Fer_Count       Number of fertilization events                        \-

               Pest_Count      Number of pesticide applications                      \-

               Pest_Cost       Pesticide cost per hectare                            CNY ha⁻¹

               Irr_Count       Number of irrigation events                           \-

               Irr_Elec        Electricity used for irrigation per hectare           kWh ha⁻¹

               Irr_Sprinkler   Dummy variable for sprinkler irrigation               \-

               Irr_None        Dummy variable for no irrigation                      \-

               Irr_Drip        Dummy variable for drip irrigation                    \-

               Irr_Flood       Dummy variable for flood irrigation                   \-

               Lodging         Lodging rate during the growing season                \%

  Weather      Pre             Total precipitation during the growing season         mm

               Rad             Total solar radiation during the growing season       MJ/m²

               RH              Average relative humidity during the growing season   \%

               Wind            Maximum wind speed during the growing season          m/s

  Yield        Yield           Maize grain yield per hectare                         Kg ha⁻¹
  ------------------------------------------------------------------------------------------------

  : []{#bm_table_1 .anchor}**Table 1.** Variables used to construct the field-level yield-modelling dataset for the 2025 summer maize season in the North China Plain. Variables are grouped as management, weather, and yield-response variables; management and economic quantities are expressed on a per-hectare basis where applicable. Region was coded as 0 for Hebei and 1 for Shandong, and irrigation method was represented by mutually exclusive dummy variables for rainfed, flood, sprinkler, and drip irrigation. Yield was used as the response variable rather than as a predictor.

## Yield modelling and model evaluation

The yield model was developed to predict maize grain yield from observed management, field-status, and weather variables. The field identifier was excluded from modelling, and yield was used only as the response variable. Cost and profit variables were not included as model predictors, thereby avoiding leakage from economic accounting variables that are directly derived from yield or input-cost calculations.

All 20 predictors listed in [Table 1](#bm_table_1), excluding the field identifier and the yield target, were used as candidate inputs for the model. Standardization was applied to the two linear models, whereas tree-based models were fitted using the original variable scales. The irrigation method was already represented by mutually exclusive dummy variables and therefore required no additional categorical encoding.

Four regression algorithms were compared: ordinary least squares (OLS), Elastic Net, random forest (RF), and extreme gradient boosting (XGBoost). OLS was used as a baseline linear model, whereas Elastic Net was included because it combines L1 and L2 regularization and is suitable for correlated predictors ([Zou and Hastie 2005](#bm_ref_zou_hastie_2005)). RF and XGBoost were considered because tree-based methods can flexibly capture non-linear responses and interactions that are common in agricultural systems ([Breiman 2001](#bm_ref_breiman_2001); T. [Chen and Guestrin 2016](#bm_ref_chen_guestrin_2016)). Previous studies have shown that machine-learning approaches integrating environmental, management, and remotely sensed variables can improve crop-yield prediction across a range of spatial scales ([Maseko et al. 2024](#bm_ref_maseko_2024); [Paudel et al. 2021](#bm_ref_paudel_2021); [Van Klompenburg et al. 2020](#bm_ref_van_klompenburg_2020)). Field- and site-scale studies have also reported strong performance of tree-based models such as RF and XGBoost for maize and other crop-yield prediction tasks ([Maseko et al. 2024](#bm_ref_maseko_2024); [Nyéki et al. 2021](#bm_ref_nyeki_2021)).

The dataset was randomly divided into a training set (80%) and an independent validation set (20%) using a fixed random seed of 42. The same split was used for all algorithms. Each model was trained with the optimal hyperparameter settings selected before the final comparison, and fivefold cross-validation within the training set was used to estimate training-set predictive stability using negative RMSE as the scoring criterion. Model performance was evaluated on both the training and validation sets using the coefficient of determination ($R^{2}$), root mean square error (RMSE):

  ------------------------------------------------------------------------------------------------------------------------------
         $$R^{2} = 1 - \frac{\sum_{i = 1}^{n}(y_{i} - {\widehat{y}}_{i})^{2}}{\sum_{i = 1}^{n}(y_{i} - \bar{y})^{2}}$$   \(4\)
  ------ --------------------------------------------------------------------------------------------------------------- -------
         $$RMSE = \sqrt{\frac{1}{n}\sum_{i = 1}^{n}(y_{i} - {\widehat{y}}_{i})^{2}}$$                                    \(5\)

  ------------------------------------------------------------------------------------------------------------------------------

where $y_{i}$ and ${\widehat{y}}_{i}$ are the observed and predicted yields for observation $i$, respectively, $\bar{y}$ is the mean observed yield, and $n$ is the number of observations in the evaluated split. The final model for scenario analysis was selected primarily according to validation performance, while also considering the gap between training and validation $R^{2}$.

## Regional management optimization and profit evaluation

### Optimization objective and decision variables

After model comparison, the selected yield model was used as a regional response function for management-scenario evaluation. For each field, candidate values replaced the optimized management variables while non-optimized background variables were retained from the original field record. The regional objective was the median predicted yield across all fields within a region, and the management combination that maximized this objective was defined as the regional maximum-yield strategy.

The direct decision variables were sowing day of year, plant density, N, P, and K application rates, pesticide cost, irrigation electricity use, and irrigation method. Irrigation electricity use was set to zero under the no-irrigation scenario. Fertilization frequency, pesticide application frequency, and irrigation frequency were inferred from related management intensities using monotonic isotonic-regression relationships fitted within each region. This design avoided unrealistic combinations in which input rates and operation counts moved in contradictory directions.

### Feasible search and extrapolation control

Irrigation methods observed in each region were evaluated by enumeration. For each candidate irrigation method, continuous decision variables were searched using differential evolution because the objective function was nonlinear, non-smooth, and non-differentiable ([Storn and Price 1997](#bm_ref_storn_price_1997)). The penalized objective subtracted a standardized nearest-neighbour distance penalty from median predicted yield to discourage multivariate management combinations far from observed regional management records.

The feasible search space was region specific. Search bounds for continuous variables were restricted to the 5th and 95th percentiles of the observed regional distributions. Candidate values were quantized before model evaluation: sowing date to 1 d, plant density to 500 plants ha⁻¹, N, P, and K inputs to 5 kg ha⁻¹, pesticide cost to 50 CNY ha⁻¹, and irrigation electricity use to 50 kWh ha⁻¹. The differential-evolution search used a random seed of 42, 120 maximum iterations, a population-size multiplier of 12, and a distance-penalty weight of 150.0.

### Scenario cost and profit evaluation

Optimized regional yield was reported as the unpenalized median predicted yield under the selected regional management vector. The current regional baseline was calculated directly from observed data, not from model predictions. Specifically, baseline yield, input cost, profit, and management values were calculated as regional medians of the observed field records.

Scenario input cost was estimated using a yield-first, profit-second framework. Cost records were linked to the modelling data by field identifier. Three linear cost submodels were fitted from the observed data: sowing cost as a function of plant density, fertilization cost as a function of N, P, and K application rates, and irrigation running cost as a function of irrigation electricity use. Irrigation device cost was assigned as the median observed device cost for the candidate irrigation method.

For each field, the observed cost not explained by scenario-dependent sowing, fertilization, pesticide, irrigation-running, and irrigation device components was treated as a fixed residual. Total scenario cost was calculated as the sum of this fixed residual and the scenario-dependent cost components. Scenario profit was calculated from maize grain price multiplied by predicted yield minus total scenario input cost. Yield and profit gains were then calculated by comparing the optimized maximum-yield scenario with the observed regional baseline.

# Results

## Regional differences in current management and baseline performance

Current management differed systematically between Hebei and Shandong, and these regional contrasts provided the empirical basis for understanding subsequent optimization potential. Sowing date was broadly comparable between the two regions, with median sowing dates of day 167.5 in Hebei and day 169.0 in Shandong ([Figure 4](#bm_fig_4)). Large interregional differences in current production performance were unlikely to be explained by planting time alone. By contrast, plant density differed more clearly between regions. Based on the source dataset, Shandong had a substantially higher median density than Hebei (67,005 vs. 55,275 plants ha⁻¹), suggesting that Shandong generally operated under a more intensive crop establishment strategy.

![[]{#bm_fig_4 .anchor}**Figure 4.** Regional differences in crop establishment under current maize management. Panel (a) shows sowing date as day of year, and panel (b) shows plant density in plants ha⁻¹ for Hebei and Shandong. Boxplots show the median and interquartile range, whiskers show the non-outlier range, points represent individual fields, and text annotations report regional means. These variables describe the establishment baseline used in yield modelling and optimization.](media/image4.png){width="5.833331146106737in" height="2.5716524496937883in"}

The regional contrast observed in crop establishment was accompanied by clear differences in nutrient management ([Figure 5](#bm_fig_5)). The median fertilization frequency was 1 event in both regions, indicating that farmers in Hebei and Shandong did not differ greatly in the number of fertilization operations. However, nutrient application intensity was consistently higher in Shandong. Median N, P, and K application rates were 217.5, 75.0, and 85.93 kg ha⁻¹ in Shandong, compared with 189.0, 40.5, and 55.88 kg ha⁻¹ in Hebei. The main regional difference in fertilization was not the frequency of application, but the amount of nutrient supplied per unit area. Together with the density pattern shown in [Figure 4](#bm_fig_4), these results indicate that Shandong had already reached a relatively higher baseline level of agronomic input intensity.

![[]{#bm_fig_5 .anchor}**Figure 5.** Regional differences in fertilization practices under current maize management. Panels show (a) number of fertilization events, (b) nitrogen application rate, (c) phosphorus application rate, and (d) potassium application rate. Nutrient rates are expressed as kg ha⁻¹. Boxplots show the median and interquartile range, whiskers show the non-outlier range, points represent individual fields, and text annotations report regional means.](media/image5.png){width="5.833332239720035in" height="4.262713254593176in"}

Regional differences were also evident in irrigation management, which adds an important water-regulation dimension to the contrast in current management intensity ([Figure 6](#bm_fig_6)). The median irrigation frequency was 1 event in both regions, but Hebei exhibited a slightly higher mean irrigation frequency than Shandong (1.27 vs. 0.95) and a higher median irrigation electricity input (329.33 vs. 300.00 kWh ha⁻¹). More importantly, irrigation mode distribution differed substantially. Drip irrigation dominated in Shandong, accounting for 32 of the 55 sampled fields, whereas Hebei showed a more even distribution among flood irrigation, sprinkler irrigation, and drip irrigation. Compared with Hebei, Shandong already relied more strongly on drip-based systems, whereas Hebei retained greater heterogeneity in water management. The irrigation contrast directly links current management differences to the yield-driving factors examined later in the modelling analysis.

![[]{#bm_fig_6 .anchor}**Figure 6.** Regional differences in irrigation management under current maize management. Panels show (a) irrigation events, (b) irrigation electricity use in kWh ha⁻¹, and (c) the number of fields using rainfed, flood, sprinkler, or drip irrigation. Boxplots show the median and interquartile range, whiskers show the non-outlier range, points represent individual fields, and bars summarize irrigation-method counts by region.](media/image6.png){width="5.833332239720035in" height="4.485170603674541in"}

Unlike fertilization and irrigation, pesticide management showed only limited regional divergence in operational frequency, but a clearer difference in expenditure level ([Figure 7](#bm_fig_7)). The median number of pesticide applications was 2 in both Hebei and Shandong, indicating similar field operation frequency. However, median pesticide cost in Hebei was markedly higher than in Shandong (530.92 vs. 255.00 CNY ha⁻¹). Regional differences in plant protection were expressed more through cost structure than through the number of spray events. Together, the evidence indicates that current management differences between the two regions were multidimensional, involving crop establishment, nutrient supply, irrigation strategy, and input expenditure.

![[]{#bm_fig_7 .anchor}**Figure 7.** Regional differences in pesticide management under current maize management. Panel (a) shows pesticide application frequency, and panel (b) shows pesticide cost in CNY ha⁻¹. Boxplots show the median and interquartile range, whiskers show the non-outlier range, points represent individual fields, and text annotations report regional means.](media/image7.png){width="5.833331146106737in" height="2.57165135608049in"}

The management contrasts were reflected in baseline production performance ([Figure 8](#bm_fig_8)). Shandong had a higher median yield than Hebei under current management (10,721.85 vs. 9,585.00 kg ha⁻¹), while median production cost was similar between regions (12,150.00 vs. 12,171.68 CNY ha⁻¹). Because the cost difference was small but the yield gap was substantial, Shandong also achieved a higher median baseline profit than Hebei (12,031.56 vs. 10,764.39 CNY ha⁻¹). The joint distribution in the cost-yield space further indicates that Shandong generally occupied a more favorable production position under current conditions. Taken together, the baseline results show a consistent pattern: compared with Hebei, Shandong had a higher baseline management intensity and already achieved higher yield and profit, whereas Hebei maintained a lower baseline with potentially larger room for improvement.

![[]{#bm_fig_8 .anchor}**Figure 8.** Regional variation in production outcomes and cost-yield relationships under current management. Panels show (a) maize grain yield in kg ha⁻¹, (b) total production cost in CNY ha⁻¹, and (c) profit in CNY ha⁻¹ for Hebei and Shandong. Panel (d) shows the joint distribution of field observations in cost-yield space, with colors indicating region and marker symbols indicating cost-yield relationship types. Dashed reference lines indicate median cost and yield, and H-L indicates high cost-low yield.](media/image8.png){width="5.833332239720035in" height="4.262714348206474in"}

## Yield model performance and predictor importance

Because the scenario analysis depends on the credibility of the yield prediction model, candidate algorithms were compared before scenario optimization ([Figure 9](#bm_fig_9)). XGBoost achieved the best overall validation performance, with a validation R² of 0.652, RMSE of 1,168.06 kg ha⁻¹, and MAE of 932.71 kg ha⁻¹. Random forest ranked second, with a validation R² of 0.608 and RMSE of 1,238.41 kg ha⁻¹, whereas linear regression and Elastic Net performed less well, especially in terms of validation error. Beyond predictive accuracy, the train-validation gap also supported the selection of XGBoost as the final model. The difference between training and validation R² was 0.077 for XGBoost, smaller than the corresponding gap for random forest (0.135), suggesting a more favorable balance between fitting ability and generalization. XGBoost was used as the core model for the subsequent explainability analysis and scenario optimization.

![[]{#bm_fig_9 .anchor}**Figure 9.** Comparison of predictive performance for four candidate maize-yield models. Panel (a) shows the coefficient of determination (R²) for training and validation sets, and panel (b) shows root mean square error (RMSE, kg ha⁻¹) for the same splits. Bars compare XGBoost, random forest, linear regression, and Elastic Net, and value labels report the corresponding metric values.](media/image9.png){alt="image" width="5.833333333333333in" height="2.4441338582677163in"}

The selected XGBoost model showed that yield variation was jointly influenced by management and environmental factors, but the leading predictors were dominated by management-related variables ([Figure 10](#bm_fig_10)). Among the individual predictors, drip irrigation had the highest importance, followed by plant density, sowing date, irrigation frequency, and pesticide application frequency. Weather variables such as wind speed, relative humidity, radiation, and precipitation also ranked among the top predictors, indicating that environmental background still shaped yield responses even when management information was explicitly included in the model. The feature ranking connects the baseline comparisons with the optimization results: the prominence of irrigation-related variables, plant density, and sowing date suggests that regional differences in water management and crop establishment were not merely descriptive contrasts, but were also closely associated with yield outcomes. Nutrient variables, particularly phosphorus and nitrogen inputs, also contributed to model prediction, although their individual importance was lower than that of the leading irrigation and establishment variables.

![[]{#bm_fig_10 .anchor}**Figure 10.** Relative importance of the top 10 predictors in the selected XGBoost maize-yield model. Bars show model-derived relative importance values, with larger values indicating stronger contribution to yield prediction in the fitted model. Irrigation-mode dummy variables for rainfed, flood, sprinkler, and drip irrigation were aggregated into a single irrigation-method category for display.](media/image10.png){width="5.833332239720035in" height="3.564623797025372in"}

## Regional gains under the maximum-yield optimization strategy

  -------------------------------------------------------------------------------------------
                                    **Baseline**   **Optimal**   **Baseline**   **Optimal**
  --------------------------------- -------------- ------------- -------------- -------------
  Region                            Hebei          Hebei         Shandong       Shandong

  Plant density                     55275          64000         67005          68000

  Sowing date                       167.5          164           169            163

  N amount                          189            215           217.5          220

  P amount                          40.5           55            75             80

  K amount                          55.88          80            85.93          85

  Fertilization frequency           1              1             1              2

  Pesticide cost                    530.92         350           255            300

  Pesticide application frequency   2              1             2              2

  Irrigation electricity usage      329.33         550           300            650

  Irrigation frequency              1              2             1              2

  Irrigation method                 Flood          Drip          Drip           Drip
  -------------------------------------------------------------------------------------------

  : []{#bm_table_2 .anchor}**Table 2.** Baseline and optimized management configurations under the constrained maximum-yield strategy in Hebei and Shandong. Baseline values represent regional current-management medians, whereas optimal values represent the model-estimated settings selected by the optimization procedure within empirically supported management bounds.

The direction of optimized management adjustment was consistent with the feature-importance results ([Table 2](#bm_table_2)). In Hebei, the optimized strategy involved earlier sowing (DOY 164 vs. 167.5), higher plant density (64,000 vs. 55,275 plants ha⁻¹), increased N, P, and K inputs, higher irrigation frequency, stronger irrigation electricity input, and a shift toward drip irrigation. In Shandong, the optimized strategy also involved earlier sowing (DOY 163 vs. 169), a slight increase in density (68,000 vs. 67,005 plants ha⁻¹), increased phosphorus input, more frequent irrigation, higher irrigation electricity input, and continued reliance on drip irrigation. The magnitude of adjustment was larger in Hebei, which is consistent with its lower initial management baseline and larger optimization gain.

Guided by the predictive model and the ranking of key management drivers, the scenario analysis quantified how much yield and profit could improve relative to the current regional baseline ([Figure 11](#bm_fig_11)). In Hebei, the optimized scenario increased median yield from 9,585.00 to 12,209.45 kg ha⁻¹, corresponding to a gain of 2,624.45 kg ha⁻¹ or 27.38%. Over the same comparison, median profit increased from 10,764.39 to 16,163.22 CNY ha⁻¹, corresponding to a gain of 5,398.83 CNY ha⁻¹ or 50.15%. Median input cost also increased, from 12,171.68 to 13,200.61 CNY ha⁻¹, but the cost increase (8.45%) was much smaller than the gains in yield and profit. Shandong also showed positive optimization effects, although the relative gain in yield was smaller than in Hebei. Under the optimized scenario, median yield increased from 10,721.85 to 12,603.64 kg ha⁻¹, a gain of 1,881.79 kg ha⁻¹ or 17.55%. Median profit increased from 12,031.56 to 17,615.88 CNY ha⁻¹, corresponding to a gain of 5,584.32 CNY ha⁻¹ or 46.41%. Median input cost increased from 12,150.00 to 13,047.58 CNY ha⁻¹, representing a 7.39% increase. The optimization framework improved both productivity and profitability in both regions, while the larger relative gain in Hebei indicates greater remaining room for improvement where the current management baseline was lower.

![[]{#bm_fig_11 .anchor}**Figure 11.** Yield and profit gains under the maximum-yield optimization strategy in Hebei and Shandong. The baseline represents the regional median under current management, whereas the optimized scenario represents the regional maximum-yield strategy identified by the machine-learning-based optimization framework. In Hebei, the optimized scenario increased median yield from 9,585.00 to 12,209.45 kg ha⁻¹ and median profit from 10,764.39 to 16,163.22 CNY ha⁻¹. In Shandong, the optimized scenario increased median yield from 10,721.85 to 12,603.64 kg ha⁻¹ and median profit from 12,031.56 to 17,615.88 CNY ha⁻¹.](media/image11.png){width="5.833331146106737in" height="2.6441458880139983in"}

# Discussion

We developed a field-level baseline-to-optimization framework for estimating how much maize yield and profit can still be improved under current farming conditions in the North China Plain. The framework combines leakage-controlled yield modelling, interpretable feature ranking, constrained maximum-yield scenario search, and post-optimization profit accounting. In doing so, it shifts the role of machine learning from yield prediction alone to opportunity estimation: how far current practice is from an empirically attainable management configuration, which management pathway accounts for that distance, and whether the predicted yield gain remains economically useful. Precision agriculture recommendations become more useful when they are baseline-aware. The same technology, model, or management package can have different value depending on the starting level of local farmer practice.

## Baseline determines regional optimization potential

The regional comparison shows that optimization potential is not an absolute property of a crop system; it is conditional on the current management baseline. Shandong already had higher plant density, greater nutrient input, higher yield, and higher profit. Hebei started from a lower productivity baseline and consequently showed the larger proportional yield response after optimization. The same algorithmic search has different agronomic meaning in the two regions: in Shandong it mainly refines an already intensive system, whereas in Hebei it identifies a broader management upgrade.

Baseline-dependent opportunity connects the study to modern yield-gap analysis. Yield-gap studies emphasize that the distance to an attainable frontier varies across space and time, and that regions closer to stagnation or farther from attainable production frontiers require different intervention priorities ([Gerber et al. 2024](#bm_ref_gerber_2024); [van Ittersum et al. 2013](#bm_ref_van_ittersum_2013)). Studies in China have reached a similar conclusion for maize: farmer management heterogeneity, input allocation, and operational implementation explain a substantial share of remaining yield gaps ([Chen et al. 2019](#bm_ref_chen_2019); [Meng et al. 2013](#bm_ref_meng_2013); [Wang et al. 2023](#bm_ref_wang_2023)). Precision agriculture should not begin with a generic prescription. It should begin with a diagnosis of where the field or region sits relative to its current attainable frontier. That diagnosis determines whether decision support should emphasize fine tuning, input reallocation, or coordinated system upgrading.

## Irrigation and crop establishment are the dominant levers

Irrigation and crop establishment emerged as the management core of the optimization pathway. Irrigation method, plant density, sowing date, and irrigation frequency ranked among the leading predictors, and the optimized scenarios generally selected drip irrigation, earlier sowing, higher density, and stronger irrigation support. These variables are not independent levers. Plant density and sowing date define the crop stand and its seasonal demand for radiation, water, and nutrients; irrigation method and frequency determine whether that demand can be met under uneven summer rainfall.

High-yield maize systems in China depend on coordinated management rather than on increasing one input in isolation. Integrated soil-crop management and optimized crop management have been shown to increase grain production while reducing environmental cost or closing maize supply gaps ([Chen et al. 2014](#bm_ref_chen_2014); [Chen et al. 2011](#bm_ref_chen_2011); [Luo et al. 2023](#bm_ref_luo_2023)). More recent work on climate-smart crop production also argues for co-optimization of calendar decisions and management practices rather than single-factor adjustment ([Xiao et al. 2024](#bm_ref_xiao_2024); [Liu et al. 2026](#bm_ref_liu_2026)). For the North China Plain, where rainfall is concentrated but unevenly distributed, water delivery reliability is a particularly important condition for turning stand-level yield capacity into harvested yield ([Fang et al. 2010](#bm_ref_fang_2010); [Mo et al. 2017](#bm_ref_mo_2017); [Yang et al. 2015](#bm_ref_yang_2015)). Decision support should recommend management bundles, not isolated actions: earlier sowing and suitable density create yield capacity, while irrigation method and frequency decide whether that capacity can be realized.

## Yield-oriented optimization can remain economically meaningful

The optimized scenarios increased profit as well as yield, which is important because agronomic optimum and economic optimum are not automatically aligned. A yield-maximizing recommendation can fail at farm level if the required input cost absorbs the additional revenue. Here, input cost increased only modestly relative to yield and profit gains, so the optimized scenarios were not simply high-cost intensification. They were higher-performing management bundles in which added yield value exceeded added cost.

Profit-aware evaluation is central to precision agriculture because the unit of decision is not only a yield response, but a management choice made under cost and risk. Within-field evidence from corn systems shows that yield stability and gross margin can vary together in ways that matter for conservation and site-specific management ([Adhikari et al. 2023](#bm_ref_adhikari_2023)). Studies of maize management also show that yield, water productivity, nitrogen use, and profit must be evaluated jointly when recommending density, nutrient input, or irrigation strategies ([Zhang et al. 2022](#bm_ref_zhang_2022); [Zhang et al. 2026](#bm_ref_zhang_2026)). The present yield-first, profit-second design makes that tradeoff explicit. It allows a biological response surface to identify attainable yield scenarios, then tests whether those scenarios survive an economic screen. That separation is useful for decision support because it shows whether profit gains arise from genuine productivity improvement rather than from hidden assumptions about input cost.

## Constrained machine learning complements process-based decision support

The methodological value of the framework lies in constrained rather than unconstrained machine learning. Process-based crop models remain indispensable for testing mechanisms, climate sensitivity, soil-water-nitrogen processes, and genotype-by-environment interactions ([Holzworth et al. 2014](#bm_ref_holzworth_2014); [Jones et al. 2003](#bm_ref_jones_2003); [Keating et al. 2003](#bm_ref_keating_2003)). However, real farmer management data contain heterogeneous irrigation devices, crop-protection operations, timing decisions, and cost structures that are difficult to parameterize completely in process models. Machine learning can learn from such observational complexity, but only if the scenario search is kept inside the support of observed management conditions.

The approach is consistent with recent agricultural systems and precision agriculture literature. Machine learning has become powerful for large-scale yield forecasting and field-level prediction ([Paudel et al. 2021](#bm_ref_paudel_2021); [Nyéki et al. 2021](#bm_ref_nyeki_2021)), but accurate prediction does not by itself justify management prescription. Fertilizer-recommendation studies in Precision Agriculture have shown that machine-learning models can be misleading when recommendations are made outside the domain where the model has reliable support ([Tanaka et al. 2024](#bm_ref_tanaka_2024)). Explainable-AI studies similarly warn against treating black-box associations as causal management effects ([Hu et al. 2023](#bm_ref_hu_2023)). The constrained search used here addresses that problem directly. It uses machine learning to prioritize management directions and estimate opportunity, while leaving mechanism testing, robustness assessment, and transferability to multi-year field experiments and process-based crop modelling.

## Limitations and future work

The dataset contains 81 field observations from Hebei and Shandong in one growing season, so it does not capture the full climatic, soil, cultivar, and management diversity of the North China Plain. Second, soil properties, cultivar traits, remote-sensing indicators, and multi-year weather variation were not explicitly included. Third, the optimized scenarios are model-estimated outcomes from observational data, not causal effects verified through field intervention. Although empirical bounds and the distance penalty reduce unsupported extrapolation, they cannot replace experimental validation.

Future work should test the identified management pathways across more years, sites, soil conditions, and cultivar backgrounds. The framework would also benefit from richer environmental covariates, remote-sensing indicators, and explicit uncertainty analysis for scenario predictions. A priority is to link this data-driven optimization workflow with process-based crop modelling and field experiments. That combination would clarify whether the estimated gains remain robust across weather years, soil types, and market conditions, and would move the framework from regional opportunity estimation toward validated management recommendation.

# Conclusions

Maize management improvement potential in the North China Plain was quantified using a leakage-controlled XGBoost model and a constrained baseline-to-optimization framework. Irrigation method, plant density, sowing date, and irrigation frequency were the leading management-related predictors of yield variation.

The maximum-yield scenarios increased both yield and profit in Hebei and Shandong. Median yield increased by 27.38% in Hebei and 17.55% in Shandong, while median profit increased by 50.15% and 46.41%, respectively. The larger relative gain in Hebei indicates that regions with lower current management baselines may retain greater optimization potential.

The framework shows how field observations can be translated into region-specific estimates of agronomic and economic opportunity. Its use should remain constrained to the support of the observed data until multi-year, multi-site, and field-experimental validation is available.

# Statements and Declarations

## Funding

This work was funded by Longping Kaihong Agricultural Technology (Beijing) Co., Ltd. (隆平开鸿(北京)农业科技有限公司). The authors gratefully acknowledge financial support from the Key Research and Development Program of Shaanxi (Grant No. 2023-ZDLNY-64).

## Competing Interests

The authors have no relevant financial or non-financial interests to disclose.

## Ethics Approval

Not applicable. This study used crop-management and field-observation data and did not involve human participants or animals.

## Consent to Participate

Not applicable.

## Consent to Publish

All authors have approved the manuscript and consent to its publication.

## Data Availability

The data that support the findings of this study are available from the corresponding author upon reasonable request.

## Code Availability

The analysis code used to generate the model outputs, optimization outputs, and figures is publicly available at https://github.com/summer-zm/maize_paper.

## Author Contributions

Zhiming Xia, Bin Chen, Qiang Yu, and Gang Zhao contributed to the study conception and design. Zhiming Xia and Bin Chen performed the methodology development, data analysis, model construction, optimization analysis, and visualization. Qi Shen, Zeyun Liang, Ming Tian, Dengke Cao, and Yan Zhao contributed to field investigation, data collection, data curation, and resources. Zhiming Xia wrote the first draft of the manuscript. Bin Chen, Qi Shen, Zeyun Liang, Ming Tian, Dengke Cao, Yan Zhao, Qiang Yu, and Gang Zhao reviewed and edited the manuscript. Qiang Yu and Gang Zhao supervised the work and contributed to funding acquisition. All authors read and approved the final manuscript.

# References

[]{#bm_ref_adhikari_2023 .anchor}Adhikari, K., Smith, D. R., Hajda, C., & Kharel, T. P. (2023). Within-field yield stability and gross margin variations across corn fields and implications for precision conservation. *Precision Agriculture*, *24*(4), 1401--1416. https://doi.org/10.1007/s11119-023-09995-7

[]{#bm_ref_baio_2023 .anchor}Baio, F. H. R., Santana, D. C., Teodoro, L. P. R., Oliveira, I. C. de, Gava, R., de Oliveira, J. L. G., et al. (2023). Maize Yield Prediction with Machine Learning, Spectral Variables and Irrigation Management. *Remote Sensing*, *15*(1), 79. https://doi.org/10.3390/rs15010079

Binder, J., Graeff, S., Link, J., Claupein, W., Liu, M., Dai, M., & Wang, P. (2008). Model-Based Approach to Quantify Production Potentials of Summer Maize and Spring Maize in the North China Plain. *Agronomy Journal*, *100*(3), AGJ2AGRONJ20070226. https://doi.org/10.2134/agronj2007.0226

[]{#bm_ref_breiman_2001 .anchor}Breiman, L. (2001). Random Forests. *Machine Learning*, *45*(1), 5--32. https://doi.org/10.1023/A:1010933404324

[]{#bm_ref_chen_2019 .anchor}Chen, G., Cao, H., Chen, D., Zhang, L., Zhao, W., Zhang, Y., et al. (2019). Developing sustainable summer maize production for smallholder farmers in the North China Plain: An agronomic diagnosis method. *Journal of Integrative Agriculture*, *18*(8), 1667--1679. https://doi.org/10.1016/S2095-3119(18)62151-3

[]{#bm_ref_chen_guestrin_2016 .anchor}Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785--794). New York, NY, USA: Association for Computing Machinery. https://doi.org/10.1145/2939672.2939785

[]{#bm_ref_chen_2014 .anchor}Chen, X., Cui, Z., Fan, M., Vitousek, P., Zhao, M., Ma, W., et al. (2014). Producing more grain with lower environmental costs. *Nature*, *514*(7523), 486--489. https://doi.org/10.1038/nature13609

[]{#bm_ref_chen_2011 .anchor}Chen, X.-P., Cui, Z.-L., Vitousek, P. M., Cassman, K. G., Matson, P. A., Bai, J.-S., et al. (2011). Integrated soil--crop system management for food security. *Proceedings of the National Academy of Sciences*, *108*(16), 6399--6404. https://doi.org/10.1073/pnas.1101419108

[]{#bm_ref_fang_2010 .anchor}Fang, Q. X., Ma, L., Green, T. R., Yu, Q., Wang, T. D., & Ahuja, L. R. (2010). Water resources and water use efficiency in the North China Plain: Current status and agronomic management options. *Agricultural Water Management*, *97*(8), 1102--1116. https://doi.org/10.1016/j.agwat.2010.01.008

Fiorentini, M., Schillaci, C., Denora, M., Zenobi, S., Deligios, P. A., Santilocchi, R., et al. (2024). Fertilization and soil management machine learning based sustainable agronomic prescriptions for durum wheat in Italy. *Precision Agriculture*, *25*(6), 2853--2880. https://doi.org/10.1007/s11119-024-10153-w

Gao, F., Li, B., Ren, B., Zhao, B., Liu, P., & Zhang, J. (2022). Achieve simultaneous increase in straw resources efficiency and nitrogen efficiency under crop yield stabilization -- A case study of NCP in China for up to 8 years. *Field Crops Research*, *278*, 108431. https://doi.org/10.1016/j.fcr.2022.108431

[]{#bm_ref_gerber_2024 .anchor}Gerber, J. S., Ray, D. K., Makowski, D., Butler, E. E., Mueller, N. D., West, P. C., et al. (2024). Global spatially explicit yield gap time trends reveal regions at risk of future crop yield stagnation. *Nature Food*, *5*(2), 125--135. https://doi.org/10.1038/s43016-023-00913-8

Guo, J., Wang, Y., Fan, T., Chen, X., & Cui, Z. (2016). Designing Corn Management Strategies for High Yield and High Nitrogen Use Efficiency. *Agronomy Journal*, *108*(2), 922--929. https://doi.org/10.2134/agronj2015.0435

[]{#bm_ref_holzworth_2014 .anchor}Holzworth, D. P., Huth, N. I., deVoil, P. G., Zurcher, E. J., Herrmann, N. I., McLean, G., et al. (2014). APSIM -- Evolution towards a new generation of agricultural systems simulation. *Environmental Modelling & Software*, *62*, 327--350. https://doi.org/10.1016/j.envsoft.2014.07.009

[]{#bm_ref_hu_2023 .anchor}Hu, T., Zhang, X., Bohrer, G., Liu, Y., Zhou, Y., Martin, J., et al. (2023). Crop yield prediction via explainable AI and interpretable machine learning: Dangers of black box models for evaluating climate change impacts on crop yield. *Agricultural and Forest Meteorology*, *336*, 109458. https://doi.org/10.1016/j.agrformet.2023.109458

[]{#bm_ref_jeong_2016 .anchor}Jeong, J. H., Resop, J. P., Mueller, N. D., Fleisher, D. H., Yun, K., Butler, E. E., et al. (2016). Random Forests for Global and Regional Crop Yield Predictions. *PLOS ONE*, *11*(6), e0156571. https://doi.org/10.1371/journal.pone.0156571

[]{#bm_ref_jones_2003 .anchor}Jones, J. W., Hoogenboom, G., Porter, C. H., Boote, K. J., Batchelor, W. D., Hunt, L. A., et al. (2003). The DSSAT cropping system model. *European Journal of Agronomy*, *18*(3), 235--265. https://doi.org/10.1016/S1161-0301(02)00107-7

[]{#bm_ref_keating_2003 .anchor}Keating, B. A., Carberry, P. S., Hammer, G. L., Probert, M. E., Robertson, M. J., Holzworth, D., et al. (2003). An overview of APSIM, a model designed for farming systems simulation. *European Journal of Agronomy*, *18*(3), 267--288. https://doi.org/10.1016/S1161-0301(02)00108-9

[]{#bm_ref_khaki_wang_2019 .anchor}Khaki, S., & Wang, L. (2019). Crop Yield Prediction Using Deep Neural Networks. *Frontiers in Plant Science*, *10*. https://doi.org/10.3389/fpls.2019.00621

[]{#bm_ref_liang_2011 .anchor}Liang, W., Carberry, P., Wang, G., Lü, R., Lü, H., & Xia, A. (2011). Quantifying the yield gap in wheat--maize cropping systems of the Hebei Plain, China. *Field Crops Research*, *124*(2), 180--185. https://doi.org/10.1016/j.fcr.2011.07.010

[]{#bm_ref_liu_by_2021 .anchor}Liu, B.-Y., Lin, B.-J., Li, X.-X., Virk, A. L., N'dri Yves, B., Zhao, X., et al. (2021). Appropriate farming practices of summer maize in the North China Plain: Reducing nitrogen use to promote sustainable agricultural development. *Resources, Conservation and Recycling*, *175*, 105889. https://doi.org/10.1016/j.resconrec.2021.105889

[]{#bm_ref_liu_2026 .anchor}Liu, D., Pan, B., Gong, H., Li, J., Wang, E., Zhao, J., et al. (2026). Optimising crop calendars with management practices promotes climate-smart agriculture in wheat-maize rotations of the North China Plain. *Agricultural Systems*, *233*, 104626. https://doi.org/10.1016/j.agsy.2025.104626

[]{#bm_ref_liu_w_2021 .anchor}Liu, W., Ye, T., & Shi, P. (2021). Decreasing wheat yield stability on the North China Plain: Relative contributions from climate change in mean and variability. *International Journal of Climatology*, *41*(S1). https://doi.org/10.1002/joc.6882

[]{#bm_ref_luo_2023 .anchor}Luo, N., Meng, Q., Feng, P., Qu, Z., Yu, Y., Liu, D. L., et al. (2023). China can be self-sufficient in maize production by 2030 with optimal crop management. *Nature Communications*, *14*(1), 2637. https://doi.org/10.1038/s41467-023-38355-2

Madias, A., Simón, C. G., Stahringer, N. I., Borrás, L., Rubio, G., & Gambin, B. L. (2025). On-farm insights in the South American Gran Chaco reveal the importance of soil organic matter and crop management decisions for boosting maize yields. *European Journal of Agronomy*, *168*, 127612. https://doi.org/10.1016/j.eja.2025.127612

[]{#bm_ref_maseko_2024 .anchor}Maseko, S., van der Laan, M., Tesfamariam, E. H., Delport, M., & Otterman, H. (2024). Evaluating machine learning models and identifying key factors influencing spatial maize yield predictions in data intensive farm management. *European Journal of Agronomy*, *157*, 127193. https://doi.org/10.1016/j.eja.2024.127193

[]{#bm_ref_meng_2013 .anchor}Meng, Q., Hou, P., Wu, L., Chen, X., Cui, Z., & Zhang, F. (2013). Understanding production potentials and yield gaps in intensive maize production in China. *Field Crops Research*, *143*, 91--97. https://doi.org/10.1016/j.fcr.2012.09.023

[]{#bm_ref_mo_2017 .anchor}Mo, X.-G., Hu, S., Lin, Z.-H., Liu, S.-X., & Xia, J. (2017). Impacts of climate change on agricultural water resources and adaptation on the North China Plain. *Advances in Climate Change Research*, *8*(2), 93--98. https://doi.org/10.1016/j.accre.2017.05.007

[]{#bm_ref_nyeki_2021 .anchor}Nyéki, A., Kerepesi, C., Daróczy, B., Benczúr, A., Milics, G., Nagy, J., et al. (2021). Application of spatio-temporal data in site-specific maize yield prediction with machine learning methods. *Precision Agriculture*, *22*(5), 1397--1415. https://doi.org/10.1007/s11119-021-09833-8

Pasquel, D., Roux, S., Richetti, J., Cammarano, D., Tisseyre, B., & Taylor, J. A. (2022). A review of methods to evaluate crop model performance at multiple and changing spatial scales. *Precision Agriculture*, *23*(4), 1489--1513. https://doi.org/10.1007/s11119-022-09885-4

[]{#bm_ref_paudel_2021 .anchor}Paudel, D., Boogaard, H., de Wit, A., Janssen, S., Osinga, S., Pylianidis, C., & Athanasiadis, I. N. (2021). Machine learning for large-scale crop yield forecasting. *Agricultural Systems*, *187*, 103016. https://doi.org/10.1016/j.agsy.2020.103016

Ren, C., He, L., & Rosa, L. (2025). Integrated irrigation and nitrogen optimization is a resource-efficient adaptation strategy for US maize and soybean production. *Nature Food*, *6*(4), 389--400. https://doi.org/10.1038/s43016-024-01107-6

Shahhosseini, M., Hu, G., Huber, I., & Archontoulis, S. V. (2021). Coupling machine learning and crop modeling improves crop yield prediction in the US Corn Belt. *Scientific Reports*, *11*(1), 1606. https://doi.org/10.1038/s41598-020-80820-1

[]{#bm_ref_shahhosseini_2019 .anchor}Shahhosseini, M., Martinez-Feria, R. A., Hu, G., & Archontoulis, S. V. (2019). Maize Yield and Nitrate Loss Prediction with Machine Learning Algorithms. *Environmental Research Letters*, *14*(12), 124026. https://doi.org/10.1088/1748-9326/ab5268

[]{#bm_ref_smith_2026 .anchor}Smith, H. W., Heffernan, C. J., Ashworth, A. J., Nalley, L. L., Bullock, D. S., Tullis, J., & Owens, P. R. (2026). Harvesting insights: interpretable machine learning to understand environmental drivers of U.S. maize and soybean yield. *Scientific Reports*, *16*(1), 8994. https://doi.org/10.1038/s41598-026-38724-z

[]{#bm_ref_storn_price_1997 .anchor}Storn, R., & Price, K. (1997). Differential Evolution -- A Simple and Efficient Heuristic for global Optimization over Continuous Spaces. *Journal of Global Optimization*, *11*(4), 341--359. https://doi.org/10.1023/A:1008202821328

[]{#bm_ref_tanaka_2024 .anchor}Tanaka, T. S. T., Heuvelink, G. B. M., Mieno, T., & Bullock, D. S. (2024). Can machine learning models provide accurate fertilizer recommendations? *Precision Agriculture*, *25*(4), 1839--1856. https://doi.org/10.1007/s11119-024-10136-x

[]{#bm_ref_van_ittersum_2013 .anchor}van Ittersum, M. K., Cassman, K. G., Grassini, P., Wolf, J., Tittonell, P., & Hochman, Z. (2013). Yield gap analysis with local to global relevance---A review. *Field Crops Research*, *143*, 4--17. https://doi.org/10.1016/j.fcr.2012.09.009

[]{#bm_ref_van_klompenburg_2020 .anchor}Van Klompenburg, T., Kassahun, A., & Catal, C. (2020). Crop yield prediction using machine learning: A systematic literature review. *Computers and Electronics in Agriculture*, *177*, 105709. https://doi.org/10.1016/j.compag.2020.105709

[]{#bm_ref_wang_2023 .anchor}Wang, H., Ren, H., Zhang, L., Zhao, Y., Liu, Y., He, Q., et al. (2023). A sustainable approach to narrowing the summer maize yield gap experienced by smallholders in the North China Plain. *Agricultural Systems*, *204*, 103541. https://doi.org/10.1016/j.agsy.2022.103541

Wang, J., Dong, X., Qiu, R., Lou, B., Tian, L., Chen, P., et al. (2023). Optimization of sowing date and irrigation schedule of maize in different cropping systems by APSIM for realizing grain mechanical harvesting in the North China Plain. *Agricultural Water Management*, *276*, 108068. https://doi.org/10.1016/j.agwat.2022.108068

[]{#bm_ref_xiao_2024 .anchor}Xiao, L., Wang, G., Wang, E., Liu, S., Chang, J., Zhang, P., et al. (2024). Spatiotemporal co-optimization of agricultural management practices towards climate-smart crop production. *Nature Food*, *5*(1), 59--71. https://doi.org/10.1038/s43016-023-00891-x

[]{#bm_ref_yang_2015 .anchor}Yang, X., Chen, Y., Pacenka, S., Gao, W., Ma, L., Wang, G., et al. (2015). Effect of diversified crop rotations on groundwater levels and crop water productivity in the North China Plain. *Journal of Hydrology*, *522*, 428--438. https://doi.org/10.1016/j.jhydrol.2015.01.010

[]{#bm_ref_zhang_2022 .anchor}Zhang, HaiYan, Zhang, C., Sun, P., Jiang, X., Xu, G., & Yang, J. (2022). Optimizing planting density and nitrogen application to enhance profit and nitrogen use of summer maize in Huanghuaihai region of China. *Scientific Reports*, *12*(1), 2704. https://doi.org/10.1038/s41598-022-06059-0

[]{#bm_ref_zhang_2026 .anchor}Zhang, Honghang, Liang, C., Zhang, W., Shukla, M., Fang, Y., Chen, S., & Du, T. (2026). Spatial distributed management strategies for maize high-yield and high-efficiency under different production-demand scenarios in Northwest China. *Agricultural Water Management*, *325*, 110166. https://doi.org/10.1016/j.agwat.2026.110166

Zhao, G., Akhter, M. J., Kromminga, H. H., & Hoffmann, H. (2026). Simulation of weed emergence and leaf development in rice fields: An integrated machine learning and phyllochron approach. *European Journal of Agronomy*, *172*, 127846. https://doi.org/10.1016/j.eja.2025.127846

Zhao, G., Zhao, Q., Webber, H., Johnen, A., Rossi, V., & Nogueira Junior, A. F. (2024). Integrating machine learning and change detection for enhanced crop disease forecasting in rice farming: A multi-regional study. *European Journal of Agronomy*, *160*, 127317. https://doi.org/10.1016/j.eja.2024.127317

[]{#bm_ref_zou_hastie_2005 .anchor}Zou, H., & Hastie, T. (2005). Regularization and Variable Selection Via the Elastic Net. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, *67*(2), 301--320. https://doi.org/10.1111/j.1467-9868.2005.00503.x
