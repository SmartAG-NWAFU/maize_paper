from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}
W = f"{{{W_NS}}}"

INPUT = Path("ms/manuscript.docx")
OUTPUT = Path("ms/manuscript_PA_submission_ready.docx")


STRUCTURED_ABSTRACT = [
    [
        ("Purpose: ", True),
        (
            "This study quantified how much maize yield and profit could improve "
            "through region-specific management optimization relative to current "
            "farmer management in the North China Plain of China.",
            False,
        ),
    ],
    [
        ("Methods: ", True),
        (
            "Field observations from 81 summer maize fields in Hebei and Shandong "
            "Provinces were used to build leakage-controlled yield prediction "
            "models from management, field-status, weather, yield, and input-cost "
            "variables. Ordinary least squares, Elastic Net, random forest, and "
            "XGBoost models were compared. The selected model was then used as a "
            "regional response function for constrained maximum-yield scenario "
            "optimization, with controllable management variables restricted to "
            "empirically supported ranges. Profit was evaluated after yield "
            "optimization using scenario-specific input-cost estimates.",
            False,
        ),
    ],
    [
        ("Results: ", True),
        (
            "XGBoost showed the best validation performance, with an R\u00b2 of 0.652 "
            "and an RMSE of 1168.06 kg ha\u207b\u00b9. Irrigation method, plant density, "
            "sowing date, and irrigation frequency were the leading "
            "management-related predictors. The optimized scenarios selected drip "
            "irrigation in both regions and involved earlier sowing, higher plant "
            "density, adjusted nutrient inputs, and stronger irrigation support. "
            "Median yield and profit increased by 27.38% and 50.15% in Hebei, and "
            "by 17.55% and 46.41% in Shandong, respectively.",
            False,
        ),
    ],
    [
        ("Conclusion: ", True),
        (
            "Constrained machine-learning scenario analysis can translate field "
            "observations into region-specific estimates of agronomic and economic "
            "improvement potential for precision maize management.",
            False,
        ),
    ],
]


INTRODUCTION = [
    (
        "Maize production in the North China Plain is important for regional "
        "grain supply, but further yield improvement increasingly depends on "
        "more precise and economically viable management under resource and cost "
        "constraints rather than simple input expansion. Summer maize production "
        "in this region is characterized by high input intensity, strong "
        "field-level management heterogeneity, water and nutrient constraints, "
        "and large differences in yield and profit among farms. Previous studies "
        "have shown that maize yield gaps in China and the North China Plain are "
        "closely related to crop establishment, nutrient input, irrigation, and "
        "farmer-to-farmer management differences (G. Chen et al. 2019; X. Chen "
        "et al. 2014; X.-P. Chen et al. 2011; Liang et al. 2011; Meng et al. "
        "2013; H. Wang et al. 2023). At the same time, inefficient nitrogen and "
        "water management can reduce resource-use efficiency, increase "
        "environmental pressure, and weaken farm profitability (B.-Y. Liu et al. "
        "2021; W. Liu et al. 2021; Mo et al. 2017; Yang et al. 2015). Therefore, "
        "the practical question for precision maize management is not only which "
        "factors are associated with yield variation, but how much yield and "
        "profit improvement potential remains relative to current management "
        "baselines."
    ),
    (
        "Machine learning provides useful tools for analysing yield variation in "
        "heterogeneous production data. Crop yield is affected by nonlinear and "
        "interacting effects of management, weather, soil, and field conditions, "
        "and algorithms such as random forest, gradient boosting, and XGBoost can "
        "capture complex relationships that are difficult to represent with "
        "simple linear models (Jeong et al. 2016; Khaki and Wang 2019; "
        "Shahhosseini et al. 2019, 2021; Van Klompenburg et al. 2020). "
        "Explainable machine-learning approaches can further identify important "
        "predictors and improve the interpretability of data-driven decision "
        "support (Hu et al. 2023; Smith et al. 2026). For maize and other crops, "
        "these methods have been used to predict yield and evaluate the relative "
        "contribution of sowing date, plant density, nutrient input, irrigation, "
        "weather, soil, and remote-sensing variables (Baio et al. 2023; Maseko "
        "et al. 2024; Paudel et al. 2021). However, prediction accuracy and "
        "variable importance alone do not establish causal effects, and model "
        "extrapolation outside the support of observed data can be unreliable. "
        "Scenario analysis based on machine learning therefore needs explicit "
        "constraints and cautious interpretation."
    ),
    (
        "Despite these advances, many yield-modelling studies still stop at "
        "model comparison or variable-importance ranking. Such outputs indicate "
        "which factors are associated with yield variation, but they do not "
        "directly quantify how controllable management variables should be "
        "combined under realistic constraints or how much improvement is "
        "possible relative to the current management baseline. This distinction "
        "is important for farmers and regional decision makers, because maize "
        "management is a season-long system involving sowing date, plant "
        "density, nutrient supply, irrigation, crop protection, harvesting, and "
        "cost structure. Yield improvement also needs economic evaluation: a "
        "high-yield scenario is not practically useful if the additional input "
        "cost offsets the yield benefit (Adhikari et al. 2023; HaiYan Zhang et "
        "al. 2022; Honghang Zhang et al. 2026). A transparent approach is to "
        "estimate agronomic improvement potential under a constrained "
        "maximum-yield strategy first and then evaluate whether the "
        "yield-oriented scenario also improves profit."
    ),
    (
        "The North China Plain provides an appropriate setting for this "
        "baseline-dependent analysis because field-level maize management varies "
        "substantially across farms and sampled provinces. In the present study, "
        "field observations from Hebei and Shandong were used to integrate crop "
        "establishment, nutrient input, irrigation, crop protection, weather, "
        "yield, and cost information. A leakage-controlled yield model was "
        "developed using management and environmental variables while excluding "
        "cost and profit variables that are directly derived from yield-cost "
        "accounting. The trained model was then used as a response function in a "
        "constrained maximum-yield scenario search, where only controllable "
        "management variables were changed and candidate solutions were kept "
        "within empirically supported regional ranges. Profit was evaluated "
        "after yield optimization using scenario-specific input-cost changes. "
        "The optimized scenarios should therefore be interpreted as "
        "model-estimated regional decision-support configurations, not as "
        "field-validated causal prescriptions for individual farms."
    ),
    (
        "This study aimed to develop a leakage-controlled and interpretable "
        "machine-learning model for maize yield prediction, identify the key "
        "management and environmental predictors associated with yield variation, "
        "quantify model-estimated yield improvement potential under constrained "
        "maximum-yield scenarios relative to current management baselines, and "
        "evaluate whether the yield-oriented optimization scenarios also improve "
        "profit after accounting for input-cost changes. It was expected that "
        "irrigation-related variables, plant density, sowing date, and nutrient "
        "input would be major predictors of maize yield variation; that "
        "constrained maximum-yield scenarios would increase predicted yield "
        "relative to current baseline management; and that regions with lower "
        "current management baselines would show greater relative optimization "
        "potential. By linking current management baselines with constrained "
        "maximum-yield scenario optimization and post-optimization profit "
        "assessment, this study provides a practical framework for quantifying "
        "model-estimated management improvement potential in maize production in "
        "the North China Plain."
    ),
]


RESULT_REPLACEMENTS = {
    "Current management differed consistently between Hebei and Shandong, providing the baseline against which regional optimization potential was evaluated.": (
        "Current management differed systematically between Hebei and Shandong, "
        "and these regional contrasts provided the empirical basis for "
        "understanding subsequent optimization potential. Sowing date was broadly "
        "comparable between the two regions, with median sowing dates of day "
        "167.5 in Hebei and day 169.0 in Shandong (Figure 4). This indicates "
        "that large interregional differences in current production performance "
        "were unlikely to be explained by planting time alone. By contrast, "
        "plant density differed more clearly between regions. Based on the "
        "source dataset, Shandong had a substantially higher median density than "
        "Hebei (67,005 vs. 55,275 plants ha\u207b\u00b9), suggesting that Shandong "
        "generally operated under a more intensive crop establishment strategy."
    ),
    "This regional contrast extended to nutrient management": (
        "The regional contrast observed in crop establishment was accompanied by "
        "clear differences in nutrient management (Figure 5). The median "
        "fertilization frequency was 1 event in both regions, indicating that "
        "farmers in Hebei and Shandong did not differ greatly in the number of "
        "fertilization operations. However, nutrient application intensity was "
        "consistently higher in Shandong. Median N, P, and K application rates "
        "were 217.5, 75.0, and 85.93 kg ha\u207b\u00b9 in Shandong, compared with "
        "189.0, 40.5, and 55.88 kg ha\u207b\u00b9 in Hebei. Therefore, the main "
        "regional difference in fertilization was not the frequency of "
        "application, but the amount of nutrient supplied per unit area. "
        "Together with the density pattern shown in Figure 4, these results "
        "indicate that Shandong had already reached a relatively higher baseline "
        "level of agronomic input intensity."
    ),
    "Irrigation management showed a different form of regional contrast": (
        "Regional differences were also evident in irrigation management, which "
        "adds an important water-regulation dimension to the contrast in current "
        "management intensity (Figure 6). The median irrigation frequency was 1 "
        "event in both regions, but Hebei exhibited a slightly higher mean "
        "irrigation frequency than Shandong (1.27 vs. 0.95) and a higher median "
        "irrigation electricity input (329.33 vs. 300.00 kWh ha\u207b\u00b9). More "
        "importantly, irrigation mode distribution differed substantially. Drip "
        "irrigation dominated in Shandong, accounting for 32 of the 55 sampled "
        "fields, whereas Hebei showed a more even distribution among flood "
        "irrigation, sprinkler irrigation, and drip irrigation. This suggests "
        "that Shandong already relied more strongly on drip-based systems, "
        "whereas Hebei retained greater heterogeneity in water management. The "
        "irrigation contrast directly links current management differences to "
        "the yield-driving factors examined later in the modelling analysis."
    ),
    "By comparison, pesticide management showed less regional divergence": (
        "Unlike fertilization and irrigation, pesticide management showed only "
        "limited regional divergence in operational frequency, but a clearer "
        "difference in expenditure level (Figure 7). The median number of "
        "pesticide applications was 2 in both Hebei and Shandong, indicating "
        "similar field operation frequency. However, median pesticide cost in "
        "Hebei was markedly higher than in Shandong (530.92 vs. 255.00 CNY "
        "ha\u207b\u00b9). This pattern suggests that regional differences in plant "
        "protection were expressed more through cost structure than through the "
        "number of spray events. Together, the evidence indicates that current "
        "management differences between the two regions were multidimensional, "
        "involving crop establishment, nutrient supply, irrigation strategy, and "
        "input expenditure."
    ),
    "The management baseline was reflected in production performance": (
        "These management contrasts were reflected in baseline production "
        "performance (Figure 8). Shandong had a higher median yield than Hebei "
        "under current management (10,721.85 vs. 9,585.00 kg ha\u207b\u00b9), while "
        "median production cost was similar between regions (12,150.00 vs. "
        "12,171.68 CNY ha\u207b\u00b9). Because the cost difference was small but the "
        "yield gap was substantial, Shandong also achieved a higher median "
        "baseline profit than Hebei (12,031.56 vs. 10,764.39 CNY ha\u207b\u00b9). The "
        "joint distribution in the cost-yield space further indicates that "
        "Shandong generally occupied a more favorable production position under "
        "current conditions. Taken together, the baseline results show a "
        "consistent pattern: compared with Hebei, Shandong had a higher baseline "
        "management intensity and already achieved higher yield and profit, "
        "whereas Hebei maintained a lower baseline and therefore potentially "
        "larger room for improvement."
    ),
    "The four candidate algorithms differed clearly in predictive performance": (
        "Because the scenario analysis depends on the credibility of the yield "
        "prediction model, candidate algorithms were compared before scenario "
        "optimization (Figure 9). XGBoost achieved the best overall validation "
        "performance, with a validation R\u00b2 of 0.652, RMSE of 1,168.06 kg "
        "ha\u207b\u00b9, and MAE of 932.71 kg ha\u207b\u00b9. Random forest ranked second, "
        "with a validation R\u00b2 of 0.608 and RMSE of 1,238.41 kg ha\u207b\u00b9, whereas "
        "linear regression and Elastic Net performed less well, especially in "
        "terms of validation error. Beyond predictive accuracy, the "
        "train-validation gap also supported the selection of XGBoost as the "
        "final model. The difference between training and validation R\u00b2 was "
        "0.077 for XGBoost, smaller than the corresponding gap for random forest "
        "(0.135), suggesting a more favorable balance between fitting ability "
        "and generalization. Therefore, XGBoost was used as the core model for "
        "the subsequent explainability analysis and scenario optimization."
    ),
    "The selected XGBoost model showed that yield variation was jointly associated": (
        "The selected XGBoost model showed that yield variation was jointly "
        "influenced by management and environmental factors, but the leading "
        "predictors were dominated by management-related variables (Figure 10). "
        "Among the individual predictors, drip irrigation had the highest "
        "importance, followed by plant density, sowing date, irrigation "
        "frequency, and pesticide application frequency. Weather variables such "
        "as wind speed, relative humidity, radiation, and precipitation also "
        "ranked among the top predictors, indicating that environmental "
        "background still shaped yield responses even when management "
        "information was explicitly included in the model. The feature ranking "
        "provides an important bridge between the baseline comparisons and the "
        "optimization results. Specifically, the prominence of "
        "irrigation-related variables, plant density, and sowing date suggests "
        "that regional differences in water management and crop establishment "
        "were not merely descriptive contrasts, but were also closely associated "
        "with yield outcomes. Nutrient variables, particularly phosphorus and "
        "nitrogen inputs, also contributed to model prediction, although their "
        "individual importance was lower than that of the leading irrigation and "
        "establishment variables."
    ),
    "The optimized strategy for Hebei was characterized": (
        "The direction of optimized management adjustment was consistent with "
        "the feature-importance results (Table 2). In Hebei, the optimized "
        "strategy involved earlier sowing (DOY 164 vs. 167.5), higher plant "
        "density (64,000 vs. 55,275 plants ha\u207b\u00b9), increased N, P, and K "
        "inputs, higher irrigation frequency, stronger irrigation electricity "
        "input, and a shift toward drip irrigation. In Shandong, the optimized "
        "strategy also involved earlier sowing (DOY 163 vs. 169), a slight "
        "increase in density (68,000 vs. 67,005 plants ha\u207b\u00b9), increased "
        "phosphorus input, more frequent irrigation, higher irrigation "
        "electricity input, and continued reliance on drip irrigation. The "
        "magnitude of adjustment was larger in Hebei, which is consistent with "
        "its lower initial management baseline and larger optimization gain."
    ),
    "The scenario analysis showed that both regions retained substantial": (
        "Guided by the predictive model and the ranking of key management "
        "drivers, the scenario analysis quantified how much yield and profit "
        "could be improved relative to the current regional baseline (Figure "
        "11). In Hebei, the optimized scenario increased median yield from "
        "9,585.00 to 12,209.45 kg ha\u207b\u00b9, corresponding to a gain of 2,624.45 "
        "kg ha\u207b\u00b9 or 27.38%. Over the same comparison, median profit increased "
        "from 10,764.39 to 16,163.22 CNY ha\u207b\u00b9, corresponding to a gain of "
        "5,398.83 CNY ha\u207b\u00b9 or 50.15%. Median input cost also increased, from "
        "12,171.68 to 13,200.61 CNY ha\u207b\u00b9, but the cost increase (8.45%) was "
        "much smaller than the gains in yield and profit. Shandong also showed "
        "positive optimization effects, although the relative gain in yield was "
        "smaller than in Hebei. Under the optimized scenario, median yield "
        "increased from 10,721.85 to 12,603.64 kg ha\u207b\u00b9, a gain of 1,881.79 kg "
        "ha\u207b\u00b9 or 17.55%. Median profit increased from 12,031.56 to 17,615.88 "
        "CNY ha\u207b\u00b9, corresponding to a gain of 5,584.32 CNY ha\u207b\u00b9 or 46.41%. "
        "Median input cost increased from 12,150.00 to 13,047.58 CNY ha\u207b\u00b9, "
        "representing a 7.39% increase. Thus, the optimization framework "
        "improved both productivity and profitability in both regions, while "
        "the larger relative gain in Hebei indicates greater remaining room for "
        "improvement where the current management baseline was lower."
    ),
}


DISCUSSION = [
    ("Heading2", "Baseline-dependent optimization potential"),
    ("2",
        "This study developed a field-level baseline-to-optimization framework "
        "to estimate how much maize yield and profit could be improved through "
        "management adjustment under current production conditions in the North "
        "China Plain. The main finding was that substantial improvement "
        "potential remained in both sampled regions, but the magnitude of that "
        "potential depended strongly on the starting baseline. Shandong already "
        "had higher planting density, nutrient inputs, yield, and profit, "
        "whereas Hebei started from a lower baseline and showed larger "
        "proportional gains after optimization. This pattern is consistent with "
        "yield-gap theory and evidence that farms farther from attainable "
        "management frontiers often retain larger relative improvement "
        "potential (Gerber et al. 2024; van Ittersum et al. 2013). It also "
        "agrees with studies showing that maize yield gaps in China are closely "
        "linked to farmer management heterogeneity and inefficient input "
        "allocation (G. Chen et al. 2019; Meng et al. 2013; H. Wang et al. "
        "2023). The regional contrast therefore indicates that optimization "
        "should not be viewed as a single best practice applied uniformly across "
        "the North China Plain. Instead, the value of each management adjustment "
        "depends on the current production baseline, the local input structure, "
        "and the distance from an attainable management frontier."
    ),
    ("Heading2", "Management levers and agronomic meaning"),
    ("2",
        "The results identify irrigation and crop establishment as the central "
        "management levers connecting regional baseline differences, yield "
        "prediction, and optimized scenarios. Drip irrigation, plant density, "
        "sowing date, and irrigation frequency were among the most influential "
        "predictors, and the optimized scenarios generally selected earlier "
        "sowing, higher density, and stronger irrigation support. This is "
        "agronomically plausible for the North China Plain, where summer "
        "rainfall can support maize growth but uneven precipitation and water "
        "constraints make irrigation timing and method important for yield "
        "stability (Fang et al. 2010; Mo et al. 2017; Yang et al. 2015). The "
        "finding supports the broader view that maize improvement depends on "
        "coordinated adjustment of water, crop establishment, and nutrients, "
        "rather than on increasing a single input alone (X. Chen et al. 2014; "
        "X.-P. Chen et al. 2011; Luo et al. 2023; HaiYan Zhang et al. 2022). "
        "Nutrient variables were not always the highest-ranked individual "
        "predictors, but they still entered the optimized management packages. "
        "This suggests that fertilizer effects should be interpreted in relation "
        "to crop establishment and water supply rather than as isolated input "
        "responses."
    ),
    ("Heading2", "Economic and decision-support value"),
    ("2",
        "Yield-oriented optimization also improved profit in both regions because "
        "the proportional increase in input cost was much smaller than the gains "
        "in yield and profit. This result suggests that the optimized scenarios "
        "were not simply high-cost intensification, but coordinated management "
        "packages with favorable economic returns. The profit result is "
        "important because management recommendations based only on predicted "
        "yield can be misleading when additional inputs raise production costs. "
        "By separating yield modelling from post-optimization profit accounting, "
        "the framework keeps the biophysical response and economic consequence "
        "transparent. Similar model-based optimization studies have shown that "
        "maize management can improve yield, water productivity, and economic "
        "returns when multiple performance dimensions are considered together "
        "(Honghang Zhang et al. 2026). Thus, the contribution of this framework "
        "is not only prediction, but translation of prediction into "
        "region-specific estimates of agronomic and economic opportunity."
    ),
    ("2",
        "The framework also has methodological value for precision agriculture "
        "decision support. Process-based crop models remain essential for "
        "mechanistic analysis and extrapolation across weather, soil, and "
        "genotype conditions (Holzworth et al. 2014; Jones et al. 2003; Keating "
        "et al. 2003), but they cannot always represent the full observed "
        "sequence of farmer management, including heterogeneous irrigation "
        "devices, crop-protection operations, timing choices, and field-specific "
        "cost structures. The constrained machine-learning approach used here "
        "provides a complementary pathway: it uses field records to identify "
        "priority adjustment directions while restricting optimization within "
        "empirically supported management ranges. This is consistent with recent "
        "calls for interpretable and cautiously constrained machine learning in "
        "crop decision support (Hu et al. 2023; Smith et al. 2026; Tanaka et al. "
        "2024). Because the results are based on observational data, however, "
        "the optimized scenarios should be interpreted as model-estimated "
        "regional decision-support scenarios rather than field-validated causal "
        "recommendations."
    ),
    ("Heading2", "Limitations and future work"),
    ("2",
        "Several limitations should be considered. The dataset contains 81 field "
        "observations from Hebei and Shandong in one growing season, so it does "
        "not capture the full climatic, soil, cultivar, and management diversity "
        "of the North China Plain. Soil properties, cultivar traits, "
        "remote-sensing indicators, and multi-year weather variation were not "
        "explicitly included. In addition, the optimized scenarios represent "
        "model-estimated outcomes rather than causal effects verified through "
        "field interventions. Although the empirical bounds and distance penalty "
        "reduce unsupported extrapolation, they cannot replace experimental "
        "validation. Future work should test the identified management pathways "
        "across more years, sites, soil conditions, and cultivar backgrounds, "
        "combine field records with richer environmental and remote-sensing "
        "data, and evaluate optimized strategies in field trials. A further "
        "priority is to link data-driven optimization with process-based crop "
        "modelling, because the two approaches address different weaknesses and "
        "would improve confidence in whether the estimated gains remain robust "
        "across weather years, soil types, and market conditions."
    ),
]


CONCLUSIONS = [
    (
        "Using field-level observations from the North China Plain, this study "
        "developed a leakage-controlled and interpretable machine-learning "
        "framework to quantify how much maize yield and profit could be improved "
        "through management optimization under a maximum-yield strategy. The "
        "results showed that current maize production still retains substantial "
        "room for improvement in both Hebei and Shandong. Among the candidate "
        "algorithms, XGBoost provided the best predictive performance and "
        "identified irrigation-related variables, plant density, and sowing date "
        "as major predictors of yield variation."
    ),
    (
        "The scenario optimization analysis further demonstrated that management "
        "adjustment could simultaneously improve yield and profit. Under the "
        "optimized scenario, median yield increased from 9,585.00 to 12,209.45 "
        "kg ha\u207b\u00b9 in Hebei and from 10,721.85 to 12,603.64 kg ha\u207b\u00b9 in "
        "Shandong, while median profit increased from 10,764.39 to 16,163.22 CNY "
        "ha\u207b\u00b9 and from 12,031.56 to 17,615.88 CNY ha\u207b\u00b9, respectively. The "
        "larger relative gain observed in Hebei suggests that regions with a "
        "lower current management baseline may have greater optimization "
        "potential. Across regions, the main directions of improvement included "
        "earlier sowing, denser planting, strengthened nutrient input, and more "
        "effective irrigation support, particularly under drip irrigation."
    ),
    (
        "Overall, the study shows that a baseline-to-optimization framework "
        "driven by explainable machine learning can move beyond factor "
        "identification and directly quantify region-specific yield and profit "
        "gains from management improvement. This makes the approach relevant for "
        "precision agriculture decision support at the regional scale. "
        "Nevertheless, the current conclusions should be interpreted within the "
        "support of the existing dataset, and further validation across more "
        "years, locations, and field experiments is needed before broader "
        "application."
    ),
]


DECLARATIONS = [
    ("Heading1", [("Statements and Declarations", False)]),
    ("Heading2", [("Funding", False)]),
    (
        "BodyText",
        [
            (
                "This work was funded by Longping Kaihong Agricultural Technology "
                "(Beijing) Co., Ltd. (\u9686\u5e73\u5f00\u9e3f(\u5317\u4eac)"
                "\u519c\u4e1a\u79d1\u6280\u6709\u9650\u516c\u53f8). No grant number "
                "was assigned.",
                False,
            )
        ],
    ),
    ("Heading2", [("Competing Interests", False)]),
    ("BodyText", [("The authors have no relevant financial or non-financial interests to disclose.", False)]),
    ("Heading2", [("Ethics Approval", False)]),
    (
        "BodyText",
        [
            (
                "Not applicable. This study used crop-management and "
                "field-observation data and did not involve human participants or "
                "animals.",
                False,
            )
        ],
    ),
    ("Heading2", [("Consent to Participate", False)]),
    ("BodyText", [("Not applicable.", False)]),
    ("Heading2", [("Consent to Publish", False)]),
    ("BodyText", [("All authors have approved the manuscript and consent to its publication.", False)]),
    ("Heading2", [("Data Availability", False)]),
    ("BodyText", [("The data that support the findings of this study are available from the corresponding author upon reasonable request.", False)]),
    ("Heading2", [("Code Availability", False)]),
    (
        "BodyText",
        [
            (
                "The analysis code used to generate the model outputs, "
                "optimization outputs, and figures is publicly available at "
                "https://github.com/summer-zm/maize_paper.",
                False,
            )
        ],
    ),
    ("Heading2", [("Author Contributions", False)]),
    (
        "BodyText",
        [
            (
                "Zhiming Xia, Bin Chen, Qiang Yu, and Gang Zhao contributed to the "
                "study conception and design. Zhiming Xia and Bin Chen performed "
                "the methodology development, data analysis, model construction, "
                "optimization analysis, and visualization. Qi Shen, Zeyun Liang, "
                "Ming Tian, Dengke Cao, and Yan Zhao contributed to field "
                "investigation, data collection, data curation, and resources. "
                "Zhiming Xia wrote the first draft of the manuscript. Bin Chen, "
                "Qi Shen, Zeyun Liang, Ming Tian, Dengke Cao, Yan Zhao, Qiang Yu, "
                "and Gang Zhao reviewed and edited the manuscript. Qiang Yu and "
                "Gang Zhao supervised the work and contributed to funding "
                "acquisition. All authors read and approved the final manuscript.",
                False,
            )
        ],
    ),
]


HEADING_REPLACEMENTS = {
    "2. Materials and methods": "Materials and methods",
    "2.1 Study area": "Study area",
    "2.2 Data collection and variable construction": "Data collection and variable construction",
    "2.3 Yield modelling and model evaluation": "Yield modelling and model evaluation",
    "2.4 Regional management optimization and profit evaluation": "Regional management optimization and profit evaluation",
    "3. Results": "Results",
    "3.1 Regional baseline conditions": "Regional differences in current management and baseline performance",
    "3.2 Yield model performance and predictor importance": "Yield model performance and predictor importance",
    "3.3 Optimized management scenarios": "Regional gains under the maximum-yield optimization strategy",
    "4. Discussion": "Discussion",
    "4.1 Overall findings and regional potential": "Baseline-dependent optimization potential",
    "4.2 Management levers and agronomic meaning": "Management levers and agronomic meaning",
    "4.3 Decision-support value": "Economic and decision-support value",
    "4.4 Limitations and future work": "Limitations and future work",
    "5. Conclusions": "Conclusions",
}


CAPTIONS = {
    "Figure 1.": "Distribution of maize cultivation in China and location of the surveyed summer maize fields in the North China Plain. Panel (a) shows national maize planting area, the North China Plain boundary, and the regional position of the study sites; panels (b) and (c) show the field locations in Hebei and Shandong Province, respectively, on satellite basemaps. Orange shading indicates maize planting area, the green outline indicates the North China Plain, and point markers indicate the 81 surveyed fields during the 2025 summer maize season, including 26 fields in Hebei and 55 fields in Shandong. Scale bars and geographic coordinates are shown for spatial reference.",
    "Figure 2.": "Overall workflow for field data collection, yield modelling, and regional management optimization. The workflow links the Hebei and Shandong study fields, field-level records of weather, crop establishment, irrigation, fertilization, crop protection, other field-status variables, yield and cost, descriptive statistical analysis, machine-learning model comparison, XGBoost-based yield prediction, and constrained optimization simulation. Model evaluation used R\u00b2 and RMSE, and the optimization stage evaluated yield improvement, input-cost response, profit response, and region-specific management levers.",
    "Figure 3.": "Calendar distribution of major farming operations in Hebei and Shandong during the 2025 summer maize season. Each point represents one recorded field operation, colors distinguish regions, the x-axis shows operation date, and the y-axis groups management activities, including sowing, fertilization, herbicide application, insecticide application, irrigation, fungicide application, growth-regulator application, and harvest. The figure summarizes the timing window and regional overlap of field operations used to characterize current farmer management.",
    "Figure 4.": "Regional differences in crop establishment under current maize management. Panel (a) shows sowing date as day of year, and panel (b) shows plant density in plants ha\u207b\u00b9 for Hebei and Shandong. Boxplots show the median and interquartile range, whiskers show the non-outlier range, points represent individual fields, and text annotations report regional means. These variables describe the establishment baseline used in yield modelling and optimization.",
    "Figure 5.": "Regional differences in fertilization practices under current maize management. Panels show (a) number of fertilization events, (b) nitrogen application rate, (c) phosphorus application rate, and (d) potassium application rate. Nutrient rates are expressed as kg ha\u207b\u00b9. Boxplots show the median and interquartile range, whiskers show the non-outlier range, points represent individual fields, and text annotations report regional means.",
    "Figure 6.": "Regional differences in irrigation management under current maize management. Panels show (a) irrigation events, (b) irrigation electricity use in kWh ha\u207b\u00b9, and (c) the number of fields using rainfed, flood, sprinkler, or drip irrigation. Boxplots show the median and interquartile range, whiskers show the non-outlier range, points represent individual fields, and bars summarize irrigation-method counts by region.",
    "Figure 7.": "Regional differences in pesticide management under current maize management. Panel (a) shows pesticide application frequency, and panel (b) shows pesticide cost in CNY ha\u207b\u00b9. Boxplots show the median and interquartile range, whiskers show the non-outlier range, points represent individual fields, and text annotations report regional means.",
    "Figure 8.": "Regional variation in production outcomes and cost-yield relationships under current management. Panels show (a) maize grain yield in kg ha\u207b\u00b9, (b) total production cost in CNY ha\u207b\u00b9, and (c) profit in CNY ha\u207b\u00b9 for Hebei and Shandong. Panel (d) shows the joint distribution of field observations in cost-yield space, with colors indicating region and marker symbols indicating cost-yield relationship types. Dashed reference lines indicate median cost and yield, and H-L indicates high cost-low yield.",
    "Figure 9.": "Comparison of predictive performance for four candidate maize-yield models. Panel (a) shows the coefficient of determination (R\u00b2) for training and validation sets, and panel (b) shows root mean square error (RMSE, kg ha\u207b\u00b9) for the same splits. Bars compare XGBoost, random forest, linear regression, and Elastic Net, and value labels report the corresponding metric values.",
    "Figure 10.": "Relative importance of the top 10 predictors in the selected XGBoost maize-yield model. Bars show model-derived relative importance values, with larger values indicating stronger contribution to yield prediction in the fitted model. Irrigation-mode dummy variables for rainfed, flood, sprinkler, and drip irrigation were aggregated into a single irrigation-method category for display.",
    "Figure 11.": "Yield and profit gains under the maximum-yield optimization strategy in Hebei and Shandong. The baseline represents the regional median under current management, whereas the optimized scenario represents the regional maximum-yield strategy identified by the machine-learning-based optimization framework. In Hebei, the optimized scenario increased median yield from 9,585.00 to 12,209.45 kg ha\u207b\u00b9 and median profit from 10,764.39 to 16,163.22 CNY ha\u207b\u00b9. In Shandong, the optimized scenario increased median yield from 10,721.85 to 12,603.64 kg ha\u207b\u00b9 and median profit from 12,031.56 to 17,615.88 CNY ha\u207b\u00b9.",
    "Table 1.": "Variables used to construct the field-level yield-modelling dataset for the 2025 summer maize season in the North China Plain. Variables are grouped as management, weather, and yield-response variables; management and economic quantities are expressed on a per-hectare basis where applicable. Region was coded as 0 for Hebei and 1 for Shandong, and irrigation method was represented by mutually exclusive dummy variables for rainfed, flood, sprinkler, and drip irrigation. Yield was used as the response variable rather than as a predictor.",
    "Table 2.": "Baseline and optimized management configurations under the constrained maximum-yield strategy in Hebei and Shandong. Baseline values represent regional current-management medians, whereas optimal values represent the model-estimated settings selected by the optimization procedure within empirically supported management bounds.",
}


TABLE1_SUMMARY = [
    (
        "Management",
        "Region (0 = Hebei, 1 = Shandong); Sow_DOY; Density; Fer_N; Fer_P; Fer_K; "
        "Fer_Count; Pest_Count; Pest_Cost; Irr_Count; Irr_Elec; Irr_None; "
        "Irr_Flood; Irr_Sprinkler; Irr_Drip; Lodging. Management and economic "
        "variables were expressed per hectare where applicable.",
    ),
    (
        "Weather",
        "Precip, Rad, RH, and Wind, representing total precipitation, total solar "
        "radiation, average relative humidity, and maximum wind speed during the "
        "growing season.",
    ),
    (
        "Response",
        "Yield, maize grain yield per hectare (kg ha\u207b\u00b9). Yield was used as the "
        "prediction target and was not used as a model predictor.",
    ),
]


TABLE2_SUMMARY = [
    (
        "Hebei baseline",
        "Density 55,275 plants ha\u207b\u00b9; sowing date DOY 167.5; N 189 kg ha\u207b\u00b9; "
        "P 40.5 kg ha\u207b\u00b9; K 55.88 kg ha\u207b\u00b9; fertilization frequency 1; "
        "pesticide cost 530.92 CNY ha\u207b\u00b9; pesticide frequency 2; irrigation "
        "electricity use 329.33 kWh ha\u207b\u00b9; irrigation frequency 1; irrigation "
        "method flood.",
    ),
    (
        "Hebei optimized",
        "Density 64,000 plants ha\u207b\u00b9; sowing date DOY 164; N 215 kg ha\u207b\u00b9; "
        "P 55 kg ha\u207b\u00b9; K 80 kg ha\u207b\u00b9; fertilization frequency 1; pesticide "
        "cost 350 CNY ha\u207b\u00b9; pesticide frequency 1; irrigation electricity use "
        "550 kWh ha\u207b\u00b9; irrigation frequency 2; irrigation method drip.",
    ),
    (
        "Shandong baseline",
        "Density 67,005 plants ha\u207b\u00b9; sowing date DOY 169; N 217.5 kg ha\u207b\u00b9; "
        "P 75 kg ha\u207b\u00b9; K 85.93 kg ha\u207b\u00b9; fertilization frequency 1; pesticide "
        "cost 255 CNY ha\u207b\u00b9; pesticide frequency 2; irrigation electricity use "
        "300 kWh ha\u207b\u00b9; irrigation frequency 1; irrigation method drip.",
    ),
    (
        "Shandong optimized",
        "Density 68,000 plants ha\u207b\u00b9; sowing date DOY 163; N 220 kg ha\u207b\u00b9; "
        "P 80 kg ha\u207b\u00b9; K 85 kg ha\u207b\u00b9; fertilization frequency 2; pesticide "
        "cost 300 CNY ha\u207b\u00b9; pesticide frequency 2; irrigation electricity use "
        "650 kWh ha\u207b\u00b9; irrigation frequency 2; irrigation method drip.",
    ),
]


def paragraph_text(p):
    return "".join(p.xpath(".//w:t/text()", namespaces=NS)).strip()


def style_of(p):
    vals = p.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
    return vals[0] if vals else None


def clear_runs(p):
    for child in list(p):
        if child.tag != W + "pPr":
            p.remove(child)


def set_alignment(p, value):
    ppr = p.find(W + "pPr")
    if ppr is None:
        ppr = etree.SubElement(p, W + "pPr")
    jc = ppr.find(W + "jc")
    if jc is None:
        jc = etree.SubElement(ppr, W + "jc")
    jc.set(W + "val", value)


def set_paragraph_text(p, segments):
    clear_runs(p)
    for text, bold in segments:
        r = etree.SubElement(p, W + "r")
        if bold:
            rpr = etree.SubElement(r, W + "rPr")
            etree.SubElement(rpr, W + "b")
        t = etree.SubElement(r, W + "t")
        if text[:1].isspace() or text[-1:].isspace():
            t.set(f"{{{XML_NS}}}space", "preserve")
        t.text = text


def set_title_text(p):
    clear_runs(p)
    set_alignment(p, "left")
    parts = [
        "Quantifying Yield and Profit Improvement Potential of Maize",
        "through Management Optimization under Current Farming",
        "Conditions in the North China Plain of China",
    ]
    for idx, text in enumerate(parts):
        r = etree.SubElement(p, W + "r")
        rpr = etree.SubElement(r, W + "rPr")
        etree.SubElement(rpr, W + "b")
        sz = etree.SubElement(rpr, W + "sz")
        sz.set(W + "val", "28")
        t = etree.SubElement(r, W + "t")
        t.text = text
        if idx < len(parts) - 1:
            etree.SubElement(r, W + "br")


def make_paragraph(style, segments, template):
    p = etree.Element(W + "p")
    ppr = template.find(W + "pPr")
    if ppr is not None:
        p.append(deepcopy(ppr))
    if style:
        ppr = p.find(W + "pPr")
        if ppr is None:
            ppr = etree.SubElement(p, W + "pPr")
        pstyle = ppr.find(W + "pStyle")
        if pstyle is None:
            pstyle = etree.SubElement(ppr, W + "pStyle")
        pstyle.set(W + "val", style)
    set_paragraph_text(p, segments)
    return p


def make_table_cell(text, bold=False):
    tc = etree.Element(W + "tc")
    tcpr = etree.SubElement(tc, W + "tcPr")
    tcw = etree.SubElement(tcpr, W + "tcW")
    tcw.set(W + "type", "auto")
    p = etree.SubElement(tc, W + "p")
    r = etree.SubElement(p, W + "r")
    if bold:
        rpr = etree.SubElement(r, W + "rPr")
        etree.SubElement(rpr, W + "b")
    t = etree.SubElement(r, W + "t")
    t.text = text
    return tc


def make_compact_table(rows):
    tbl = etree.Element(W + "tbl")
    tblpr = etree.SubElement(tbl, W + "tblPr")
    tblw = etree.SubElement(tblpr, W + "tblW")
    tblw.set(W + "w", "5000")
    tblw.set(W + "type", "pct")
    borders = etree.SubElement(tblpr, W + "tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = etree.SubElement(borders, W + side)
        border.set(W + "val", "single")
        border.set(W + "sz", "4")
        border.set(W + "space", "0")
        border.set(W + "color", "BFBFBF")
    grid = etree.SubElement(tbl, W + "tblGrid")
    col1 = etree.SubElement(grid, W + "gridCol")
    col1.set(W + "w", "1800")
    col2 = etree.SubElement(grid, W + "gridCol")
    col2.set(W + "w", "7200")
    for row_idx, (left, right) in enumerate(rows):
        tr = etree.SubElement(tbl, W + "tr")
        tr.append(make_table_cell(left, row_idx == 0))
        tr.append(make_table_cell(right, row_idx == 0))
    return tbl


def replace_range(body, start_pred, end_pred, new_paragraphs):
    children = list(body)
    start = next(i for i, p in enumerate(children) if p.tag == W + "p" and start_pred(p))
    end = next(i for i, p in enumerate(children[start + 1 :], start + 1) if p.tag == W + "p" and end_pred(p))
    body[start + 1 : end] = new_paragraphs


def replace_table_after_caption_with_paragraphs(body, caption_prefix, rows, body_template):
    children = list(body)
    cap_idx = next(i for i, p in enumerate(children) if p.tag == W + "p" and paragraph_text(p).startswith(caption_prefix))
    tbl_idx = next(i for i in range(cap_idx + 1, len(children)) if children[i].tag == W + "tbl")
    paras = [
        make_paragraph("2", [(label + ": ", True), (text, False)], body_template)
        for label, text in rows
    ]
    body[tbl_idx : tbl_idx + 1] = paras


def main():
    with ZipFile(INPUT, "r") as zin:
        root = etree.fromstring(zin.read("word/document.xml"))
        body = root.find(".//w:body", namespaces=NS)
        children = list(body)

        heading_template = next(p for p in children if p.tag == W + "p" and style_of(p) == "Heading1")
        body_template = next(p for p in children if p.tag == W + "p" and style_of(p) == "2")
        abstract_template = next(p for p in children if p.tag == W + "p" and paragraph_text(p).startswith("Abstract:"))

        title_p = next(p for p in children if p.tag == W + "p" and paragraph_text(p).startswith("Quantifying Yield"))
        set_title_text(title_p)

        for p in children:
            if p.tag != W + "p":
                continue
            text = paragraph_text(p)
            if text in HEADING_REPLACEMENTS:
                set_paragraph_text(p, [(HEADING_REPLACEMENTS[text], False)])

        children = list(body)
        abstract_idx = next(i for i, p in enumerate(children) if p.tag == W + "p" and paragraph_text(p).startswith("Abstract:"))
        keyword_idx = next(i for i, p in enumerate(children) if p.tag == W + "p" and paragraph_text(p).startswith("Keywords:"))
        abstract_block = [make_paragraph("Heading1", [("Abstract", False)], heading_template)]
        set_alignment(abstract_block[0], "left")
        for row in STRUCTURED_ABSTRACT:
            p = make_paragraph("2", row, body_template)
            set_alignment(p, "left")
            abstract_block.append(p)
        body[abstract_idx:keyword_idx] = abstract_block

        children = list(body)
        keyword_p = next(p for p in children if p.tag == W + "p" and paragraph_text(p).startswith("Keywords:"))
        set_alignment(keyword_p, "left")
        set_paragraph_text(
            keyword_p,
            [
                ("Keywords: ", True),
                ("Maize; precision agriculture; management optimization; machine learning; yield prediction; profitability", False),
            ],
        )

        replace_range(
            body,
            lambda p: paragraph_text(p) == "Introduction",
            lambda p: paragraph_text(p) == "Materials and methods",
            [make_paragraph("2", [(text, False)], body_template) for text in INTRODUCTION],
        )

        children = list(body)
        for p in children:
            if p.tag != W + "p":
                continue
            text = paragraph_text(p)
            for prefix, replacement in RESULT_REPLACEMENTS.items():
                if text.startswith(prefix):
                    set_paragraph_text(p, [(replacement, False)])
                    break

        replace_range(
            body,
            lambda p: paragraph_text(p) == "Discussion",
            lambda p: paragraph_text(p) == "Conclusions",
            [
                make_paragraph(style, [(text, False)], heading_template if style.startswith("Heading") else body_template)
                for style, text in DISCUSSION
            ],
        )

        replace_range(
            body,
            lambda p: paragraph_text(p) == "Conclusions",
            lambda p: paragraph_text(p) == "References",
            [make_paragraph("2", [(text, False)], body_template) for text in CONCLUSIONS],
        )

        children = list(body)
        ref_idx = next(i for i, p in enumerate(children) if p.tag == W + "p" and paragraph_text(p) == "References")
        declaration_block = [
            make_paragraph(style, segments, heading_template if style.startswith("Heading") else body_template)
            for style, segments in DECLARATIONS
        ]
        body[ref_idx:ref_idx] = declaration_block

        for p in list(body):
            if p.tag != W + "p":
                continue
            text = paragraph_text(p)
            for label, caption in CAPTIONS.items():
                if text.startswith(label):
                    set_paragraph_text(p, [(label, True), (" " + caption, False)])
                    break

        replace_table_after_caption_with_paragraphs(body, "Table 1.", TABLE1_SUMMARY, body_template)
        replace_table_after_caption_with_paragraphs(body, "Table 2.", TABLE2_SUMMARY, body_template)

        updated_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
        with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = updated_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, data)


if __name__ == "__main__":
    main()
