from copy import deepcopy
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}
W = f"{{{W_NS}}}"

INPUT = Path("ms/manuscript.docx")
OUTPUT = Path("ms/manuscript_PA_submission_ready.docx")

MEDIA_REPLACEMENTS = {
    "word/media/image3.png": Path("fig/fig3_region_management_timeline.png"),
    "word/media/image4.png": Path("fig/fig4_region_sowing_density_boxplots.png"),
    "word/media/image5.png": Path("fig/fig5_region_fertilization_boxplots.png"),
    "word/media/image6.png": Path("fig/fig6_region_irrigation_boxplots.png"),
    "word/media/image7.png": Path("fig/fig7_region_pesticide_boxplots.png"),
    "word/media/image8.png": Path("fig/fig8_region_yield_cost_profit_boxplots.png"),
    "word/media/image9.png": Path("fig/fig9_model_performance_comparison.png"),
    "word/media/image10.png": Path("fig/fig10_xgboost_feature_importance.png"),
}

FIGURE_TABLE_ANCHORS = {
    **{f"Figure {idx}": f"bm_fig_{idx}" for idx in range(1, 12)},
    "Table 1": "bm_table_1",
    "Table 2": "bm_table_2",
}

CITATION_TARGETS = [
    ("Adhikari et al. 2023", "Adhikari, K.", "bm_ref_adhikari_2023"),
    ("Baio et al. 2023", "Baio, F.", "bm_ref_baio_2023"),
    ("Breiman 2001", "Breiman,", "bm_ref_breiman_2001"),
    ("Chen and Guestrin 2016", "Chen, T.", "bm_ref_chen_guestrin_2016"),
    ("Chen et al. 2011", "Chen, X.-P.", "bm_ref_chen_2011"),
    ("Chen et al. 2014", "Chen, X.,", "bm_ref_chen_2014"),
    ("Chen et al. 2019", "Chen, G.", "bm_ref_chen_2019"),
    ("Fang et al. 2010", "Fang, Q.", "bm_ref_fang_2010"),
    ("Gerber et al. 2024", "Gerber, J.", "bm_ref_gerber_2024"),
    ("Holzworth et al. 2014", "Holzworth, D.", "bm_ref_holzworth_2014"),
    ("Hu et al. 2023", "Hu, T.", "bm_ref_hu_2023"),
    ("Jeong et al. 2016", "Jeong, J.", "bm_ref_jeong_2016"),
    ("Jones et al. 2003", "Jones, J.", "bm_ref_jones_2003"),
    ("Keating et al. 2003", "Keating, B.", "bm_ref_keating_2003"),
    ("Khaki and Wang 2019", "Khaki, S.", "bm_ref_khaki_wang_2019"),
    ("Liang et al. 2011", "Liang, W.", "bm_ref_liang_2011"),
    ("B.-Y. Liu et al. 2021", "Liu, B.-Y.", "bm_ref_liu_by_2021"),
    ("W. Liu et al. 2021", "Liu, W.", "bm_ref_liu_w_2021"),
    ("Liu et al. 2026", "Liu, D.", "bm_ref_liu_2026"),
    ("Luo et al. 2023", "Luo, N.", "bm_ref_luo_2023"),
    ("Maseko et al. 2024", "Maseko, S.", "bm_ref_maseko_2024"),
    ("Meng et al. 2013", "Meng, Q.", "bm_ref_meng_2013"),
    ("Mo et al. 2017", "Mo, X.-G.", "bm_ref_mo_2017"),
    ("Nyéki et al. 2021", "Nyéki, A.", "bm_ref_nyeki_2021"),
    ("Paudel et al. 2021", "Paudel, D.", "bm_ref_paudel_2021"),
    ("Shahhosseini et al. 2019", "Shahhosseini, M., Martinez", "bm_ref_shahhosseini_2019"),
    ("Shahhosseini et al. 2021", "Shahhosseini, M., Hu", "bm_ref_shahhosseini_2021"),
    ("Smith et al. 2026", "Smith, H.", "bm_ref_smith_2026"),
    ("Storn and Price 1997", "Storn, R.", "bm_ref_storn_price_1997"),
    ("Tanaka et al. 2024", "Tanaka, T.", "bm_ref_tanaka_2024"),
    ("van Ittersum et al. 2013", "van Ittersum, M.", "bm_ref_van_ittersum_2013"),
    ("Van Klompenburg et al. 2020", "Van Klompenburg, T.", "bm_ref_van_klompenburg_2020"),
    ("Wang et al. 2023", "Wang, H.", "bm_ref_wang_2023"),
    ("Xiao et al. 2024", "Xiao, L.", "bm_ref_xiao_2024"),
    ("Yang et al. 2015", "Yang, X.", "bm_ref_yang_2015"),
    ("Zhang et al. 2022", "Zhang, HaiYan", "bm_ref_zhang_2022"),
    ("Zhang et al. 2026", "Zhang, Honghang", "bm_ref_zhang_2026"),
    ("Zou and Hastie 2005", "Zou, H.", "bm_ref_zou_hastie_2005"),
]


STRUCTURED_ABSTRACT = [
    [
        ("Purpose: ", True),
        (
            "We quantified how much maize yield and profit could improve "
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
            "models from management, field-status, and weather variables, with "
            "yield used only as the response variable. Ordinary least squares, "
            "Elastic Net, random forest, and XGBoost models were compared. The "
            "selected model was then used as a "
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
            "A constrained baseline-to-optimization framework can translate "
            "field observations into region-specific estimates of agronomic and "
            "economic improvement potential for precision maize management.",
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
        "farmer-to-farmer management differences (Chen et al. 2019; Chen et "
        "al. 2014; Chen et al. 2011; Liang et al. 2011; Meng et al. 2013; "
        "Wang et al. 2023). At the same time, inefficient nitrogen and "
        "water management can reduce resource-use efficiency, increase "
        "environmental pressure, and weaken farm profitability (B.-Y. Liu et al. "
        "2021; W. Liu et al. 2021; Mo et al. 2017; Yang et al. 2015). The "
        "practical question for precision maize management is not only which "
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
        "Scenario analysis based on machine learning needs explicit constraints "
        "and cautious interpretation."
    ),
    (
        "Despite these advances, many yield-modelling studies still stop at "
        "model comparison or variable-importance ranking. Such outputs indicate "
        "which factors are associated with yield variation, but they do not "
        "directly quantify how controllable management variables should be "
        "combined under realistic constraints or how much improvement is "
        "possible relative to the current management baseline. The distinction "
        "is important for farmers and regional decision makers, because maize "
        "management is a season-long system involving sowing date, plant "
        "density, nutrient supply, irrigation, crop protection, harvesting, and "
        "cost structure. Yield improvement also needs economic evaluation: a "
        "high-yield scenario is not practically useful if the additional input "
        "cost offsets the yield benefit (Adhikari et al. 2023; Zhang et al. "
        "2022; Zhang et al. 2026). A transparent approach is to "
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
        "The optimized scenarios are best interpreted as model-estimated "
        "regional decision-support configurations, not as "
        "field-validated causal prescriptions for individual farms."
    ),
    (
        "The objectives were to develop a leakage-controlled and interpretable "
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
        "assessment, the study provides a practical framework for quantifying "
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
        "167.5 in Hebei and day 169.0 in Shandong (Figure 4). Large "
        "interregional differences in current production performance "
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
        "189.0, 40.5, and 55.88 kg ha\u207b\u00b9 in Hebei. The main "
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
        "irrigation, sprinkler irrigation, and drip irrigation. Compared with "
        "Hebei, Shandong already relied more strongly on drip-based systems, "
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
        "ha\u207b\u00b9). Regional differences in plant protection were expressed more "
        "through cost structure than through the "
        "number of spray events. Together, the evidence indicates that current "
        "management differences between the two regions were multidimensional, "
        "involving crop establishment, nutrient supply, irrigation strategy, and "
        "input expenditure."
    ),
    "The management baseline was reflected in production performance": (
        "The management contrasts were reflected in baseline production "
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
        "whereas Hebei maintained a lower baseline with potentially larger room "
        "for improvement."
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
        "and generalization. XGBoost was used as the core model for "
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
        "connects the baseline comparisons with the optimization results: the "
        "prominence of "
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
        "could improve relative to the current regional baseline (Figure "
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
        "representing a 7.39% increase. The optimization framework "
        "improved both productivity and profitability in both regions, while "
        "the larger relative gain in Hebei indicates greater remaining room for "
        "improvement where the current management baseline was lower."
    ),
}


DISCUSSION = [
    ("2",
        "We developed a field-level baseline-to-optimization framework for "
        "estimating how much maize yield and profit can still be improved under "
        "current farming conditions in the North China Plain. The framework "
        "combines leakage-controlled yield modelling, interpretable feature "
        "ranking, constrained maximum-yield scenario search, and "
        "post-optimization profit accounting. In doing so, it shifts the role of "
        "machine learning from yield prediction alone to opportunity estimation: "
        "how far current practice is from an empirically attainable management "
        "configuration, which management pathway accounts for that distance, and "
        "whether the predicted yield gain remains economically useful. Precision "
        "agriculture recommendations become more useful when they are "
        "baseline-aware. The same technology, model, or management package can "
        "have different value depending on the starting level of local farmer "
        "practice."
    ),
    ("Heading2", "Baseline determines regional optimization potential"),
    ("2",
        "The regional comparison shows that optimization potential is not an "
        "absolute property of a crop system; it is conditional on the current "
        "management baseline. Shandong already had higher plant density, greater "
        "nutrient input, higher yield, and higher profit. Hebei started from a "
        "lower productivity baseline and consequently showed the larger "
        "proportional yield response after optimization. The same algorithmic "
        "search has different agronomic meaning in the two regions: "
        "in Shandong it mainly refines an already intensive system, whereas in "
        "Hebei it identifies a broader management upgrade."
    ),
    ("2",
        "Baseline-dependent opportunity connects the study to modern yield-gap "
        "analysis. Yield-gap studies emphasize that the distance to an attainable frontier "
        "varies across space and time, and that regions closer to stagnation or "
        "farther from attainable production frontiers require different "
        "intervention priorities (Gerber et al. 2024; van Ittersum et al. "
        "2013). Studies in China have reached a similar conclusion for maize: "
        "farmer management heterogeneity, input allocation, and operational "
        "implementation explain a substantial share of remaining yield gaps "
        "(Chen et al. 2019; Meng et al. 2013; Wang et al. 2023). Precision "
        "agriculture should not begin with a generic prescription. It should "
        "begin with a diagnosis of where the "
        "field or region sits relative to its current attainable frontier. That "
        "diagnosis determines whether decision support should emphasize fine "
        "tuning, input reallocation, or coordinated system upgrading."
    ),
    ("Heading2", "Irrigation and crop establishment are the dominant levers"),
    ("2",
        "Irrigation and crop establishment emerged as the management core of the "
        "optimization pathway. Irrigation method, plant density, sowing date, "
        "and irrigation frequency ranked among the leading predictors, and the "
        "optimized scenarios generally selected drip irrigation, earlier sowing, "
        "higher density, and stronger irrigation support. These variables are "
        "not independent levers. Plant density and sowing date define the crop "
        "stand and its seasonal demand for radiation, water, and nutrients; "
        "irrigation method and frequency determine whether that demand can be "
        "met under uneven summer rainfall."
    ),
    ("2",
        "High-yield maize systems in China depend on coordinated management "
        "rather than on increasing one "
        "input in isolation. Integrated soil-crop management and optimized crop "
        "management have been shown to increase grain production while reducing "
        "environmental cost or closing maize supply gaps (Chen et al. 2014; "
        "Chen et al. 2011; Luo et al. 2023). More recent work on "
        "climate-smart crop production also argues for co-optimization of "
        "calendar decisions and management practices rather than single-factor "
        "adjustment (Xiao et al. 2024; Liu et al. 2026). For the North China "
        "Plain, where rainfall is concentrated but unevenly distributed, water "
        "delivery reliability is a particularly important condition for turning "
        "stand-level yield capacity into harvested yield (Fang et al. 2010; Mo "
        "et al. 2017; Yang et al. 2015). Decision support should recommend "
        "management bundles, not isolated actions: earlier sowing and suitable "
        "density create yield capacity, while irrigation method and frequency "
        "decide whether that capacity can be realized."
    ),
    ("Heading2", "Yield-oriented optimization can remain economically meaningful"),
    ("2",
        "The optimized scenarios increased profit as well as yield, which is "
        "important because agronomic optimum and economic optimum are not "
        "automatically aligned. A yield-maximizing recommendation can fail at "
        "farm level if the required input cost absorbs the additional revenue. "
        "Here, input cost increased only modestly relative to yield and profit "
        "gains, so the optimized scenarios were not simply high-cost "
        "intensification. They were higher-performing management bundles in "
        "which added yield value exceeded added cost."
    ),
    ("2",
        "Profit-aware evaluation is central to precision agriculture because "
        "the unit of decision is not only a yield response, but a management "
        "choice made under cost and risk. Within-field evidence from corn "
        "systems shows that yield stability and gross margin can vary together "
        "in ways that matter for conservation and site-specific management "
        "(Adhikari et al. 2023). Studies of maize management also show that "
        "yield, water productivity, nitrogen use, and profit must be evaluated "
        "jointly when recommending density, nutrient input, or irrigation "
        "strategies (Zhang et al. 2022; Zhang et al. 2026). The present "
        "yield-first, profit-second design makes that tradeoff explicit. It "
        "allows a biological response surface to identify attainable yield "
        "scenarios, then tests whether those scenarios survive an economic "
        "screen. That separation is useful for decision support because it shows "
        "whether profit gains arise from genuine productivity improvement rather "
        "than from hidden assumptions about input cost."
    ),
    ("Heading2", "Constrained machine learning complements process-based decision support"),
    ("2",
        "The methodological value of the framework lies in constrained rather "
        "than unconstrained machine learning. Process-based crop models remain "
        "indispensable for testing mechanisms, climate sensitivity, "
        "soil-water-nitrogen processes, and genotype-by-environment interactions "
        "(Holzworth et al. 2014; Jones et al. 2003; Keating et al. 2003). "
        "However, real farmer management data contain heterogeneous irrigation "
        "devices, crop-protection operations, timing decisions, and cost "
        "structures that are difficult to parameterize completely in process "
        "models. Machine learning can learn from such observational complexity, "
        "but only if the scenario search is kept inside the support of observed "
        "management conditions."
    ),
    ("2",
        "The approach is consistent with recent agricultural systems and "
        "precision agriculture literature. Machine learning has become powerful "
        "for large-scale yield forecasting and field-level prediction (Paudel et "
        "al. 2021; Nyéki et al. 2021), but accurate prediction does not by "
        "itself justify management prescription. Fertilizer-recommendation "
        "studies in Precision Agriculture have shown that machine-learning "
        "models can be misleading when recommendations are made outside the "
        "domain where the model has reliable support (Tanaka et al. 2024). "
        "Explainable-AI studies similarly warn against treating black-box "
        "associations as causal management effects (Hu et al. 2023). The "
        "constrained search used here addresses that problem directly. It uses "
        "machine learning to prioritize management directions and estimate "
        "opportunity, while leaving mechanism testing, robustness assessment, "
        "and transferability to multi-year field experiments and process-based "
        "crop modelling."
    ),
    ("Heading2", "Limitations and future work"),
    ("2",
        "The dataset contains 81 field observations from Hebei and Shandong in "
        "one growing season, so it "
        "does not capture the full climatic, soil, cultivar, and management "
        "diversity of the North China Plain. Second, soil properties, cultivar "
        "traits, remote-sensing indicators, and multi-year weather variation "
        "were not explicitly included. Third, the optimized scenarios are "
        "model-estimated outcomes from observational data, not causal effects "
        "verified through field intervention. Although empirical bounds and the "
        "distance penalty reduce unsupported extrapolation, they cannot replace "
        "experimental validation."
    ),
    ("2",
        "Future work should test the identified management pathways across more "
        "years, sites, soil conditions, and cultivar backgrounds. The framework "
        "would also benefit from richer environmental covariates, remote-sensing "
        "indicators, and explicit uncertainty analysis for scenario predictions. "
        "A priority is to link this data-driven optimization workflow with "
        "process-based crop modelling and field experiments. That combination "
        "would clarify whether the estimated gains remain robust across weather "
        "years, soil types, and market conditions, and would move the framework "
        "from regional opportunity estimation toward validated management "
        "recommendation."
    ),
]


CONCLUSIONS = [
    (
        "Maize management improvement potential in the North China Plain was "
        "quantified using a leakage-controlled XGBoost model and a "
        "constrained baseline-to-optimization framework. Irrigation method, plant "
        "density, sowing date, and irrigation frequency were the leading "
        "management-related predictors of yield variation."
    ),
    (
        "The maximum-yield scenarios increased both yield and profit in Hebei and "
        "Shandong. Median yield increased by 27.38% in Hebei and 17.55% in "
        "Shandong, while median profit increased by 50.15% and 46.41%, "
        "respectively. The larger relative gain in Hebei indicates that regions "
        "with lower current management baselines may retain greater optimization "
        "potential."
    ),
    (
        "The framework shows how field observations can be translated into "
        "region-specific estimates of agronomic and economic opportunity. Its "
        "use should remain constrained to the support of the observed data until "
        "multi-year, multi-site, and field-experimental validation is available."
    ),
]


DECLARATIONS = [
    ("Heading1", [("Statements and Declarations", False)]),
    ("Heading2", [("Funding", False)]),
    (
        "2",
        [
            (
                "This work was funded by Longping Kaihong Agricultural Technology "
                "(Beijing) Co., Ltd. (\u9686\u5e73\u5f00\u9e3f(\u5317\u4eac)"
                "\u519c\u4e1a\u79d1\u6280\u6709\u9650\u516c\u53f8). The authors "
                "gratefully acknowledge financial support from the Key Research "
                "and Development Program of Shaanxi (Grant No. 2023-ZDLNY-64).",
                False,
            )
        ],
    ),
    ("Heading2", [("Competing Interests", False)]),
    ("2", [("The authors have no relevant financial or non-financial interests to disclose.", False)]),
    ("Heading2", [("Ethics Approval", False)]),
    (
        "2",
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
    ("2", [("Not applicable.", False)]),
    ("Heading2", [("Consent to Publish", False)]),
    ("2", [("All authors have approved the manuscript and consent to its publication.", False)]),
    ("Heading2", [("Data Availability", False)]),
    ("2", [("The data that support the findings of this study are available from the corresponding author upon reasonable request.", False)]),
    ("Heading2", [("Code Availability", False)]),
    (
        "2",
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
        "2",
        [
            (
                "Zhiming Xia, Bin Chen, Haijing Shi, Qiang Yu, and Gang Zhao contributed to the "
                "study conception and design. Zhiming Xia and Bin Chen performed "
                "the methodology development, data analysis, model construction, "
                "optimization analysis, and visualization. Qi Shen, Zeyun Liang, "
                "Ming Tian, Dengke Cao, and Yan Zhao contributed to field "
                "investigation, data collection, data curation, and resources. "
                "Zhiming Xia wrote the first draft of the manuscript. Bin Chen, "
                "Qi Shen, Zeyun Liang, Ming Tian, Dengke Cao, Yan Zhao, Haijing "
                "Shi, Qiang Yu, and Gang Zhao reviewed and edited the manuscript. "
                "Qiang Yu and Gang Zhao supervised the work and contributed to funding "
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


OPTIMIZATION_METHOD_DETAILS = [
    (
        "After model comparison, the selected yield model was used as a response "
        "function for regional management-scenario evaluation. For each field, "
        "candidate management values replaced the optimized management variables "
        "while non-optimized background variables were retained from the original "
        "field record. The optimized scenario was therefore evaluated against "
        "the observed regional production context rather than as a fully "
        "synthetic field."
    ),
    (
        "The direct decision variables were sowing day of year, plant density, "
        "N, P, and K application rates, pesticide cost, irrigation electricity "
        "use, and irrigation method. Irrigation electricity use was set to zero "
        "under the no-irrigation scenario. Fertilization frequency, pesticide "
        "application frequency, and irrigation frequency were inferred from "
        "related management intensities using monotonic isotonic-regression "
        "relationships fitted within each region: fertilization frequency from "
        "total nutrient input, pesticide frequency from pesticide cost, and "
        "irrigation frequency from irrigation electricity use under the "
        "corresponding irrigation mode."
    ),
    (
        "For each region, irrigation methods observed in that region were "
        "evaluated by enumeration. For each candidate irrigation method, "
        "continuous decision variables were searched using differential "
        "evolution. The regional objective maximized median predicted yield "
        "across fields in the region after subtracting a distance penalty based "
        "on the standardized nearest-neighbour distance between the candidate "
        "management vector and observed regional management records. This "
        "penalty reduced the influence of unsupported extrapolation."
    ),
    (
        "The feasible search space was region specific. Search bounds for "
        "continuous variables were restricted to the 5th and 95th percentiles "
        "of the observed regional distributions. Candidate values were "
        "quantized before model evaluation: sowing date to 1 d, plant density "
        "to 500 plants ha\u207b\u00b9, N, P, and K inputs to 5 kg ha\u207b\u00b9, pesticide "
        "cost to 50 CNY ha\u207b\u00b9, and irrigation electricity use to 50 kWh "
        "ha\u207b\u00b9. The differential-evolution search used a random seed of 42, "
        "120 maximum iterations, a population-size multiplier of 12, and a "
        "distance-penalty weight of 150.0."
    ),
    (
        "Optimized regional yield was reported as the unpenalized median "
        "predicted yield under the selected regional management vector. The "
        "current regional baseline was calculated directly from observed data, "
        "not from model predictions. Specifically, baseline yield, input cost, "
        "profit, and management values were calculated as regional medians of "
        "the observed field records."
    ),
    (
        "Scenario input cost was estimated using a yield-first, profit-second "
        "framework. Cost records were linked to the modelling data by field "
        "identifier. Three linear cost submodels were fitted from the observed "
        "data: sowing cost as a function of plant density, fertilization cost "
        "as a function of N, P, and K application rates, and irrigation running "
        "cost as a function of irrigation electricity use. Irrigation device "
        "cost was assigned as the median observed device cost for the candidate "
        "irrigation method."
    ),
    (
        "For each field, the observed cost not explained by scenario-dependent "
        "sowing, fertilization, pesticide, irrigation-running, and irrigation "
        "device components was treated as a fixed residual. Total scenario cost "
        "was calculated as the sum of this fixed residual and the "
        "scenario-dependent cost components. Scenario profit was then calculated "
        "from maize grain price multiplied by predicted yield minus total "
        "scenario input cost, and regional scenario profit was summarized as "
        "the median across fields within each region."
    ),
    (
        "Yield and profit gains were calculated by comparing the optimized "
        "maximum-yield scenario with the observed regional baseline. Absolute "
        "gains were computed as optimized minus baseline values, and relative "
        "gains were computed as absolute gains divided by the corresponding "
        "baseline value."
    ),
]


REGIONAL_OPTIMIZATION_METHOD_BLOCK = [
    (
        "Heading3",
        "Optimization objective and decision variables",
    ),
    (
        "2",
        "After model comparison, the selected yield model was used as a regional "
        "response function for management-scenario evaluation. For each field, "
        "candidate values replaced the optimized management variables while "
        "non-optimized background variables were retained from the original "
        "field record. The regional objective was the median predicted yield "
        "across all fields within a region, and the management combination that "
        "maximized this objective was defined as the regional maximum-yield "
        "strategy."
    ),
    (
        "2",
        "The direct decision variables were sowing day of year, plant density, "
        "N, P, and K application rates, pesticide cost, irrigation electricity "
        "use, and irrigation method. Irrigation electricity use was set to zero "
        "under the no-irrigation scenario. Fertilization frequency, pesticide "
        "application frequency, and irrigation frequency were inferred from "
        "related management intensities using monotonic isotonic-regression "
        "relationships fitted within each region. This design avoided "
        "unrealistic combinations in which input rates and operation counts "
        "moved in contradictory directions."
    ),
    (
        "Heading3",
        "Feasible search and extrapolation control",
    ),
    (
        "2",
        "Irrigation methods observed in each region were evaluated by "
        "enumeration. For each candidate irrigation method, continuous decision "
        "variables were searched using differential evolution because the "
        "objective function was nonlinear, non-smooth, and non-differentiable "
        "(Storn and Price 1997). The penalized objective subtracted a "
        "standardized nearest-neighbour distance penalty from median predicted "
        "yield to discourage multivariate management combinations far from "
        "observed regional management records."
    ),
    (
        "2",
        "The feasible search space was region specific. Search bounds for "
        "continuous variables were restricted to the 5th and 95th percentiles "
        "of the observed regional distributions. Candidate values were "
        "quantized before model evaluation: sowing date to 1 d, plant density "
        "to 500 plants ha\u207b\u00b9, N, P, and K inputs to 5 kg ha\u207b\u00b9, pesticide "
        "cost to 50 CNY ha\u207b\u00b9, and irrigation electricity use to 50 kWh "
        "ha\u207b\u00b9. The differential-evolution search used a random seed of 42, "
        "120 maximum iterations, a population-size multiplier of 12, and a "
        "distance-penalty weight of 150.0."
    ),
    (
        "Heading3",
        "Scenario cost and profit evaluation",
    ),
    (
        "2",
        "Optimized regional yield was reported as the unpenalized median "
        "predicted yield under the selected regional management vector. The "
        "current regional baseline was calculated directly from observed data, "
        "not from model predictions. Specifically, baseline yield, input cost, "
        "profit, and management values were calculated as regional medians of "
        "the observed field records."
    ),
    (
        "2",
        "Scenario input cost was estimated using a yield-first, profit-second "
        "framework. Cost records were linked to the modelling data by field "
        "identifier. Three linear cost submodels were fitted from the observed "
        "data: sowing cost as a function of plant density, fertilization cost "
        "as a function of N, P, and K application rates, and irrigation running "
        "cost as a function of irrigation electricity use. Irrigation device "
        "cost was assigned as the median observed device cost for the candidate "
        "irrigation method."
    ),
    (
        "2",
        "For each field, the observed cost not explained by scenario-dependent "
        "sowing, fertilization, pesticide, irrigation-running, and irrigation "
        "device components was treated as a fixed residual. Total scenario cost "
        "was calculated as the sum of this fixed residual and the "
        "scenario-dependent cost components. Scenario profit was calculated "
        "from maize grain price multiplied by predicted yield minus total "
        "scenario input cost. Yield and profit gains were then calculated by "
        "comparing the optimized maximum-yield scenario with the observed "
        "regional baseline."
    ),
]


FIELD_MEASUREMENT_BLOCK = [
    (
        "At maize maturity, grain yield was estimated from field sampling. Grain "
        "yield (Y, kg ha\u207b\u00b9) was calculated as:"
    ),
    ("Y = HEN \u00d7 KNE \u00d7 TKW / 1,000,000"),
    (
        "where HEN is harvested ear number per hectare, KNE is average kernel "
        "number per ear, and TKW is 1000-kernel weight (g). The number of "
        "sampling points was determined according to field size and within-field "
        "growth variation. Three sampling points were randomly selected for "
        "fields smaller than 0.67 ha with relatively small growth variation, "
        "five sampling points were selected for fields larger than 0.67 ha or "
        "with relatively large growth variation, and nine sampling points were "
        "selected for fields of 6.67 ha or larger."
    ),
    (
        "At each sampling point, harvested ear number per hectare was calculated "
        "as:"
    ),
    ("HEN = 10,000 / (PS \u00d7 RS)"),
    (
        "where PS and RS are plant spacing and row spacing (m), respectively. "
        "Average kernel number per ear was estimated from 20 consecutive ears, "
        "and 1000-kernel weight was measured after air-drying the grains to 14% "
        "moisture content."
    ),
    (
        "All agronomic and economic variables used in the analysis were "
        "expressed on a per-hectare basis. Total input cost was calculated as "
        "the sum of sowing cost, irrigation equipment cost, irrigation operating "
        "cost, pesticide cost, fertilization cost, and land rent. Profit was "
        "calculated after yield prediction rather than being used as a model "
        "input:"
    ),
    ("Profit = p \u00d7 Y - C"),
    (
        "where Profit is net return (CNY ha\u207b\u00b9), p is maize grain price "
        "(CNY kg\u207b\u00b9), Y is grain yield (kg ha\u207b\u00b9), and C is total input "
        "cost (CNY ha\u207b\u00b9). In this study, p was set to 2.4 CNY kg\u207b\u00b9. A "
        "detailed description of the variables used for yield modelling is "
        "provided in Table 1."
    ),
]


MODEL_EVALUATION_BLOCK = [
    (
        "The dataset was randomly divided into a training set (80%) and an "
        "independent validation set (20%) using a fixed random seed of 42. The "
        "same split was used for all algorithms. Each model was trained with the "
        "optimal hyperparameter settings selected before the final comparison, "
        "and fivefold cross-validation within the training set was used to "
        "estimate training-set predictive stability using negative RMSE as the "
        "scoring criterion. Model performance was evaluated on both the training "
        "and validation sets using the coefficient of determination (R\u00b2) and "
        "root mean square error (RMSE):"
    ),
    ("R\u00b2 = 1 - sum((y_i - yhat_i)^2) / sum((y_i - ybar)^2)"),
    ("RMSE = sqrt(sum((y_i - yhat_i)^2) / n)"),
    (
        "where y_i and yhat_i are the observed and predicted yields for "
        "observation i, respectively, ybar is the mean observed yield, and n is "
        "the number of observations in the evaluated split. The final model for "
        "scenario analysis was selected primarily according to validation "
        "performance, while also considering the gap between training and "
        "validation R\u00b2."
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
        append_run(p, text, bold=bold)


def append_run(parent, text, bold=False, hyperlink=False, superscript=False):
    r = etree.SubElement(parent, W + "r")
    if bold or hyperlink or superscript:
        rpr = etree.SubElement(r, W + "rPr")
        if bold:
            etree.SubElement(rpr, W + "b")
        if hyperlink:
            color = etree.SubElement(rpr, W + "color")
            color.set(W + "val", "000000")
            u = etree.SubElement(rpr, W + "u")
            u.set(W + "val", "none")
        if superscript:
            vert = etree.SubElement(rpr, W + "vertAlign")
            vert.set(W + "val", "superscript")
    t = etree.SubElement(r, W + "t")
    if text[:1].isspace() or text[-1:].isspace():
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text


def set_author_line(p):
    clear_runs(p)
    parts = [
        ("Zhiming Xia", "1", ", "),
        ("Bin Chen", "1", ", "),
        ("Qi Shen", "2", ", "),
        ("Zeyun Liang", "2", ", "),
        ("Ming Tian", "2", ", "),
        ("Dengke Cao", "3", ", "),
        ("Yan Zhao", "4", ", "),
        ("Haijing Shi", "1", ", "),
        ("Qiang Yu", "1", ", "),
        ("Gang Zhao", "1,*", ""),
    ]
    for name, marker, suffix in parts:
        append_run(p, name)
        append_run(p, marker, superscript=True)
        if suffix:
            append_run(p, suffix)


def append_hyperlink(p, text, anchor):
    link = etree.SubElement(p, W + "hyperlink")
    link.set(W + "anchor", anchor)
    link.set(W + "history", "1")
    append_run(link, text, hyperlink=True)


def max_bookmark_id(root):
    values = root.xpath(".//w:bookmarkStart/@w:id", namespaces=NS)
    return max([int(v) for v in values if v.isdigit()], default=0)


def add_bookmark(p, name, bookmark_id):
    start = etree.Element(W + "bookmarkStart")
    start.set(W + "id", str(bookmark_id))
    start.set(W + "name", name)
    end = etree.Element(W + "bookmarkEnd")
    end.set(W + "id", str(bookmark_id))
    insert_at = 1 if len(p) and p[0].tag == W + "pPr" else 0
    p.insert(insert_at, start)
    p.append(end)


def add_internal_links(body, root):
    next_id = max_bookmark_id(root) + 1
    linked_terms = {}

    for p in list(body):
        if p.tag != W + "p":
            continue
        text = paragraph_text(p)
        for idx in range(1, 12):
            if text.startswith(f"Figure {idx}."):
                add_bookmark(p, FIGURE_TABLE_ANCHORS[f"Figure {idx}"], next_id)
                next_id += 1
        for idx in range(1, 3):
            if text.startswith(f"Table {idx}."):
                add_bookmark(p, FIGURE_TABLE_ANCHORS[f"Table {idx}"], next_id)
                next_id += 1

    for citation, prefix, anchor in CITATION_TARGETS:
        for p in list(body):
            if p.tag != W + "p":
                continue
            if paragraph_text(p).startswith(prefix):
                add_bookmark(p, anchor, next_id)
                linked_terms[citation] = anchor
                next_id += 1
                break

    linked_terms.update(FIGURE_TABLE_ANCHORS)
    terms = sorted(linked_terms, key=len, reverse=True)
    matcher = re.compile("|".join(re.escape(term) for term in terms))

    in_references = False
    for p in list(body):
        if p.tag != W + "p":
            continue
        text = paragraph_text(p)
        if text == "References":
            in_references = True
            continue
        if in_references or text.startswith(("Figure ", "Table ")):
            continue
        matches = list(matcher.finditer(text))
        if not matches:
            continue
        clear_runs(p)
        last = 0
        for match in matches:
            if match.start() > last:
                append_run(p, text[last:match.start()])
            term = match.group(0)
            append_hyperlink(p, term, linked_terms[term])
            last = match.end()
        if last < len(text):
            append_run(p, text[last:])


def restore_study_area_geography(body):
    study_area_text = (
        "The North China Plain (NCP), located approximately between 32°-40°N "
        "and 114°-121°E, is one of the most important grain-producing regions "
        "in China and contributes substantially to national food production "
        "(Fang et al. 2010; W. Liu et al. 2021). Cropping systems in this "
        "region are highly intensive, and the winter wheat-summer maize "
        "rotation is widely practiced across the plain (Fang et al. 2010). The "
        "NCP is a low-relief alluvial plain with an average elevation of "
        "approximately 20 m above sea level. It is characterized by a "
        "warm-temperate monsoon climate, with mean annual temperatures ranging "
        "from 8 to 15°C and annual precipitation of approximately 500-900 mm. "
        "Precipitation is unevenly distributed throughout the year, with most "
        "rainfall concentrated in summer. Although this seasonal pattern "
        "generally supports summer maize growth, irrigation remains important "
        "for stabilizing crop water supply and reducing the risk of drought "
        "stress during critical growth stages (Fang et al. 2010; Mo et al. "
        "2017; Yang et al. 2015)."
    )
    for p in list(body):
        if p.tag == W + "p" and paragraph_text(p).startswith("The North China Plain (NCP), located approximately"):
            set_paragraph_text(p, [(study_area_text, False)])
            break


def set_title_text(p):
    clear_runs(p)
    set_alignment(p, "left")
    parts = [
        "Quantifying Yield and Profit Improvement Potential of Maize ",
        "through Management Optimization under Current Farming ",
        "Conditions in the North China Plain of China",
    ]
    for idx, text in enumerate(parts):
        r = etree.SubElement(p, W + "r")
        rpr = etree.SubElement(r, W + "rPr")
        etree.SubElement(rpr, W + "b")
        sz = etree.SubElement(rpr, W + "sz")
        sz.set(W + "val", "28")
        t = etree.SubElement(r, W + "t")
        if text[:1].isspace() or text[-1:].isspace():
            t.set(f"{{{XML_NS}}}space", "preserve")
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


def first_child(parent, tag):
    child = parent.find(W + tag)
    if child is None:
        child = etree.Element(W + tag)
        parent.insert(0, child)
    return child


def set_width(el, width, width_type="dxa"):
    el.set(W + "w", str(width))
    el.set(W + "type", width_type)


def set_cell_margins(tblpr, margin=80):
    margins = tblpr.find(W + "tblCellMar")
    if margins is None:
        margins = etree.SubElement(tblpr, W + "tblCellMar")
    for side in ("top", "left", "bottom", "right"):
        elem = margins.find(W + side)
        if elem is None:
            elem = etree.SubElement(margins, W + side)
        elem.set(W + "w", str(margin))
        elem.set(W + "type", "dxa")


def set_table_borders(tblpr):
    borders = tblpr.find(W + "tblBorders")
    if borders is None:
        borders = etree.SubElement(tblpr, W + "tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = borders.find(W + side)
        if border is None:
            border = etree.SubElement(borders, W + side)
        border.set(W + "val", "single")
        border.set(W + "sz", "4")
        border.set(W + "space", "0")
        border.set(W + "color", "9A9A9A")


def set_run_size(run, half_points):
    rpr = first_child(run, "rPr")
    for tag in ("sz", "szCs"):
        elem = rpr.find(W + tag)
        if elem is None:
            elem = etree.SubElement(rpr, W + tag)
        elem.set(W + "val", str(half_points))


def set_paragraph_compact(p):
    ppr = first_child(p, "pPr")
    spacing = ppr.find(W + "spacing")
    if spacing is None:
        spacing = etree.SubElement(ppr, W + "spacing")
    spacing.set(W + "before", "0")
    spacing.set(W + "after", "0")
    spacing.set(W + "line", "210")
    spacing.set(W + "lineRule", "auto")


def mark_header_row(row):
    trpr = first_child(row, "trPr")
    if trpr.find(W + "tblHeader") is None:
        etree.SubElement(trpr, W + "tblHeader")


def normalize_table(tbl, column_widths, font_half_points=16):
    tblpr = first_child(tbl, "tblPr")
    tblw = tblpr.find(W + "tblW")
    if tblw is None:
        tblw = etree.SubElement(tblpr, W + "tblW")
    set_width(tblw, sum(column_widths))
    layout = tblpr.find(W + "tblLayout")
    if layout is None:
        layout = etree.SubElement(tblpr, W + "tblLayout")
    layout.set(W + "type", "fixed")
    jc = tblpr.find(W + "jc")
    if jc is None:
        jc = etree.SubElement(tblpr, W + "jc")
    jc.set(W + "val", "center")
    set_cell_margins(tblpr)
    set_table_borders(tblpr)

    grid = tbl.find(W + "tblGrid")
    if grid is None:
        grid = etree.Element(W + "tblGrid")
        tbl.insert(1 if tbl.find(W + "tblPr") is not None else 0, grid)
    for child in list(grid):
        grid.remove(child)
    for width in column_widths:
        col = etree.SubElement(grid, W + "gridCol")
        col.set(W + "w", str(width))

    for row_idx, row in enumerate(tbl.findall(W + "tr")):
        if row_idx == 0:
            mark_header_row(row)
        for col_idx, cell in enumerate(row.findall(W + "tc")):
            tcpr = first_child(cell, "tcPr")
            tcw = tcpr.find(W + "tcW")
            if tcw is None:
                tcw = etree.SubElement(tcpr, W + "tcW")
            set_width(tcw, column_widths[min(col_idx, len(column_widths) - 1)])
            v_align = tcpr.find(W + "vAlign")
            if v_align is None:
                v_align = etree.SubElement(tcpr, W + "vAlign")
            v_align.set(W + "val", "center")
            for p in cell.findall(W + "p"):
                set_paragraph_compact(p)
                for run in p.findall(W + "r"):
                    set_run_size(run, font_half_points)
                    if row_idx == 0:
                        rpr = first_child(run, "rPr")
                        if rpr.find(W + "b") is None:
                            etree.SubElement(rpr, W + "b")


def normalize_table_after_caption(body, caption_prefix, column_widths, font_half_points=16):
    children = list(body)
    cap_idx = next(i for i, p in enumerate(children) if p.tag == W + "p" and paragraph_text(p).startswith(caption_prefix))
    tbl = next(children[i] for i in range(cap_idx + 1, len(children)) if children[i].tag == W + "tbl")
    normalize_table(tbl, column_widths, font_half_points)


def table_rows_as_text(tbl):
    rows = []
    replacements = {
        "Pre": "Precip",
        "Kg ha\u207b\u00b9": "kg ha\u207b\u00b9",
    }
    for row in tbl.findall(W + "tr"):
        row_text = []
        for cell in row.findall(W + "tc"):
            text = "".join(cell.xpath(".//w:t/text()", namespaces=NS)).strip()
            row_text.append(replacements.get(text, text))
        rows.append(row_text)
    return rows


def make_clean_cell(text, width, bold=False, font_half_points=15):
    tc = etree.Element(W + "tc")
    tcpr = etree.SubElement(tc, W + "tcPr")
    tcw = etree.SubElement(tcpr, W + "tcW")
    set_width(tcw, width)
    valign = etree.SubElement(tcpr, W + "vAlign")
    valign.set(W + "val", "center")

    p = etree.SubElement(tc, W + "p")
    set_paragraph_compact(p)
    ppr = p.find(W + "pPr")
    jc = ppr.find(W + "jc")
    if jc is None:
        jc = etree.SubElement(ppr, W + "jc")
    jc.set(W + "val", "left")

    r = etree.SubElement(p, W + "r")
    rpr = etree.SubElement(r, W + "rPr")
    fonts = etree.SubElement(rpr, W + "rFonts")
    fonts.set(W + "ascii", "Times New Roman")
    fonts.set(W + "hAnsi", "Times New Roman")
    fonts.set(W + "cs", "Times New Roman")
    if bold:
        etree.SubElement(rpr, W + "b")
    for tag in ("sz", "szCs"):
        size = etree.SubElement(rpr, W + tag)
        size.set(W + "val", str(font_half_points))
    t = etree.SubElement(r, W + "t")
    if text[:1].isspace() or text[-1:].isspace():
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text
    return tc


def make_clean_table(rows, column_widths, font_half_points=15):
    tbl = etree.Element(W + "tbl")
    tblpr = etree.SubElement(tbl, W + "tblPr")
    tblw = etree.SubElement(tblpr, W + "tblW")
    set_width(tblw, sum(column_widths))
    jc = etree.SubElement(tblpr, W + "jc")
    jc.set(W + "val", "center")
    layout = etree.SubElement(tblpr, W + "tblLayout")
    layout.set(W + "type", "fixed")
    set_cell_margins(tblpr)
    set_table_borders(tblpr)

    grid = etree.SubElement(tbl, W + "tblGrid")
    for width in column_widths:
        col = etree.SubElement(grid, W + "gridCol")
        col.set(W + "w", str(width))

    for row_idx, row_values in enumerate(rows):
        tr = etree.SubElement(tbl, W + "tr")
        if row_idx == 0:
            mark_header_row(tr)
        for col_idx, text in enumerate(row_values):
            width = column_widths[min(col_idx, len(column_widths) - 1)]
            tr.append(make_clean_cell(text, width, row_idx == 0, font_half_points))
    return tbl


def replace_table_after_caption_with_clean_table(body, caption_prefix, column_widths, font_half_points=15):
    children = list(body)
    cap_idx = next(i for i, p in enumerate(children) if p.tag == W + "p" and paragraph_text(p).startswith(caption_prefix))
    tbl_idx = next(i for i in range(cap_idx + 1, len(children)) if children[i].tag == W + "tbl")
    rows = table_rows_as_text(children[tbl_idx])
    body[tbl_idx] = make_clean_table(rows, column_widths, font_half_points)


def replace_range(body, start_pred, end_pred, new_paragraphs):
    children = list(body)
    start = next(i for i, p in enumerate(children) if p.tag == W + "p" and start_pred(p))
    end = next(i for i, p in enumerate(children[start + 1 :], start + 1) if p.tag == W + "p" and end_pred(p))
    body[start + 1 : end] = new_paragraphs


def replace_range_including_start(body, start_pred, end_pred, new_paragraphs):
    children = list(body)
    start = next(i for i, p in enumerate(children) if p.tag == W + "p" and start_pred(p))
    end = next(i for i, p in enumerate(children[start + 1 :], start + 1) if p.tag == W + "p" and end_pred(p))
    body[start:end] = new_paragraphs


def append_to_section(body, heading_text, next_heading_text, new_paragraphs):
    children = list(body)
    start = next(i for i, p in enumerate(children) if p.tag == W + "p" and paragraph_text(p) == heading_text)
    end = next(i for i, p in enumerate(children[start + 1 :], start + 1) if p.tag == W + "p" and paragraph_text(p) == next_heading_text)
    body[end:end] = new_paragraphs


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

        author_p = next(p for p in children if p.tag == W + "p" and paragraph_text(p).startswith("Zhiming Xia"))
        set_paragraph_text(
            author_p,
            [(
                "Zhiming Xia1, Bin Chen1, Qi Shen2, Zeyun Liang2, Ming Tian2, "
                "Dengke Cao3, Yan Zhao4, Haijing Shi1, Qiang Yu1, Gang Zhao1,*",
                False,
            )],
        )

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

        children = list(body)
        for p in children:
            if p.tag != W + "p":
                continue
            text = paragraph_text(p)
            if "T. Chen and Guestrin 2016" in text:
                set_paragraph_text(p, [(text.replace("T. Chen and Guestrin 2016", "Chen and Guestrin 2016"), False)])
            if text.startswith("The yield model was developed to predict maize grain yield"):
                set_paragraph_text(
                    p,
                    [(
                        "The yield model was developed to predict maize grain yield "
                        "from observed management, field-status, and weather "
                        "variables. The field identifier was excluded from "
                        "modelling, and yield was used only as the response "
                        "variable. Total input cost and profit were not included "
                        "as model predictors, thereby avoiding leakage from "
                        "economic accounting variables that are directly derived "
                        "from yield or input-cost calculations. Pesticide cost was "
                        "retained as a crop-protection management variable because "
                        "it describes observed management intensity rather than "
                        "the total economic outcome.",
                        False,
                    )],
                )
                break

        replace_range_including_start(
            body,
            lambda p: paragraph_text(p).startswith("At maize maturity, grain yield was estimated"),
            lambda p: paragraph_text(p).startswith("Table 1."),
            [make_paragraph("2", [(text, False)], body_template) for text in FIELD_MEASUREMENT_BLOCK],
        )

        replace_range_including_start(
            body,
            lambda p: paragraph_text(p).startswith("The dataset was randomly divided into a training set"),
            lambda p: paragraph_text(p) == "Regional management optimization and profit evaluation",
            [make_paragraph("2", [(text, False)], body_template) for text in MODEL_EVALUATION_BLOCK],
        )

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

        replace_range(
            body,
            lambda p: paragraph_text(p) == "Regional management optimization and profit evaluation",
            lambda p: paragraph_text(p) == "Results",
            [
                make_paragraph(
                    style,
                    [(text, False)],
                    heading_template if style.startswith("Heading") else body_template,
                )
                for style, text in REGIONAL_OPTIMIZATION_METHOD_BLOCK
            ],
        )

        replace_table_after_caption_with_clean_table(body, "Table 1.", [1200, 1450, 5100, 1250], 15)
        replace_table_after_caption_with_clean_table(body, "Table 2.", [2350, 1600, 1600, 1725, 1725], 15)

        for p in list(body):
            if p.tag != W + "p":
                continue
            text = paragraph_text(p)
            if "T. Chen and Guestrin 2016" in text:
                set_paragraph_text(p, [(text.replace("T. Chen and Guestrin 2016", "Chen and Guestrin 2016"), False)])

        restore_study_area_geography(body)
        add_internal_links(body, root)

        updated_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
        with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    data = updated_xml
                elif item.filename in MEDIA_REPLACEMENTS and MEDIA_REPLACEMENTS[item.filename].exists():
                    data = MEDIA_REPLACEMENTS[item.filename].read_bytes()
                else:
                    data = zin.read(item.filename)
                zout.writestr(item, data)


if __name__ == "__main__":
    main()
