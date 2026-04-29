# Introduction outline

本文档用于作为论文 Introduction 部分的写作结构。研究区域统一表述为 **North China Plain**（华北平原），文章主线建议围绕：

**How much maize yield and profit improvement potential can be estimated by optimizing current management under a constrained maximum-yield strategy?**

即：在当前管理基线下，在观测数据和农学合理范围约束内，通过优化可调控管理措施，玉米产量和利润还具有多大提升潜力。

---

## Recommended Structure

建议 Introduction 写成 **5 段**。每段承担一个明确功能，避免把引言写成泛泛的机器学习综述。

整体逻辑为：

**production challenge → machine-learning progress → remaining gap → study framework → objectives and hypotheses**

---

## Paragraph 1. Maize production, management improvement, and precision agriculture demand

### Main purpose

引出研究的现实背景：玉米生产对粮食安全和农业收益具有重要意义，而在华北平原这类高强度农业区，未来增产不能只依赖扩大投入，更需要在资源约束和成本上升背景下，通过精细化管理提高产量、收益和投入效率。

### Key points

- Maize is a major staple crop and an important component of grain security.
- The North China Plain is one of the most important intensive agricultural regions in China.
- Maize production in this region is characterized by high input intensity, strong management heterogeneity, water and nutrient constraints, and large differences in field-level performance.
- Under limited land, water, and input-cost constraints, yield improvement increasingly depends on better allocation and coordination of management practices rather than simply increasing inputs.
- Precision agriculture should not only identify factors associated with yield variation, but also support practical management optimization.

### Writing focus

这一段不要急着讲机器学习。重点是建立问题的重要性：为什么需要研究“当前管理还具有多大提升潜力”。

### Possible topic sentence

Maize production in the North China Plain plays an important role in regional grain supply, but further yield improvement increasingly depends on more precise and economically viable management under resource and cost constraints rather than simple input expansion.

---

## Paragraph 2. Existing progress in crop yield prediction and explainable machine learning

### Main purpose

概括已有研究进展：机器学习已经广泛用于作物产量预测和关键因子识别，为在复杂田间条件下分析管理响应和支持精准农业决策提供了技术基础。

### Key points

- Machine-learning models have been widely used to predict crop yield from management, weather, soil, and remote-sensing variables.
- Algorithms such as random forest, gradient boosting, and XGBoost can accommodate nonlinear and interacting effects among management, weather, soil, and yield variables.
- Explainable machine-learning approaches can identify important predictors and improve interpretability.
- Previous studies have shown that sowing date, plant density, nutrient input, irrigation, weather conditions, and soil properties can strongly influence maize yield.
- These methods provide useful tools for understanding yield variation under real production conditions.

### Writing focus

这一段是“承认前人工作已经很充分”。不要把文献综述写太长，也不要写成算法宣传；重点是说明已有方法可以支持复杂田间数据中的产量响应分析，并为后续情景优化提供工具基础。

### Possible topic sentence

Recent advances in machine learning have improved the ability to predict crop yield and identify the relative contributions of management and environmental factors in heterogeneous field-level production data.

---

## Paragraph 3. Research gap: from factor identification to baseline-based management optimization

### Main purpose

明确你的研究缺口。这一段是 Introduction 的核心，需要把文章与普通产量预测论文区分开。

### Key points

- Many studies focus on yield prediction accuracy or variable-importance ranking.
- However, variable importance alone does not directly answer how much yield improvement potential exists relative to current management.
- Farmers and regional decision makers need to know the improvement potential relative to their current management baseline.
- Few studies explicitly define a baseline management scenario and compare it with an optimized management scenario under realistic constraints.
- Yield improvement should also be evaluated economically; otherwise, a high-yield scenario may not be practically useful.
- Directly optimizing profit can be difficult to interpret because profit is jointly determined by yield, cost, and price; a more transparent strategy is to estimate agronomic improvement potential under a maximum-yield strategy first and then evaluate whether this yield-oriented scenario remains economically beneficial.
- Because machine-learning optimization based on observational data does not establish field-level causal effects, the expected contribution should be framed as scenario-based decision support rather than direct agronomic prescription.

### Writing focus

这一段要突出两个关键词：

- **current management baseline**
- **yield and profit improvement potential**

建议明确写出研究问题从：

**What factors affect yield?**

推进到：

**How much model-estimated yield and profit improvement potential exists under constrained optimization of current management?**

### Possible topic sentence

Despite these advances, most yield-modeling studies stop at prediction or factor ranking, whereas practical decision support requires quantifying model-estimated yield and profit improvement potential relative to current management baselines.

---

## Paragraph 4. Study framework and rationale in the North China Plain

### Main purpose

说明本文如何解决上述缺口，以及为什么华北平原适合作为研究区域。这里要突出河北和山东是华北平原内具有不同管理基线的 sampled provinces，而不是把样本直接泛化为整个华北平原的处方推荐。

### Key points

- The North China Plain provides a meaningful case because maize fields differ in crop establishment, fertilization, irrigation, pest management, yield, cost, and profit.
- The study uses field-level observations from Hebei and Shandong as sampled provinces within the North China Plain.
- Management variables cover crop establishment, nutrient input, irrigation management, and pest management.
- Weather variables are included to control for environmental background.
- A leakage-controlled yield prediction model is developed to avoid using variables directly derived from yield or profit.
- Explainable machine learning is used to identify key predictors.
- The maximum-yield scenario is searched within observed and agronomically reasonable management ranges to reduce unsupported extrapolation.
- Profit is evaluated after yield optimization using the corresponding input-cost changes.
- The optimized scenario should be described as a model-estimated regional management configuration, not as a field-validated causal recommendation for individual farms.

### Writing focus

这一段要把你的技术路线讲清楚，但不要展开成 Methods。写成“研究框架说明”即可。具体变量名和优化算法细节留到 Methods，Introduction 只讲变量类别、基线、约束、最高产量情景和利润评价。

### Possible topic sentence

The North China Plain, where field-level maize management varies substantially across farms and provinces, provides an appropriate setting for evaluating baseline-dependent and scenario-based management optimization using explainable machine learning.

---

## Paragraph 5. Objectives and hypotheses

### Main purpose

收束引言，明确本文的研究目标和假设。建议写得直接，不要再引入新的背景信息。

### Recommended objectives

This study aimed to:

1. develop a leakage-controlled and interpretable machine-learning model for maize yield prediction using field-level management and weather variables;
2. identify the key management and environmental predictors associated with maize yield variation in the North China Plain;
3. quantify model-estimated yield improvement potential under constrained maximum-yield management scenarios relative to current management baselines;
4. evaluate whether the yield-oriented optimization scenarios also improve profit after accounting for changes in input cost.

### Recommended hypotheses

We hypothesized that:

1. irrigation-related variables, plant density, sowing date, and nutrient input would be major predictors of maize yield variation;
2. constrained maximum-yield scenarios would increase predicted yield relative to current baseline management, but profit gains would depend on whether yield benefits exceed additional input costs;
3. regions with lower current management baselines would show greater relative optimization potential.

### Writing focus

目标建议控制在 3-4 个。假设可以写成 1 句话，也可以写成 2-3 个短句。投稿时如果期刊不偏好显式 hypotheses，也可以只保留 objectives。注意避免把基于观测数据的模型情景结果写成已被田间试验证实的因果效应。

### Possible closing sentence

By linking current management baselines with constrained maximum-yield scenario optimization and post-optimization profit assessment, this study provides a practical framework for quantifying model-estimated management improvement potential in maize production on the North China Plain.

---

## Suggested Chinese Introduction Draft

以下为中文引言草稿，可在此基础上继续补充文献和结果衔接，后续再统一翻译润色为英文。

### Paragraph 1

华北平原是我国小麦-玉米集约化生产体系的核心区域，夏玉米生产对区域粮食供给具有重要意义。随着耕地、水资源和农业投入成本约束不断增强，该区域玉米持续增产已难以依赖肥料、灌溉等投入的简单增加，而更依赖于现有管理措施的优化配置。已有研究表明，中国及华北平原玉米生产中仍存在显著的农户间管理差异和可挖掘的产量差距，产量变异与播种建植、施肥和灌溉等管理措施密切相关（Chen et al., 2011; Meng et al., 2013; Chen et al., 2019; Wang et al., 2023）。同时，不合理的氮肥和水分管理可能降低资源利用效率、增加环境压力并削弱农户收益，表明高产目标需要与投入效率和经济收益协同考虑（Chen et al., 2014; Liu et al., 2021; Wang et al., 2023）。在这一背景下，华北平原玉米精准管理的关键问题已不只是识别产量限制因素，而是评估当前管理基线下仍可实现的产量和利润提升空间。

### Paragraph 2

机器学习为复杂农业系统中的产量预测和产量变异解析提供了有效工具。作物产量形成受到品种、气象、土壤和管理因素的非线性作用及其交互影响，随机森林、梯度提升、XGBoost 等树模型及集成学习算法能够较好地刻画这类复杂响应关系，并已被广泛用于不同区域尺度的玉米产量预测和产量风险评估（Jeong et al., 2016; Khaki and Wang, 2019; Shahhosseini et al., 2019, 2020; van Klompenburg et al., 2020）。在预测建模基础上，可解释机器学习方法能够评估不同预测因子的相对贡献及其响应模式，从而增强数据驱动模型在精准农业和管理决策支持中的应用价值（Hu et al., 2023; Smith et al., 2026）。然而，预测精度并不等同于因果解释，模型在观测数据支撑范围之外的外推也可能存在较大不确定性。因此，将机器学习模型用于管理情景分析时，需要同时考虑模型解释、数据支撑范围和情景约束（Crane-Droesch, 2018; Hu et al., 2023）。

### Paragraph 3

尽管已有研究在产量预测和关键因子识别方面取得了重要进展，但多数研究仍停留在模型精度比较或变量重要性排序层面。变量重要性能够揭示哪些因素与产量变异关系更密切，却难以回答这些因素在现实约束下应如何组合调整，也难以量化某一区域相对于当前管理状态的可改进空间。对于农户和区域决策者而言，更关键的是在观测数据支持且农学合理的范围内，评估播期、种植密度、养分投入、灌溉和植保等可调控管理措施的协同优化能够带来多大的产量提升潜力。此外，高产情景并不必然具有经济可行性；若额外投入成本超过增产收益，模型预测的产量提升难以转化为可采纳的管理方案。因此，产量建模研究需要进一步从因子识别扩展到基于当前管理基线的情景评价，并将产量提升与投入成本和利润变化联系起来。

### Paragraph 4

华北平原玉米生产的管理异质性为开展上述分析提供了适宜场景。该区域玉米生产投入强度较高，播种建植、施肥、灌溉和植保等措施同时影响产量形成和生产成本；不同农户和省份之间的管理水平、产量表现和经济收益差异，也为比较不同管理基线下的优化潜力提供了基础。本研究基于河北和山东两个采样省份的地块尺度观测数据，结合田间管理和气象变量构建玉米产量预测模型。为降低信息泄漏风险，模型仅使用管理和环境背景变量预测产量，避免纳入由产量或利润直接派生的变量。随后，利用可解释机器学习识别影响产量变异的关键因子，并将训练后的产量模型用于受约束的管理情景搜索，在观测数据支持和农学合理的范围内估计最高产量情景。进一步结合播种、施肥、灌溉和植保等投入变化核算成本和利润，以评价模型估计的产量提升是否能够转化为经济收益。

### Paragraph 5

在此基础上，本研究旨在构建一个基于地块尺度管理和气象变量的可解释机器学习分析框架，用于评估华北平原玉米生产中相对于当前管理基线的管理优化潜力。具体目标包括：识别影响玉米产量变异的关键管理和环境因子；在受约束的最高产量管理情景下，量化模型估计的增产潜力；并进一步评价该情景下投入成本和利润的变化。本文假设，灌溉相关变量、种植密度、播种日期和养分投入是影响玉米产量变异的重要预测因子；受约束的最高产量情景能够相对于当前管理基线提高预测产量，但利润提升取决于增产收益是否超过额外投入成本；同时，当前管理基线较低的区域可能表现出更大的相对优化潜力。通过将当前管理基线、受约束的最高产量情景和利润评价相结合，本研究可为华北平原玉米生产中管理改进空间的量化和区域精准管理策略的制定提供参考。

---

## Writing cautions

1. 不建议把 Introduction 写成“XGBoost 应用论文”，否则会削弱农业管理优化主线。
2. 不建议过早强调模型精度，模型精度应主要放在 Results。
3. 不建议把研究问题写成泛泛的“提高玉米产量”，应明确写成“relative to current management baseline”。
4. 不建议只讲 yield，不讲 profit；本文的亮点之一是最高产量方案下同步评价经济收益。
5. 华北平原是主研究区，河北和山东应表述为 sampled provinces 或 study provinces within the North China Plain。
6. 不建议把模型情景优化写成田间试验验证过的因果推荐；应使用 model-estimated、scenario-based、constrained optimization 等审慎表述。
7. 不建议把本文表述为 profit maximization；更准确的定位是 maximum-yield strategy followed by profit assessment。

---

## Literature Used for Draft Paragraphs 1-2

- Chen X P, Cui Z L, Vitousek P M, Cassman K G, Matson P A, Bai J S, Meng Q F, Hou P, Yue S C, Romheld V, Zhang F S. 2011. Integrated soil-crop system management for food security. *Proceedings of the National Academy of Sciences*, 108(16): 6399-6404. https://doi.org/10.1073/pnas.1101419108
- Meng Q F, Hou P, Wu L, Chen X P, Cui Z L, Zhang F S. 2013. Understanding production potentials and yield gaps in intensive maize production in China. *Field Crops Research*, 143: 91-97. https://doi.org/10.1016/j.fcr.2012.09.023
- Chen X P, Cui Z L, Fan M S, Vitousek P, Zhao M, Ma W Q, Wang Z L, Zhang W J, Yan X Y, Yang J C, Deng X P, Gao Q, Zhang Q, Guo S W, Ren J, Li S Q, Ye Y L, Wang Z H, Huang J L, Tang Q Y, Sun Y X, Peng X L, Zhang J W, He M R, Zhu Y J, Xue J Q, Wang G L, Wu L, An N, Wu L Q, Ma L, Zhang W F, Zhang F S. 2014. Producing more grain with lower environmental costs. *Nature*, 514: 486-489. https://doi.org/10.1038/nature13609
- Chen G F, Cao H Z, Chen D D, Zhang L B, Zhao W L, Zhang Y, Ma W Q, Jiang R F, Zhang H Y, Zhang F S. 2019. Developing sustainable summer maize production for smallholder farmers in the North China Plain: An agronomic diagnosis method. *Journal of Integrative Agriculture*, 18(8): 1667-1679. https://doi.org/10.1016/S2095-3119(18)62151-3
- Wang H Z, Ren H, Zhang L H, Zhao Y L, Liu Y E, He Q J, Li G, Han K, Zhang J W, Zhao B, Ren B Z, Liu P. 2023. A sustainable approach to narrowing the summer maize yield gap experienced by smallholders in the North China Plain. *Agricultural Systems*, 204: 103541. https://doi.org/10.1016/j.agsy.2022.103541
- Liu B Y, Lin B J, Li X X, Virk A L, N'dri Y B, Zhao X, Kan Z R, Zhang H L. 2021. Appropriate farming practices of summer maize in the North China Plain: Reducing nitrogen use to promote sustainable agricultural development. *Resources, Conservation and Recycling*, 175: 105889. https://doi.org/10.1016/j.resconrec.2021.105889
- Jeong J H, Resop J P, Mueller N D, Fleisher D H, Yun K, Butler E E, Timlin D J, Shim K M, Gerber J S, Reddy V R, Kim S H. 2016. Random forests for global and regional crop yield predictions. *PLOS ONE*, 11(6): e0156571. https://doi.org/10.1371/journal.pone.0156571
- Khaki S, Wang L. 2019. Crop yield prediction using deep neural networks. *Frontiers in Plant Science*, 10: 621. https://doi.org/10.3389/fpls.2019.00621
- Shahhosseini M, Martinez-Feria R A, Hu G, Archontoulis S V. 2019. Maize yield and nitrate loss prediction with machine learning algorithms. *Environmental Research Letters*, 14(12): 124026. https://doi.org/10.1088/1748-9326/ab5268
- Shahhosseini M, Hu G, Archontoulis S V. 2020. Forecasting corn yield with machine learning ensembles. *Frontiers in Plant Science*, 11: 1120. https://doi.org/10.3389/fpls.2020.01120
- Shahhosseini M, Hu G, Huber I, Archontoulis S V. 2021. Coupling machine learning and crop modeling improves crop yield prediction in the US Corn Belt. *Scientific Reports*, 11: 1606. https://doi.org/10.1038/s41598-020-80820-1
- van Klompenburg T, Kassahun A, Catal C. 2020. Crop yield prediction using machine learning: A systematic literature review. *Computers and Electronics in Agriculture*, 177: 105709. https://doi.org/10.1016/j.compag.2020.105709
- Crane-Droesch A. 2018. Machine learning methods for crop yield prediction and climate change impact assessment in agriculture. *Environmental Research Letters*, 13: 114003. https://doi.org/10.1088/1748-9326/aae159
- Hu T, Zhang X, Bohrer G, Liu Y, Zhou Y, Martin J, Li Y, Zhao K. 2023. Crop yield prediction via explainable AI and interpretable machine learning: Dangers of black box models for evaluating climate change impacts on crop yield. *Agricultural and Forest Meteorology*, 336: 109458. https://doi.org/10.1016/j.agrformet.2023.109458
- Smith H W, Heffernan C J, Ashworth A J, Nalley L L, Bullock D S, Tullis J, Owens P R. 2026. Harvesting insights: interpretable machine learning to understand environmental drivers of U.S. maize and soybean yield. *Scientific Reports*, 16: 8994. https://doi.org/10.1038/s41598-026-38724-z
