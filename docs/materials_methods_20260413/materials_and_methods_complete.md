# 材料与方法完整草稿

本文档基于 `themes/paper0314_precision_agriculture.md`、`docs/project_brief.md`、`docs/model_variable_table.md` 以及当前分析脚本 `src/model.py`、`src/optimize_management_yield.py`、`src/plot_management_responses.py` 整理，可直接作为论文第 2 节的中文写作底稿。

## 2. Materials and Methods

### 2.1 研究区、数据来源与分析单元

本研究以黄淮海地区夏玉米生产系统为对象，样本主要来自河北和山东两省。黄淮海地区是中国重要的夏玉米生产带，管理方式、灌溉条件和投入水平差异明显，因此适合开展基于真实生产记录的管理优化潜力评估（Ren et al., 2022; Wu et al., 2021）。在当前整理后的数据库中，共纳入 81 个地块级观测，采用 `uuid` 作为唯一标识；其中河北 26 个样本，山东 55 个样本。研究的分析单元为地块级观测记录，每个样本同时包含产量、管理措施、灌溉方式、成本信息以及播后天气聚合变量。

建模数据表为 `data/model_vars.csv`，成本数据表为 `data/cost.csv`。前者用于产量建模与情景模拟，后者用于成本拆分与利润核算。目标变量为玉米单产 `Yield`，当前数据库已统一折算为 kg ha^-1；论文展示时可按需要换算为 kg 亩^-1。经济评价以利润为结果指标，但利润不直接参与机器学习训练，而是在产量预测完成后结合情景成本单独核算，以避免将产量和成本的派生关系引入预测目标。

### 2.2 变量构建、预处理与泄漏控制

本研究将变量划分为管理变量、环境变量和结果变量三类。管理变量包括播种日期（`Sow_DOY`）、种植密度（`Density`）、施氮量（`Fer_N`）、施磷量（`Fer_P`）、施钾量（`Fer_K`）、施肥次数（`Fer_Count`）、打药次数（`Pest_Count`）、打药成本（`Pest_Cost`）、灌溉次数（`Irr_Count`）、灌溉用电量（`Irr_Elec`）以及灌溉方式哑变量（`Irr_None`、`Irr_Flood`、`Irr_Sprinkler`、`Irr_Drip`）。环境变量包括播后累计降水（`Precip`）、播后累计辐射（`Rad`）、播后平均相对湿度（`RH`）和生长季最大风速（`Wind`）。其中，播后天气特征用于刻画地块所处的气象背景，但在情景优化阶段保持不变，仅允许管理变量发生改变。

数据预处理遵循统一、保守的原则。首先，将除样本标识外的全部字段转换为数值型，并删除目标变量缺失的观测。其次，对特征中的缺失值采用中位数填补；在线性模型和 Elastic Net 中进一步进行标准化处理，而树模型使用原始量纲输入。灌溉方式已在原始数据中处理为互斥哑变量，因此不再重复编码。由于样本量有限，本文不引入额外的高维交互项和衍生指标，以降低过拟合风险。

泄漏控制是本研究方法设计的核心。根据 Kaufman et al. (2012)，如果将与目标变量共享派生路径的信息直接输入模型，容易导致性能高估并削弱解释有效性。基于这一原则，本研究的产量模型严格以 `Yield` 为唯一预测目标，不将 `profit`、`total_cost`、`land_rent_cost` 或其他由产量和成本直接派生的综合经济指标纳入特征集合。利润仅在情景评价阶段基于预测产量和估算成本进行后验核算。这样可以确保模型回答的是“在既定天气背景和可观测管理条件下，产量如何响应管理调整”，而不是混合了收益核算逻辑的复合问题。

### 2.3 产量建模与可解释性分析

为获得能够支持后续情景优化的稳健产量响应模型，本文比较了四类回归算法：普通线性回归、Elastic Net、随机森林和 XGBoost。树模型能够较好刻画农业系统中常见的非线性关系和变量交互，其中 XGBoost 在复杂回归任务中的性能和计算效率已被广泛验证（Chen and Guestrin, 2016）；农业场景下，基于地块级管理和天气数据的机器学习建模也已被证明能够有效提升产量预测能力（Dhaliwal and Williams, 2024; Baio et al., 2023）。

在实现上，样本首先按 80%:20% 随机划分为训练集和验证集，并固定随机种子为 42。模型选择阶段在训练集内部执行 5 折交叉验证，以负 RMSE 作为搜索准则。各候选模型采用统一的预处理流程，不同之处仅在于是否进行标准化以及拟合器本身。训练完成后，在训练集和独立验证集上分别计算决定系数（R^2）、均方根误差（RMSE）和平均绝对误差（MAE），其计算形式分别为：

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i-\hat y_i)^2}{\sum_{i=1}^{n}(y_i-\bar y)^2}
$$

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
$$

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
$$

本文以验证集表现最优且过拟合程度可接受的模型作为后续管理情景模拟的基础模型。在当前数据集上，XGBoost 取得了最优综合表现，因此被选作最终产量预测器。

为解释关键管理变量与产量之间的关系，本文在最优树模型基础上计算 SHAP（SHapley Additive exPlanations）值。SHAP 将单个样本预测分解为各特征对预测值的边际贡献，能够在保证局部可加性的同时提供全局排序与响应方向信息（Lundberg and Lee, 2017）。具体实现上，使用 TreeExplainer 计算全样本 SHAP 值，并针对播期、密度、养分投入、灌溉次数、灌溉用电和打药成本等管理变量绘制依赖关系图，以识别管理优化中的高优先级变量和可能的非线性响应区间。解释分析的目的不是单独构建因果结论，而是为情景搜索中“哪些变量值得优先调整”提供经验支撑。

### 2.4 基线定义、最高产量情景优化与利润评价

本研究采用“当前管理基线 vs. 最高产量方案”的双情景框架。当前管理基线并不使用模型预测值，而是直接基于观测样本的中位数水平构建：在整体样本或区域子样本内，分别计算观测产量、利润和各管理变量的中位数，以代表当前生产条件下的典型水平。采用中位数而非均值，主要是为了降低小样本中极端值对代表性管理组合和经济结果的影响。

最高产量情景仅对可调控管理变量进行搜索，天气背景变量和其他非管理特征保持为各地块观测值不变，从而使优化结果可以解释为“在当前地块背景下，仅通过管理调整可达到的潜在提升”。为避免超出观测数据支撑范围，情景值全部限定在样本经验分布内。连续管理变量采用观测分布的 3 个分位点构建候选水平，具体为 0.10、0.55 和 1.00 分位；计数型变量（施肥次数、打药次数、灌溉次数）则在观测最小值与最大值之间取整数水平。灌溉方式通过互斥哑变量枚举为无灌溉、漫灌、喷灌和滴灌四类情景。基于这些候选水平，构建管理变量的笛卡尔组合，并逐一替换到全部地块样本上进行预测。

对于任一候选情景 $s$，将其管理变量值替换到每个地块的特征向量中，得到预测产量 $\hat Y_{is}$。随后以该情景下全体地块预测产量的中位数作为区域层面的情景单产：

$$
\tilde Y_s = \text{median}(\hat Y_{1s}, \hat Y_{2s}, \ldots, \hat Y_{ns})
$$

最终将 $\tilde Y_s$ 最大的情景定义为最高产量方案；若多个情景的中位产量相同，则优先选择中位利润更高的方案。这样定义的优化目标是区域代表性管理组合，而不是为每个地块分别生成个体化处方，因此更符合本文“区域化管理优化潜力评估”的研究定位。

利润评价遵循“先产量、后收益”的路径。成本表与建模表通过 `uuid` 内连接后，分别拟合三组线性成本子模型：播种成本以密度为自变量，施肥成本以氮、磷、钾投入量为自变量，灌溉运行成本以灌溉用电量为自变量。同时，根据观测样本计算不同灌溉方式对应的灌溉设备成本中位数。对于每个样本，还计算一个不随情景变化的固定成本项，其定义为总成本减去上述可变成本部分。于是，任一情景下第 $i$ 个样本的投入成本可写为：

$$
C_{is} = C^{fixed}_i + C^{sow}_s + C^{fert}_s + C^{pest}_s + C^{irr}_s + C^{device}_s
$$

在给定粮食价格 $p$ 的情况下，本研究按脚本设定取 $p=2.4$ 元 kg^-1，并据此计算情景利润：

$$
\Pi_{is} = p \cdot \hat Y_{is} - C_{is}
$$

区域层面的情景利润定义为全体样本情景利润的中位数：

$$
\tilde \Pi_s = \text{median}(\Pi_{1s}, \Pi_{2s}, \ldots, \Pi_{ns})
$$

在整体和分区域层面，增产和增利幅度分别按以下公式计算：

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

其中，$Y_{base}$ 和 $\Pi_{base}$ 分别为基线情景下的观测中位产量和中位利润，$Y_{opt}$ 和 $\Pi_{opt}$ 分别为最高产量方案对应的预测中位产量和预测中位利润。

### 2.5 区域比较与不确定性说明

为识别优化潜力的空间异质性，本文在整体样本之外，进一步分别在河北和山东子样本上重复相同的基线构建、情景搜索和利润核算流程，并比较不同区域的管理调整方向及其增产增利幅度。区域比较的目的在于判断“当前管理基础不同是否导致优化潜力不同”，而不是在现有样本量条件下生成更细尺度的田间处方。

需要说明的是，本文结果本质上属于基于观测样本和机器学习模型的情景估计，而不是田间干预试验的实测效果。因此，最高产量方案应被理解为在当前数据支持范围内、在既定天气背景下的潜在提升水平，而非任何地块都必然达到的实际产量。另一方面，当前样本量仅为 81 个地块，且空间覆盖主要集中于河北和山东；土壤、品种和逐日环境信息的维度仍然有限，因此模型外推能力和管理建议的普适性仍需在更大范围样本及田间验证中进一步检验。尽管如此，采用严格泄漏控制、区域内基线比较和观测范围约束的情景搜索框架，能够较稳健地回答本文的核心问题，即在当前管理基础上，黄淮海地区玉米生产还存在多大的可量化增产增利空间。

## 可直接用于参考文献的条目

以下条目均为本次补充时核对过的真实文献，可按目标期刊格式进一步调整。

1. Baio, F.H.R., Santana, D.C., Teodoro, L.P.R., Oliveira, I.C., Gava, R., Oliveira, J.L.G., Silva Junior, C.A., Teodoro, P.E., and Shiratsuchi, L.S. 2023. Maize yield prediction with machine learning, spectral variables and irrigation management. *Remote Sensing*, 15(1), 79. https://doi.org/10.3390/rs15010079
2. Chen, T., and Guestrin, C. 2016. XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785
3. Dhaliwal, D.S., and Williams, M.M. II. 2024. Sweet corn yield prediction using machine learning models and field-level data. *Precision Agriculture*, 25, 51-64. https://doi.org/10.1007/s11119-023-10057-1
4. Kaufman, S., Rosset, S., Perlich, C., and Stitelman, O. 2012. Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 15. https://doi.org/10.1145/2382577.2382579
5. Lundberg, S.M., and Lee, S.-I. 2017. A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems*, 30, 4765-4774. https://arxiv.org/abs/1705.07874
6. Ren, H., Liu, M., Zhang, J., Liu, P., and Liu, C. 2022. Effects of agronomic traits and climatic factors on yield and yield stability of summer maize (*Zea mays* L.) in the Huang-Huai-Hai Plain in China. *Frontiers in Plant Science*, 13, 1050064. https://doi.org/10.3389/fpls.2022.1050064
7. Sihi, D., Dari, B., Kuruvila, A.P., Jha, G., and Basu, K. 2022. Explainable machine learning approach quantified the long-term (1981-2015) impact of climate and soil properties on yields of major agricultural crops across CONUS. *Frontiers in Sustainable Food Systems*, 6, 847892. https://doi.org/10.3389/fsufs.2022.847892
8. Wu, D., Xie, R., Ming, B., Hou, P., Xue, J., Ren, H., Zhang, W., Wang, K., and Li, S. 2021. The priority of management factors for reducing the yield gap of summer maize in the north of Huang-Huai-Hai region, China. *Journal of Integrative Agriculture*, 20(2), 450-459. https://doi.org/10.1016/S2095-3119(20)63294-4

## 投稿前建议核对的两项元数据

1. 当前项目文档未保留样本年份范围与原始气象数据源名称；正式投稿前，建议在本节 2.1 或补充材料中补上具体年份、站点或再分析数据来源。
2. 论文正文若统一以“kg/亩”和“元/亩”报告结果，建议在方法开头明确说明所有原始变量的面积单位换算规则，以保证模型变量、成本核算和结果表述完全一致。

## 本次补充时查阅的在线来源

1. XGBoost: https://arxiv.org/pdf/1603.02754
2. SHAP: https://arxiv.org/abs/1705.07874
3. Leakage in data mining: https://www.researchgate.net/publication/221653692_Leakage_in_Data_Mining_Formulation_Detection_and_Avoidance
4. Sweet corn yield prediction using machine learning models and field-level data: https://doi.org/10.1007/s11119-023-10057-1
5. Maize yield prediction with machine learning, spectral variables and irrigation management: https://doi.org/10.3390/rs15010079
6. Effects of agronomic traits and climatic factors on yield and yield stability of summer maize in the Huang-Huai-Hai Plain in China: https://doi.org/10.3389/fpls.2022.1050064
7. Explainable machine learning approach quantified the long-term impact of climate and soil properties on yields of major agricultural crops across CONUS: https://doi.org/10.3389/fsufs.2022.847892
8. The priority of management factors for reducing the yield gap of summer maize in the north of Huang-Huai-Hai region, China: https://doi.org/10.1016/S2095-3119(20)63294-4
