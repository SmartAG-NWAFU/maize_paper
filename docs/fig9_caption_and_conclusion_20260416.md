# Fig. 9 图注与 Conclusions 正式稿

本文档基于当前 `fig/fig9_region_optimization_results.png`、`fig/fig9_region_optimization_summary.csv` 和 `outputs/management_yield_region_optimized/management_region_baseline_vs_optimal.csv` 整理，可直接用于英文 SCI 稿件。

## Fig. 9 caption

### Recommended English caption

**Figure 9. Yield and profit gains under the maximum-yield optimization strategy in Hebei and Shandong**

**Figure 9.** Comparison of the baseline and optimized scenarios for (a) yield and (b) profit in Hebei and Shandong. The baseline represents the regional median under current management, whereas the optimized scenario represents the regional maximum-yield strategy identified by the machine-learning-based optimization framework. In Hebei, the optimized scenario increased median yield from 9,585.00 to 12,209.45 kg ha^-1 and median profit from 10,764.39 to 16,163.22 CNY ha^-1. In Shandong, the optimized scenario increased median yield from 10,721.85 to 12,603.64 kg ha^-1 and median profit from 12,031.56 to 17,615.88 CNY ha^-1.

### Shorter journal-style version

**Figure 9.** Comparison of baseline and optimized regional median (a) yield and (b) profit in Hebei and Shandong under the maximum-yield strategy. The optimized scenario increased both yield and profit in the two regions, with larger relative gains observed in Hebei than in Shandong.

### 中文对应版本

**图9 最高产量优化策略下河北和山东的产量与利润提升**

**图9** 河北和山东当前管理基线与最高产量优化情景下的区域中位产量和利润比较。(a) 产量；(b) 利润。基线表示当前管理条件下的区域中位水平，优化情景表示基于机器学习优化框架识别的最高产量方案。与基线相比，优化情景在河北将中位产量由 9,585.00 提高到 12,209.45 kg ha^-1，中位利润由 10,764.39 提高到 16,163.22 CNY ha^-1；在山东将中位产量由 10,721.85 提高到 12,603.64 kg ha^-1，中位利润由 12,031.56 提高到 17,615.88 CNY ha^-1。

## Conclusions

### Recommended SCI-style English version

## 5. Conclusions

Using field-level observations from the Huang-Huai-Hai region, this study developed a leakage-controlled and interpretable machine-learning framework to quantify how much maize yield and profit could be improved through management optimization under a maximum-yield strategy. The results showed that current maize production still retains substantial room for improvement in both Hebei and Shandong. Among the candidate algorithms, XGBoost provided the best predictive performance and identified irrigation-related variables, plant density, and sowing date as major predictors of yield variation.

The scenario optimization analysis further demonstrated that management adjustment could simultaneously improve yield and profit. Under the optimized scenario, median yield increased from 9,585.00 to 12,209.45 kg ha^-1 in Hebei and from 10,721.85 to 12,603.64 kg ha^-1 in Shandong, while median profit increased from 10,764.39 to 16,163.22 CNY ha^-1 and from 12,031.56 to 17,615.88 CNY ha^-1, respectively. The larger relative gain observed in Hebei suggests that regions with a lower current management baseline may have greater optimization potential. Across regions, the main directions of improvement included earlier sowing, denser planting, strengthened nutrient input, and more effective irrigation support, particularly under drip irrigation.

Overall, the study shows that a baseline-to-optimization framework driven by explainable machine learning can move beyond factor identification and directly quantify region-specific yield and profit gains from management improvement. This makes the approach relevant for precision agriculture decision support at the regional scale. Nevertheless, the current conclusions should be interpreted within the support of the existing dataset, and further validation across more years, locations, and field experiments is needed before broader application.

### More concise version

## 5. Conclusions

This study quantified the potential yield and profit gains achievable through management optimization in maize production in the Huang-Huai-Hai region. Using a leakage-controlled XGBoost model and a region-level optimization framework, we showed that current management can still be substantially improved in both Hebei and Shandong. Irrigation management, plant density, and sowing date were identified as the most influential predictors of yield variation.

Under the maximum-yield strategy, yield and profit increased simultaneously in both regions, with larger relative gains in Hebei than in Shandong. These results indicate that regions with a lower current management baseline may have greater optimization potential. The proposed framework provides a practical way to quantify management improvement potential for regional decision support in precision agriculture, although additional validation across more environments and years remains necessary.

### 中文正式版

## 5. 结论

本研究基于黄淮海地区玉米地块级观测数据，构建了一个兼顾泄漏控制与可解释性的机器学习分析框架，用于量化当前管理条件下通过管理优化所能实现的增产增利潜力。结果表明，河北和山东玉米生产在现有管理基础上均仍存在显著的提升空间。在候选模型中，XGBoost 具有最佳预测性能，并表明灌溉相关变量、种植密度和播期是驱动产量差异的关键因素。

情景优化结果进一步表明，以最高产量为导向的管理调整能够同步提高产量和利润。与当前管理基线相比，优化情景下河北中位产量由 9,585.00 提高至 12,209.45 kg ha^-1，山东由 10,721.85 提高至 12,603.64 kg ha^-1；河北中位利润由 10,764.39 提高至 16,163.22 CNY ha^-1，山东由 12,031.56 提高至 17,615.88 CNY ha^-1。河北表现出更高的相对增益，说明当前管理基础较低的区域通常具有更大的优化潜力。两区域的最优调整方向总体一致，主要包括适当前移播期、提高种植密度、优化养分投入以及强化灌溉保障，尤其是滴灌条件下的管理强化。

总体来看，本研究表明，基于可解释机器学习的“当前基线-优化情景”框架能够突破单纯变量识别的局限，直接量化区域尺度管理改进所带来的增产增利潜力，从而为精准农业中的区域化决策支持提供依据。但应当指出，本文结论仍受限于现有样本覆盖范围和变量维度，未来仍需结合更多年份、区域和田间验证进一步检验其稳健性与推广性。

## 使用建议

1. 如果正文偏精炼，优先使用上面的 `More concise version`。
2. 如果你准备投 `Precision Agriculture` 一类期刊，建议使用 `Recommended SCI-style English version`。
3. `Fig. 9` 当前图中只显示产量和利润两个面板，因此图注不要写 input cost；成本变化可放在 Results 正文中解释。
