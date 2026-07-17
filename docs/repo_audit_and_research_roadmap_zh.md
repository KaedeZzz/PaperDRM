# PaperDRM 仓库审计、技术路线评估与研究路线图

更新日期：2026-07-16

## 1. 执行结论

PaperDRM 已经是一个有真实研究价值的原型，而不是一个无效项目。当前最可靠的成果是：

- 多方向图像的频谱聚合可以较准确地估计帘纹线密度；
- 在 9 个具有 per-folio 人工标注的样本上，8/9 的密度误差小于 10%；
- 9 个样本的绝对百分比误差中位数为 2.30%；
- 排除唯一灾难性失败样本后，平均绝对百分比误差为 2.68%。

但它还不是可以直接作为稳定科研工具发布的系统。最重要的问题不是平均精度，而是：

1. 存在无法自动拒绝的灾难性错误峰；
2. 评估指标过去混淆了全局谱周期和容易受漏检影响的局部间隔均值；
3. 当前主路线只聚合了多光照图像的频谱和相位，没有真正利用已知光照方向重建表面法向或深度；
4. 结果的统计验证、失败检测和跨数据集复现仍不完整。

因此，当前合理定位是：

> 一个表现良好的频谱检测 baseline，具有明确的 photometric stereo 升级路线，但尚未完成可靠性闭环。

## 2. 当前仓库与工程状态

### 2.1 主技术路线

仓库包含三条检测路线：

1. `multi_phi`：从多个方位角的掠射光图像分别提取频谱，聚合功率谱，估计周期，并对各方向相位进行极性对齐。这是当前推荐路线。
2. `simple`：单张图像的径向 FFT、Gabor 清理和相位网格，是消融基线。
3. `legacy`：DRP 方向图、三角掩膜和 patch Gabor。该路线存在绝对值响应导致的谐波加倍/半周期偏差，只应保留为历史消融。

### 2.2 本轮已完成的工程修复

- 缓存身份加入输入文件、预处理参数、ROI、角度筛选和背景文件指纹；
- 结果归档改为显式 artifact 列表，运行前清理旧的受管结果，避免跨实验污染；
- 默认配置改为仓库内相对路径；
- 包安装、wheel 构建和隔离环境导入已经验证；
- 修复负 self-contrast z 分数被当成正置信度的问题；
- 增加周期搜索边界诊断，命中边界时报告直接判为无效；
- 主报告改用全局谱周期作为间距/密度主结果；
- 局部峰间隔仅作为中位数和 IQR 描述，不再用受长间隔离群值影响的均值冒充主测量；
- GT 汇总改为优先使用每个 folio 自己的 `manual_gt.json`，spreadsheet 数值只保留为次级参考；
- 当前自动测试为 13 项，全部通过。

### 2.3 仍存在的工程债务

- 测试主要覆盖缓存、归档、报告和诊断，尚未覆盖主要图像算法；
- 多个历史分析脚本存在重复的结果 schema 兼容逻辑；
- 部分绘图函数在无界面后端仍调用 `plt.show()`，会产生警告；
- 仓库根目录保留了一批被忽略的历史诊断产物，应在确认研究价值后迁移到专门的实验归档目录；
- 旧数据集结果尚未全部用当前代码和边界诊断重新生成。

## 3. 修正后的基线结果

### 3.1 数据集 4 的完整 multi-phi 运行

输入和筛选：

- 原始匹配图像：480 张；
- 角度预筛选后加载：80 张；
- 主检测使用：20 个 phi 方向的掠射光图像；
- 运行时间：约 18 秒。

主要结果：

| 指标 | 结果 |
|---|---:|
| 全局谱周期 | 55.351 px |
| 物理间距 | 1.1689 mm |
| 帘纹密度 | 8.5549 条/cm |
| 独立 Fourier 最优周期 | 56.0 px |
| 周期搜索边界 | 未命中，距边界 11 个频率 bin |
| 局部间隔中位数 | 59 px / 1.246 mm |
| 局部间隔 IQR | 55–64 px |
| 局部中位数相对谱周期偏差 | 6.6% |
| 线影 FWHM | 0.319 mm |
| split-half 周期差标准差 | 0.686 px |
| 1 px 内一致率 | 100% |
| self-contrast | z = +2.27 |
| 4 阶 Fourier R² | 0.0326 |
| 频率集中度 | 0.311 |

解释：

- 周期估计不再被搜索范围截断，且与独立 Fourier 扫描一致；
- split-half 表明结果具有较好的数值重复性；
- self-contrast 提供中等强度的空间支持；
- 低 R² 表明周期结构只解释了图像一小部分方差，因此该结果应描述为“可信但证据中等”，不能描述为强验证；
- 数据集 4 没有配套人工 GT，不能用它单独证明绝对精度。

### 3.2 九个 folio 的人工 GT benchmark

主预测统一使用保存的全局谱周期和每个样本的物理标定。

| Folio | 人工 GT（条/cm） | 预测（条/cm） | 误差 |
|---|---:|---:|---:|
| Kk1-5_f5v | 9.352 | 9.137 | -2.30% |
| Kk1-5_f9v | 9.351 | 9.289 | -0.66% |
| Hh2-12_f190 | 8.909 | 8.864 | -0.51% |
| Ee5-22_f328r | 7.570 | 7.667 | +1.28% |
| Ff2-6_f140r | 10.158 | 9.714 | -4.37% |
| Ff4-9_f42r | 5.901 | 5.476 | -7.20% |
| Ff4-15_f24r | 12.076 | 5.368 | -55.55% |
| Hh2-10_f24r | 11.518 | 11.073 | -3.86% |
| Ii3-8_f135v | 7.402 | 7.311 | -1.22% |

汇总：

- 8/9 样本误差小于 10%；
- 中位绝对百分比误差：2.30%；
- 全部样本平均绝对百分比误差：8.55%；
- 排除唯一灾难性失败后的平均绝对百分比误差：2.68%。

这说明算法的“正常工作状态”精度很好，但目前没有可靠的自动 abstention/rejection 机制。

失败样本 `Ff4-15_f24r` 同时表现为：

- 频率集中度仅 0.0046；
- 4 阶 Fourier R² 仅 0.00054；
- self-contrast z = -0.81；
- 检测周期与其他拟合候选不一致。

这些信号足以说明结果可疑，但现有阈值体系还不能在不误伤其他正确样本的情况下稳定拒绝它。

## 4. 与文献的关系

### 4.1 直接相关的帘纹/链线检测

Grossmann、Schönlieb 和 Da Rold 使用 spectral total variation 分解增强反射光图像中的低对比帘纹和链线，再结合 Radon/Fourier 方法提取线结构。该工作证明了普通反射光图像也可以支持纸张结构分析，并提供了当前项目应加入的重要单图像 baseline：

- [Extracting chain lines and laid lines from digital images of medieval paper using spectral total variation decomposition, Heritage Science, 2023](https://www.nature.com/articles/s40494-023-01013-3)

ChainLineNet 将分割、方向对齐、2D Fourier、候选线生成和可微直线拟合结合，在 95 张透射光图像上实现了高精度链线参数化。其价值不在于当前就复制一个 GAN，而在于它给出了更合适的评估方式：线级 precision/recall、位置误差和间隔误差，而不是只报告一个全局周期。

- [ChainLineNet: Deep-Learning-Based Segmentation and Parameterization of Chain Lines in Historical Prints, 2021](https://www.mdpi.com/2313-433X/7/7/120)

Gorske 等人的 moldmate 工作表明，研究目标不应停留在一个全局 laid-line density 数字。沿纸张位置变化的局部密度图、链线间隔序列和水印位置共同构成更有区分力的纸模指纹。

- [Moldmate Identification in Pre-19th-Century European Paper, 2021](https://ahnp.ub.uni-heidelberg.de/journals/dah/article/view/71232)

### 4.2 多光照与 photometric stereo

Brenner 对 graphical heritage 的系统研究与本项目数据最接近。其结果表明：

- 对近似平坦的文献表面，6–12 个在约 32°–51° elevation 环形分布的光源可以接近完整 dome 的重建质量；
- 对近距离点光源，直接使用平行光假设会产生系统误差；
- 对平面文献进行局部 patch photometric stereo、使用平面灰卡校正入射强度，可以显著降低误差；
- 法向/深度图经过高通后可提取纸张压痕等微弱表面结构。

- [Multi-Light Imaging for Graphical Heritage, doctoral thesis, 2024](https://repositum.tuwien.at/handle/20.500.12708/197553)

当前数据已经提供 phi/theta 光照几何，因此只做多图频谱求和没有用完采集信息。最自然的升级是显式估计法向和高通深度，再在几何域检测周期结构。

实际采集还需要考虑阴影、镜面反射和非 Lambertian 响应。低秩 photometric stereo 将理想 Lambertian 图像栈建模为低秩矩阵，并把阴影和高光视为稀疏异常，是适合当前多光照栈的稳健预处理/求解方向：

- [Robust Photometric Stereo via Low-Rank Matrix Completion and Recovery, 2011](https://people.eecs.berkeley.edu/~yima/matrix-rank/stereo.html)

近光模型对文化遗产成像尤其重要；忽略光源距离和空间衰减会造成整体弯曲等伪影：

- [Near Light Correction for Image Relighting and 3D Shape Recovery, 2015](https://diglib.eg.org/bitstream/handle/10.2312/14536/04_DigitalHeritage2015_submission_82.pdf)

RTI 文献也证明了从多光照法向图对文档表面轮廓进行定量测量的可行性，并通过共焦显微测量进行了对照：

- [The Preliminary Attempts to Quantify the Three-dimensional Details of Document Surfaces with RTI](https://journal.asqde.org/articles/10.69525/jasqde.236)

## 5. 推荐的研究主线

### P0：先完成可信 benchmark 和失败拒绝

这是发表前必须完成的工作。

1. 用当前代码重新运行全部 9 个 folio，确保所有结果都包含：
   - 搜索边界诊断；
   - 全局谱周期与局部间隔分离；
   - 一致的 polarity 语义；
   - 当前缓存和归档 schema。
2. 将人工 GT 作为唯一主 benchmark，spreadsheet 只作为外部参考。
3. 报告：
   - 密度 MAE/MAPE；
   - line-position precision/recall 和位置误差；
   - 成功覆盖率；
   - coverage-error 曲线。
4. 开发自动拒绝分数，但不能在 9 个样本上硬编码阈值。候选特征包括：
   - 谱峰与次峰比；
   - 峰距离搜索边界的距离；
   - 不同 phi 子集的周期分布；
   - 不同空间 patch 的周期一致性；
   - self-contrast 的符号与强度；
   - 局部间隔中位数与谱周期的一致性。
5. 使用 leave-one-folio-out 或预注册阈值，避免用测试样本调阈值。

### P1：建立真正的 photometric stereo 路线

建议新增独立路线，而不是替换现有 baseline：

```text
多光照图像 + 已知 phi/theta
        ↓
阴影/饱和/异常值掩膜 + flat-field 校正
        ↓
局部 calibrated / robust photometric stereo
        ↓
albedo + normal map + integrated depth
        ↓
去除纸张大尺度弯曲的高通深度/法向分量
        ↓
Radon / Fourier / 局部线参数化
        ↓
帘纹密度、局部密度图、链线位置与不确定度
```

第一阶段不需要神经网络。建议先实现：

- 经典 calibrated least squares PS；
- 阴影和饱和像素剔除；
- patchwise 求解和重叠融合；
- 6、12、20 个光照方向的消融；
- 一个 robust regression 或低秩恢复版本；
- 法向分量和高通深度上的频谱检测。

研究问题应写成：

> 在历史纸张的多光照图像中，显式表面几何重建是否能比直接强度频谱聚合更准确、更稳定，并减少灾难性错误？

### P2：从全局密度扩展到空间纸模指纹

在获得稳定的几何增强图后，生成：

- 滑动窗口局部 period/density map；
- 帘纹方向和弯曲度；
- 链线位置及其间隔序列；
- 水印区域及其与链线的相对位置；
- 不同 folio 之间经过尺度、旋转和翻转配准后的相似度。

这条路线与 moldmate identification 的文献目标一致，也比单一密度值更有历史研究价值。

### P3：加入 reflected-light spectral-TV baseline

将 spectral-TV 路线用于：

- 单张普通反射光图像；
- photometric stereo 的 albedo、法向分量或高通深度图；
- 与当前 Gabor、原始 FFT、多光照频谱方法做消融。

如果该路线在单张反射光图像上接近多光照结果，项目可以同时形成：

1. 面向低成本档案图像的单图像方法；
2. 面向专门采集的高可靠多光照方法。

## 6. 不建议立即采取的方向

- 不建议现在直接训练大型深度网络：当前人工标注样本太少，且采集域单一；
- 不建议继续把主要精力投入 legacy trig-mask/Gabor 路线：其偏差机制已经明确；
- 不建议只优化平均误差而忽略 abstention：当前最大的科学风险是单个无提示的灾难性失败；
- 不建议把 split-half 稳定性当成正确性的证明：错误峰也可能在所有子集中稳定出现；
- 不建议继续用 spreadsheet 值替代 per-folio 人工 GT。

## 7. 可发表成果的合理表述

当前可以支持的表述：

> 基于多方向光照频谱聚合的帘纹密度估计，在九个具有人工标注的 folio 上有八个达到 10% 以内误差，成功样本的平均绝对误差约 2.7%，但仍存在一个无法可靠自动拒绝的灾难性失败。

当前不能支持的表述：

- 系统已在任意历史纸张上鲁棒工作；
- split-half 稳定性证明检测正确；
- 线影宽度已经获得绝对物理验证；
- 当前方法优于 photometric stereo 或 spectral-TV；
- 当前结果足以进行可靠的 moldmate identification。

最有潜力的论文贡献组合是：

1. 一个可复现的多光照纸张数据和人工 GT benchmark；
2. 直接频谱聚合 baseline；
3. robust/local photometric stereo 几何增强路线；
4. 面向灾难性错误的 uncertainty/abstention；
5. 从全局密度到空间纸模指纹的扩展。
