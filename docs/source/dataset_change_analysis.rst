数据集变化分析
==============

本节对 ``initial_population.pkl`` 与 ``final_population.pkl`` 进行配对分析。每个
individual 按 ``index`` 一一对应，比较优化前后的 ``rho``、有效数据比例、reward
统计量以及 ``tau`` 通道统计量变化。

核心结论
--------

- 共比较 200 对 individual。
- ``rho`` 均值从 ``0.024489`` 上升到 ``0.025914``，平均增量为 ``0.001424``。
- ``rho`` 的配对 t 检验 p 值为 ``1.70e-08``，说明整体上升是显著的。
- 123 个 individual 的 ``rho`` 上升，76 个下降，1 个基本不变。
- ``reward_mean``、``valid_ratio`` 和 ``tau`` 均值没有表现出同等明显的整体迁移。
- PCA 图显示 initial 与 final 的 summary feature 分布大量重叠，说明 final population
  更像是在原分布附近发生局部扰动，而不是整体迁移到新区域。
- 车-路位置热力图显示，``tau0_distance`` 基本保持不变，``tau1_queue`` 和 ``reward``
  主要是局部正负波动；``rho`` 的提升更像是 population 级别的整体改善，而不是某个固定
  car-road 区域单独驱动。

分析方法
--------

分析时先将每个 population 转换为 individual 级 CSV：每个 individual 一行，记录
``rho``、有效比例、reward 分布、best reward、top-2 reward gap，以及 ``tau`` 两个通道的均值、
标准差和分位数。随后按 ``index`` 做 final - initial 的配对差值分析，并输出：

- 均值、标准差、中位数、最小值、最大值。
- 正/负/零变化计数。
- Pearson initial-final 相关系数。
- Cohen's dz 配对效应量。
- 配对 t 检验。
- 分布图、散点图、差值直方图、箱线图和 PCA 低维可视化。
- 车-路位置热力图：横轴为 ``road_index``，纵轴为 ``car_index``，每个格子表示该位置在
  200 个 individual 中的平均值。每张图包含 initial、final 和 ``final - initial`` 三个子图；
  差值图以 0 为中心，红色表示变大，蓝色表示变小。

rho 分布变化
------------

``final_population`` 的 ``rho`` 分布整体向右移动，低 ``rho`` 样本减少，中高 ``rho`` 样本增加。
这是最直接体现 final population 改善的图。

.. image:: images/rho_distribution.png
   :alt: rho distribution
   :align: center

rho 配对差值
------------

大部分样本的 ``rho`` 变化集中在 0 附近，但正值区域更厚，右尾更长，说明多数个体有提升，
且少数样本提升幅度较大。

.. image:: images/rho_delta_histogram.png
   :alt: rho delta histogram
   :align: center

rho 初始值与最终值
------------------

虚线表示 ``final = initial``。图中较多点位于虚线上方，说明 final 的 ``rho`` 通常更高。
但点云较分散，说明 initial 的个体排序没有稳定延续到 final。

.. image:: images/rho_initial_vs_final.png
   :alt: rho initial vs final
   :align: center

reward 均值分布
---------------

``reward_mean`` 的 initial 与 final 分布高度重叠，说明平均 reward 并没有发生明显整体迁移。
因此 ``rho`` 的提升不是简单由全局平均 reward 上升解释的。

.. image:: images/reward_mean_distribution.png
   :alt: reward mean distribution
   :align: center

关键指标差值箱线图
------------------

不同指标量纲不同，因此该图主要用于观察每个指标自身是否偏离 0。``tau0_mean`` 的个体波动最大，
但中位数接近 0；``rho``、``reward_mean``、``valid_ratio`` 等指标的绝对变化量较小。

.. image:: images/selected_delta_boxplot.png
   :alt: selected metric delta boxplot
   :align: center

PCA 整体结构变化
----------------

PCA 将多个 summary feature 压缩到二维。蓝点为 initial，橙点为 final，灰线连接同一个
individual 的初始和最终状态。两类点大量重叠，说明 final population 没有形成清晰独立的新群体；
灰线方向较分散，说明个体变化方向不统一。

.. image:: images/pca_population_change.png
   :alt: PCA population change
   :align: center

车-路位置热力图
----------------

位置热力图用于观察变化是否集中在特定车辆编号或道路编号上。灰色区域表示该 car-road
位置没有有效数据，因此不参与平均。initial 和 final 子图使用相同颜色尺度，便于直接比较；
最后一个子图展示 ``final - initial``，用于突出优化前后的局部差异。

tau0_distance 位置变化
~~~~~~~~~~~~~~~~~~~~~~

``tau0_distance`` 的 initial 和 final 图几乎一致，差值图也基本接近 0。说明距离通道主要由
道路和车辆的几何关系决定，在这批数据的优化前后没有发生可见的系统性变化。换句话说，
``rho`` 的提升不太可能来自距离矩阵整体变短或变长。

.. image:: images/tau0_distance_position_heatmap.png
   :alt: tau0 distance position heatmap
   :align: center

tau1_queue 位置变化
~~~~~~~~~~~~~~~~~~~

``tau1_queue`` 的高值呈现明显的道路列结构，说明部分道路在许多车辆位置上都具有更高排队值。
差值图中红蓝条纹交错，变化集中在有效车辆区域内，但没有表现为所有道路或所有车辆同方向增加。
这说明队列变化更像是局部重新分布，而不是整体拥堵水平统一上升或下降。

.. image:: images/tau1_queue_position_heatmap.png
   :alt: tau1 queue position heatmap
   :align: center

reward 位置变化
~~~~~~~~~~~~~~~

``reward`` 的 initial 与 final 热力图整体结构非常接近，差值图呈现细碎的正负交替。
这与前面的 ``reward_mean`` 分布结论一致：平均 reward 没有明显整体迁移，位置层面也没有出现
某一批道路或车辆稳定变好的强信号。

.. image:: images/reward_position_heatmap.png
   :alt: reward position heatmap
   :align: center

rho 位置映射变化
~~~~~~~~~~~~~~~~

``rho`` 原本是 individual 级指标，并不是每个 car-road 位置单独一个值。这里将同一个
individual 的 ``rho`` 广播到它的有效 car-road 位置后再求平均，因此该图主要用于观察
``rho`` 提升覆盖了哪些有效区域，而不能直接解释为某个道路位置本身产生了 ``rho``。
差值图大面积为正，说明 final population 的 ``rho`` 提升在有效位置上普遍可见；同时横向差异较弱，
说明提升不是由某一个 ``road_index`` 列单独贡献。

.. image:: images/rho_position_heatmap.png
   :alt: rho broadcast position heatmap
   :align: center
