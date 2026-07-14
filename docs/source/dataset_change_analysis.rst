数据集变化分析
==============

本页分析 ``prediction_model/dataset/initial_population.pkl`` 到
``prediction_model/dataset/final_population.pkl`` 的总体变化，并进一步回答一个道路级问题：
**改变某条道路的奖励 R，会怎样影响 rho？**

分析代码和输出均位于 ``prediction_model``。数据生成和反事实评估调用隔壁 ``MASDiff``
中的 DQN、SUMO 与 rho 计算流程，但没有修改 ``MASDiff`` 代码。

结论摘要
--------

1. final population 的 ``rho`` 均值从 ``0.024489`` 上升到 ``0.025914``；按
   :math:`\rho=1/(1+MSE)` 反推，平均 MSE 从 ``40.34`` 下降到 ``37.80``。
2. ``reward_mean`` 仅上升 ``0.000319``，``tau0_mean`` 上升 ``0.0662``，
   ``tau1_mean`` 下降 ``0.0068``。整体输入分布几乎没有迁移，rho 的改善主要来自真实仿真评估后的
   top-M 筛选，而不是 reward 全局抬升。
3. 观察性道路筛选给出的前十道路是 ``12, 39, 53, 104, 16, 67, 24, 92, 56, 90``；
   它们只是与 rho 相关的候选，不能解释为因果重要性。
4. 对最优 final individual 的 114 条道路分别施加 ``R[:, road] += -0.5`` 和 ``+0.5``，
   重训 DQN、重跑 SUMO，共完成 ``228`` 次干预。按平均绝对 rho 效应排序，前十道路为
   ``45, 53, 15, 44, 54, 14, 16, 47, 112, 87``。
5. “重要”只表示改变该道路后 rho 变化较大，不表示增大奖励一定有利。前十道路中只有道路 ``44``、
   ``14`` 呈现“减小 R 使 rho 下降、增大 R 使 rho 上升”的预期方向；多条道路表现出反向或非单调响应。
6. 当前全道路实验只有 ``1`` 个 individual、每个干预条件 ``1`` 次运行。预实验与全道路实验在
   ``12/24/67`` 上存在明显波动，所以现有排序应视为**单个个体上的初步因果敏感度**，
   还不能作为种群级道路因果排名。

分析口径
--------

initial 与 final 不是逐行父子关系。MASDiff 迭代会选择 elite、对 reward 做扩散变异、重新训练 DQN、
重新运行 SUMO、计算 rho，再把旧 population 与 mutants 合并并保留 rho 最大的 top-M。
因此 ``final - initial`` 只描述两个种群的分布变化，不能把第 i 行差值解释为第 i 个个体的处理效应。

本页使用三层证据，结论强度依次提高：

1. **种群分布分析**：说明 final 整体变成了什么样。
2. **观察性道路筛选**：从相关系数、回归系数和父子元数据中找候选道路。
3. **道路反事实干预**：只改变一条道路的 R，并重新执行完整 DQN-SUMO-rho 流程。

种群发生了什么变化
--------------------

initial/final 各包含 ``200`` 个 individual。rho 分布整体右移，均值提高 ``0.001424``；
换算到误差空间后，平均 MSE 降低约 ``2.54``。

.. image:: images/rho_distribution.png
   :alt: initial and final rho distribution
   :align: center

.. image:: images/mse_est_inv1p_distribution.png
   :alt: MSE estimated from rho
   :align: center

reward 与 tau 的全局统计变化远小于 rho。final 中 ``65`` 个 mutant 带有 ``parent_rho``，
其中 ``56.9%`` 的 rho 高于父代，平均 ``delta_parent_rho`` 为 ``0.001317``。
这说明局部变异能产生更好候选，但改善不是稳定发生在所有 mutant 上。

.. image:: images/distribution_change_bar.png
   :alt: standardized population changes
   :align: center

完整 ``tau0/tau1/reward/valid`` 张量经过特征哈希后做 PCA，前两维解释率仅为 ``4.31%`` 和
``3.13%``。initial 与 final 大量重叠，按 rho 着色也没有清晰单调方向。

.. image:: images/highdim_pca_population_change.png
   :alt: PCA of complete high-dimensional population features
   :align: center

因此，这批数据更符合“在相近输入中筛出少量闭环表现更好的 policy”，而不是“整个 reward/tau
分布移动到一个全新区域”。这也是只用全局统计预测 rho 容易接近常数的原因：有效信号更可能位于
局部 action ranking、DQN 训练过程和 SUMO 时序中。

位置热力图
----------

热力图横轴是 ``road_index``，纵轴是 ``car_index``。每个格子是该位置在 200 个 individual
中的平均值，三个子图依次为 initial、final 和 ``final - initial``。

``tau0_distance`` 基本不变，因为路网与目的地没有改变；``tau1_queue`` 只有局部正负变化，
它是车辆首次出现时的排队快照，并不是 rho 所评价的完整仿真时序。

.. image:: images/tau0_distance_position_heatmap.png
   :alt: tau0 distance position heatmap
   :align: center

.. image:: images/tau1_queue_position_heatmap.png
   :alt: tau1 queue position heatmap
   :align: center

reward 差值正负交错，没有稳定的整列道路或整段车辆同步抬升。这个图能指出哪里发生了变化，
但不能回答某个变化是否导致 rho 改变。

.. image:: images/reward_position_heatmap.png
   :alt: reward position heatmap
   :align: center

道路观察性筛选
--------------

``prediction_model/analyze_road_importance.py`` 先对每条道路的有效 car 位置求均值，再综合：

* final 道路 reward 与 rho 的 Pearson/Spearman 相关；
* 多道路 ridge 回归系数；
* 65 个 mutant 的 child reward 与 ``delta_parent_rho`` 关系；
* initial/final 行对齐差值，仅作为诊断，不作为父子因果证据。

综合筛选前十道路为 ``12, 39, 53, 104, 16, 67, 24, 92, 56, 90``；
基于 mutant-parent delta 的前十道路为 ``24, 12, 67, 47, 106, 94, 112, 54, 65, 9``。
两类观察指标共同指向 ``12, 24, 67``，所以它们被用于第一轮反事实预实验。

.. image:: images/road_importance_screening_top_roads.png
   :alt: observational road importance screening
   :align: center

这里的分数用于缩小搜索范围。道路 reward 可能同时受 elite 来源、tau、有效位置、其他道路 reward
以及策略行为影响，所以相关性高不等于“干预这条道路一定改变 rho”。

道路反事实实验
--------------

实验设计
~~~~~~~~

``prediction_model/causal_road_intervention.py`` 对选定 individual 和 road 执行：

.. code-block:: text

   原始 reward
       -> 只修改 R[:, road_index]
       -> 重建 DQN 训练数据
       -> 重新训练 DQN
       -> 重新运行 SUMO
       -> 重新计算 rho

全道路实验选择 final population 中 rho 最高的 ``individual_index=0``，对 ``0..113`` 共 114 条道路
分别执行 ``-0.5`` 和 ``+0.5`` 加法干预，共得到 ``228`` 个结果。

实验必须使用重新计算的 baseline。该 individual 在 pkl 中保存的 rho 是 ``0.034557``，
但用原始 reward 重新训练和仿真得到的 baseline rho 是 ``0.024582``，两者相差 ``0.009975``。
因此处理效应统一定义为：

.. math::

   effect(r, \delta) = \rho\!\left(do(R_{:,r} \leftarrow R_{:,r}+\delta)\right)
                       - \rho_{recomputed\ baseline}

若直接减去 pkl 中的 stored rho，训练与仿真环境差异会被误算成道路效应。

全道路结果
~~~~~~~~~~

道路重要性定义为两个干预方向绝对效应的均值：

.. math::

   importance(r) = \frac{|effect(r,-0.5)| + |effect(r,+0.5)|}{2}

该指标衡量敏感度，不保留“增大奖励是好还是坏”的方向信息。全部 228 次干预的绝对效应均值为
``0.002272``，中位数为 ``0.002064``，90 分位数为 ``0.004855``，最大值为 ``0.008665``。

.. image:: images/road_causal_importance_ranking.png
   :alt: causal road sensitivity ranking for individual 0
   :align: center

排名前十道路及其方向如下。正值表示干预后 rho 变大，负值表示 rho 变小。

.. list-table:: 单个最优 individual 的道路因果敏感度前十
   :header-rows: 1
   :widths: 12 22 22 22 22

   * - road
     - effect(-0.5)
     - effect(+0.5)
     - mean_abs_effect
     - 响应形态
   * - 45
     - +0.005361
     - -0.005792
     - 0.005577
     - 反向
   * - 53
     - +0.003232
     - +0.007768
     - 0.005500
     - 两侧均改善
   * - 15
     - +0.005451
     - +0.005213
     - 0.005332
     - 两侧均改善
   * - 44
     - -0.001117
     - +0.008665
     - 0.004891
     - 预期方向
   * - 54
     - +0.006490
     - -0.002300
     - 0.004395
     - 反向
   * - 14
     - -0.006122
     - +0.002650
     - 0.004386
     - 预期方向
   * - 16
     - +0.006092
     - +0.002603
     - 0.004347
     - 两侧均改善
   * - 47
     - +0.003474
     - -0.005051
     - 0.004262
     - 反向
   * - 112
     - +0.005330
     - -0.003054
     - 0.004192
     - 反向
   * - 87
     - +0.004708
     - -0.003556
     - 0.004132
     - 反向

.. image:: images/road_intervention_effects.png
   :alt: signed effects for all road interventions
   :align: center
   :width: 100%

全部 114 条道路中，有 ``28`` 条呈现“减小 R 后 rho 下降、增大 R 后 rho 上升”的方向；
``33`` 条方向相反，``15`` 条在正负干预下都改善，``38`` 条在正负干预下都变差。
这表明 R 到 rho 不是简单的逐道路单调关系。改单路 R 会改变动作相对排序，随后经过 DQN 优化和
SUMO 网络传播，影响可能转移到其他道路和后续时刻。

观察筛选与因果结果的关系
~~~~~~~~~~~~~~~~~~~~~~~~

观察性综合前十与因果敏感度前十只重合 ``53`` 和 ``16``；mutant-parent 前十还命中 ``47``、
``54``、``112``。这说明观察筛选能找到部分候选，但不能替代闭环干预。

第一轮 ``12/24/67`` 预实验中，道路 12 和 24 表现为增大 R 改善 rho、减小 R 降低 rho；
但在全道路实验中，道路 12 的因果敏感度只排第 ``103``，道路 24 排第 ``50``，道路 67 排第 ``55``。
其中道路 67 的 ``-0.5`` 效应还从 ``+0.002427`` 变为 ``-0.003271``。重复运行间的差异说明
DQN 初始化、训练过程或 SUMO 运行噪声与当前道路效应处于相近量级。

因此，当前实验支持以下结论：

* 某些道路干预能使 rho 明显变化，最大观测变化约 ``0.0087``；
* 道路 ``45/53/15/44/54`` 是该最优 individual 上值得优先复验的高敏感候选；
* 道路重要性高度依赖干预方向、individual 和随机训练结果；
* 目前不能断言这些道路在整个 population 上具有稳定因果效应。

为什么数据这样变化
------------------

总体结果与道路实验可以统一解释为：

1. rho 是完整闭环结果，reward 是中间训练信号。全局 reward 均值几乎不变，局部动作排序仍可能改变。
2. top-M 保留直接按真实 rho 筛选，所以 final rho 能上升，即使 PCA 和均值统计仍与 initial 高度重叠。
3. 原始 rho 范围很窄。``softmax(temperature*rho)`` 在 ``temperature=1.5`` 时有效样本比例约
   ``0.99999``，elite 采样近似均匀，降低了高质量父代被重复变异的概率。
4. 单路 reward 干预经过 DQN 与路网耦合后会产生非局部、非线性响应，不能用 reward 均值或单变量相关解释。
5. 当前反事实运行的随机方差尚未被重复实验分离，部分“道路效应”可能包含训练和仿真噪声。

.. image:: images/selection_pressure.png
   :alt: softmax selection pressure
   :align: center

改进与下一轮实验
----------------

1. **先建立可重复的因果估计。** 对每个 ``(individual, road, delta)`` 使用相同 seed 的成对 baseline
   和 intervention，并至少运行 ``5`` 个 seed；报告均值、标准差、95% 置信区间和符号一致率。
2. **扩展到代表性种群。** 至少选择 top-rho、median-rho、low-rho 各若干 individual，使用分层平均
   得到 population-level ATE，同时保留 individual-level CATE，避免只对最优个体下结论。
3. **增加剂量曲线。** 对候选道路使用 ``-1.0, -0.5, -0.25, 0.25, 0.5, 1.0``，区分单调、阈值、
   U 型和饱和响应；道路优化应依据方向和剂量曲线，而不是绝对效应排名。
4. **把道路交互纳入实验。** 对稳定的 top-k 道路做两两干预，估计
   :math:`effect(r_i,r_j)-effect(r_i)-effect(r_j)`，识别替代路线与拥堵传播造成的协同或抵消。
5. **改进 elite 选择尺度。** 使用 rank-based selection、标准化 rho、``-MSE`` 或 ``delta_rho``，
   使 elite 采样真正偏向高质量个体。
6. **保存完整 lineage 和仿真特征。** 记录 parent/candidate/seed、是否进入 top-M，以及每条道路的平均队列、
   峰值、拥堵持续时间和分路段 MSE，才能解释“哪条路通过什么机制改变 rho”。
7. **让代理模型学习效应和排序。** 优先预测 ``delta_rho``、改善概率和 top-k 排名，并使用 sign accuracy、
   Spearman、top-k precision 评估；输入应保留 car-road 局部结构和道路 embedding。

复现实验
--------

总体变化分析：

.. code-block:: powershell

   D:\anaconda3\envs\Crowd\python.exe prediction_model\analyze_population_change.py

观察性道路筛选：

.. code-block:: powershell

   D:\anaconda3\envs\Crowd\python.exe prediction_model\analyze_road_importance.py

单个最优 individual 的全道路反事实实验：

.. code-block:: powershell

   D:\anaconda3\envs\Crowd\python.exe prediction_model\causal_road_intervention.py `
     --all-roads --top-n-by-rho 1 --mode add --values -0.5 0.5 `
     --recompute-baseline --resume `
     --output-dir dataset\road_causal_intervention_all_roads_ind0

主要结果表位于 ``road_intervention_results.csv``、``road_intervention_summary.csv`` 和
``road_causal_importance.csv``。其中 ``effect_vs_baseline`` 是应使用的处理效应列。
