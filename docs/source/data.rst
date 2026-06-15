数据与预处理
============

数据划分
--------

为了验证泛化性，不能只做随机打乱划分。推荐按个体、场景或生成来源划分：

.. list-table::
   :header-rows: 1

   * - 数据集
     - 比例
     - 用途
   * - 训练集
     - 70%
     - 学习模型参数
   * - 验证集
     - 15%
     - 调参、早停和选择模型结构
   * - 测试集
     - 15%
     - 最终报告泛化性能

如果数据来自不同 ``round``、``candidate``、agent 数量、seed 或仿真配置，
应优先使用 group split。例如按 ``round`` 或 ``source`` 分组，
确保某些组只出现在测试集中。这样可以降低数据泄漏风险，更接近未见过场景的评估。

预处理
------

推荐预处理步骤如下：

1. 对 ``tau`` 中的非有限值进行修复或填充，并保存 ``valid_mask``。
2. ``tau`` 按特征维做标准化，只使用训练集统计量计算均值和方差。
3. ``rewards`` 可做 robust scaling，例如减中位数、除以 IQR。
4. ``rho`` 可做 z-score、logit 或 ``log(rho + eps)`` 变换。
5. 所有标准化参数必须从训练集拟合，并原样应用到验证集和测试集。

全局统计特征
------------

除深度编码器外，建议保留每个样本的统计特征，作为稳健的全局分支输入：

- rewards: mean、std、min、max、median、quantiles、positive ratio、saturation ratio。
- tau distance-to-destination: mean、std、min、max、quantiles、zero ratio、valid ratio。
- tau queue length: mean、std、min、max、quantiles、congestion ratio、valid ratio。
- per-vehicle: best-road reward、top-k reward mean、top-1 与 top-2 reward gap。
- mask: missing ratio、nonfinite repaired ratio。

这些统计量可帮助模型获得低阶稳定信息，减少完全依赖深层网络造成的过拟合。
