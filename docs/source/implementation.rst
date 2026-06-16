实施建议
========

优先实现版本
------------

建议先实现稳定、可验证的基础版本：

- Road Transformer Encoder；若已有 SUMO 路网拓扑，则优先 Road Graph Encoder。
- DeepSets 或轻量 Set Transformer。
- 统计特征 MLP 分支。
- Student-t NLL 损失。
- MC Dropout 或 5-model Deep Ensemble。

如果测试集表现显示 ``rho`` 的条件预测分布明显多峰，再升级为 Mixture Density Network。
这样可以先保证训练稳定和泛化可靠，再逐步增强模型表达能力。

推荐超参数
----------

.. list-table::
   :header-rows: 1

   * - 参数
     - 推荐值
   * - ``hidden_dim``
     - 128
   * - ``road_encoder_layers``
     - 3
   * - ``set_encoder_layers``
     - 2
   * - ``attention_heads``
     - 4
   * - ``dropout``
     - 0.1
   * - ``batch_size``
     - 8 到 32，视显存和 ``tau`` 尺寸而定
   * - ``optimizer``
     - AdamW
   * - ``learning_rate``
     - ``1e-4`` 到 ``3e-4``
   * - ``weight_decay``
     - ``1e-4``
   * - ``max_epochs``
     - 200
   * - ``early_stopping_patience``
     - 20
   * - ``gradient_clip_norm``
     - 1.0
   * - ``likelihood``
     - Student-t

合理性总结
----------

Road Transformer 或 Road Graph Encoder 能建模道路之间的相互关系，
比直接 flatten 后接 MLP 更容易学到可迁移规律。

Set Transformer 或 DeepSets 能减少车辆排列顺序对模型的干扰，
更适合跨样本、跨 agent 数量或跨场景的泛化。

Student-t NLL 或 mixture NLL 可以学习
:math:`p(\rho \mid \tau, rewards)`，而不是只学习一个点估计，
因此能同时给出预测值和可信区间。

统计特征分支提供稳健的全局信息，深层编码器提供复杂车辆-道路交互模式。
两者融合能兼顾准确性和泛化性。
