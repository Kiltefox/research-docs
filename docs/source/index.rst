Rho 预测模型设计文档
====================

本文档描述一个从 ``tau`` 与奖励矩阵 ``rewards`` 预测 ``rho`` 的概率模型设计。
设计目标不是只输出单点回归值，而是学习条件预测分布
:math:`p(\rho \mid \tau, rewards)`，并同时给出预测均值与不确定性。

该设计面向 SUMO 等交通仿真环境，其中 ``tau`` 表示车辆-道路结构特征，
例如每辆车到各道路终点的距离和道路排队长度；``rewards`` 表示每辆车对每条道路的奖励。
核心方案采用道路编码器、车辆集合聚合、全局统计分支和概率预测头。

在本项目的数据生成方向中，``tau`` 和 ``rewards`` 通过仿真产生 ``rho``。
因此本文档将 :math:`p(\rho \mid \tau, rewards)` 称为条件预测分布；
除非额外建立 ``rho`` 生成 ``tau`` 与 ``rewards`` 的反向概率模型，否则不严格称为后验分布。

文档目录
--------

.. toctree::
   :maxdepth: 2
   :caption: 模型设计

   overview
   data
   architecture
   training
   implementation
   history

.. toctree::
   :hidden:

   usage

.. note::

   本文档整理自 ``rho_prediction_model_design.txt``，作为 Read the Docs/Sphinx 版本的模型设计说明。
