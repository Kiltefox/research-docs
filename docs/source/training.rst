训练与评估
==========

损失函数
--------

任务本质是概率预测，应最大化真实 ``rho`` 在模型后验预测分布下的概率。
主损失推荐使用负对数似然 NLL。

Student-t 输出的主损失为：

.. math::

   L_{nll} = -\log p(\rho_{true} \mid \tau, rewards)

混合分布输出的主损失为：

.. math::

   L_{nll} = -\log \sum_k \pi_k p_k(\rho_{true} \mid \tau, rewards)

该损失能同时学习预测中心和不确定性，比只使用 MSE 更符合贝叶斯预测设定。

辅助目标
--------

CRPS
   用于评估概率预测分布与真实观测之间的距离，可作为验证指标或辅助训练项。

KL 正则
   如果使用 Bayesian Neural Network 或 MC Dropout，可加入权重先验或 KL 正则，
   形成近似 ELBO 训练目标。

排序一致性
   如果 ``rho`` 的相对大小很重要，可加入 pairwise ranking loss：

   .. math::

      L_{rank} = -\log sigmoid((\rho_i - \rho_j)(\mu_i - \mu_j))

综合损失可写为：

.. math::

   L_{total} = L_{nll} + \beta L_{kl} + \lambda_{rank} L_{rank}

泛化训练
--------

推荐数据增强：

- 对 agent 维度随机 permutation。
- 对 ``tau`` 添加小幅高斯噪声。
- 对 ``rewards`` 做轻微缩放或扰动。
- 随机 mask 部分车辆-道路项，提升缺失数据鲁棒性。

推荐正则化：

- dropout: 0.1 到 0.3。
- weight decay: ``1e-4``。
- early stopping based on validation NLL。
- gradient clipping: 1.0。

不确定性估计
------------

可用 MC Dropout 或 Deep Ensemble 估计 epistemic uncertainty。
推荐训练 5 个不同 seed 的模型，最终预测可取混合后验或均值集成，
ensemble 方差可作为模型不确定性的参考。

评估指标
--------

验证集和测试集至少报告：

- NLL：主概率预测指标。
- CRPS：概率分布质量指标。
- MAE：仅作为辅助解释指标，不作为主损失。
- prediction interval coverage：例如 90% credible interval 应覆盖约 90% 的真实 ``rho``。
- rank correlation：当相对排序重要时报告。

训练流程
--------

1. 加载 population 数据，提取每个 individual 的 ``tau``、``rewards``、``rho``。
2. 按 group 划分 train/valid/test，避免数据泄漏。
3. 只用训练集拟合标准化参数。
4. 训练模型，优化 Student-t NLL 或 mixture NLL。
5. 在验证集上监控 NLL、CRPS、MAE、coverage 和 rank correlation。
6. 使用 early stopping 选择验证集 NLL 最优模型。
7. 在测试集上报告最终泛化性能。
