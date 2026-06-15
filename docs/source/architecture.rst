模型架构
========

总体结构
--------

推荐架构由四部分组成：

1. 车辆-道路局部编码器。
2. 车辆集合级聚合器。
3. 全局统计特征分支。
4. 贝叶斯预测头。

输入张量约定如下：

- ``tau``: ``[B, N_vehicle, N_road, D_tau]``
- ``rewards``: ``[B, N_vehicle, N_road]``
- ``mask``: ``[B, N_vehicle, N_road]``

其中 ``B`` 是 batch size，``N_vehicle`` 是车辆数量，``N_road`` 是道路数量。
例如 ``114`` 表示道路维度，不表示时间维度。

车辆-道路局部编码
--------------------

对每辆车 ``i`` 和每条道路 ``j``，构造局部特征：

.. code-block:: text

   x_i,j = concat(
       tau_i,j,
       rewards_i,j,
       road_id_embedding_j,
       valid_mask_i,j
   )

其中 ``road_id_embedding_j`` 是可学习道路 ID 嵌入；如果已有道路拓扑信息，
也可以替换或补充为 road topology embedding。

Road Transformer Encoder
~~~~~~~~~~~~~~~~~~~~~~~~

当暂时没有 SUMO 路网拓扑时，优先使用 Road Transformer：

- Linear projection: ``input_dim -> hidden_dim``
- Road ID embedding: learnable
- Transformer Encoder: 3 到 4 层
- ``hidden_dim``: 128 或 256
- attention heads: 4 或 8
- dropout: 0.1
- feedforward_dim: ``4 * hidden_dim``

每辆车的道路表示经过 masked attention pooling 得到车辆级表示：

.. math::

   h_i = AttentionPool(\{z_{i,j}\}_{j=1}^{N_{road}}, mask_i)

Road Graph Encoder
~~~~~~~~~~~~~~~~~~

如果已知道路之间的连接关系，可使用 GNN 或 GAT 编码道路拓扑：

- 节点为道路。
- 节点特征为该车在该道路上的 ``tau`` 和 ``reward`` 特征。
- GAT layers: 2 到 3 层。
- ``hidden_dim``: 128。
- 最后对道路节点做 attention pooling，得到车辆级表示 ``h_i``。

有可靠路网图时，Road Graph Encoder 通常具备更强归纳偏置；没有拓扑时，
Road Transformer 更容易落地，也能学习道路之间的相互影响。

集合级聚合
----------

车辆维度没有固定强语义顺序，因此聚合器应尽量满足排列不变性。

推荐使用 Set Transformer：

- 输入：``{h_i}_{i=1..N}``
- Induced Set Attention Blocks: 2 层
- ``num_inducing_points``: 16 或 32
- ``hidden_dim``: 128 或 256
- Multihead attention heads: 4
- Pooling by Multihead Attention 得到 ``h_global``

轻量版本可使用 DeepSets：

.. math::

   h_{global} = MLP(mean_i \phi(h_i), max_i \phi(h_i), std_i \phi(h_i))

当数据量较小时，DeepSets 更简单，也更不容易过拟合。

统计特征分支
------------

统计特征 ``s`` 通过 MLP 得到 ``h_stats``：

.. code-block:: text

   Linear -> LayerNorm -> GELU -> Dropout
   Linear -> GELU

最终表示为：

.. math::

   h = concat(h_{global}, h_{stats})

融合层可采用：

- MLP: ``hidden_dim * 2 -> 256 -> 128``
- activation: GELU
- normalization: LayerNorm
- dropout: 0.1

贝叶斯预测头
------------

基础版本采用 Student-t likelihood head：

.. code-block:: text

   mu = Linear(h, 1)
   log_sigma = Linear(h, 1)
   log_nu = Linear(h, 1)
   sigma = softplus(log_sigma) + 1e-6
   nu = softplus(log_nu) + 2.0

预测分布为：

.. math::

   \rho \mid \tau, rewards \sim StudentT(\nu, \mu, \sigma)

Student-t 对异常 ``rho`` 或高噪声样本更稳健，也比 Gaussian NLL 对离群点更不敏感。

如果 ``rho`` 后验可能明显多峰，可升级为 Mixture Density Network：

.. math::

   p(\rho \mid \tau, rewards) =
   \sum_k \pi_k StudentT(\rho; \nu_k, \mu_k, \sigma_k)

建议 ``K`` 先取 3 或 5。数据量较小时先使用 ``K=1``，避免过拟合。
