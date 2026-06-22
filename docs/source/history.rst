历史
====

.. mermaid::

   graph TD
      base["基础"]
      fix["修复数据集，结果：有一点变化，但还是均值坍缩。"]
      mlp["将模型简化为 MLP，结果：没效果"]
      methods["添加两种聚合方法 meanmax 和 attention，结果：没效果"]

      base --> fix
      fix -->|"简化模型"| mlp
      fix -->|"聚合方法修改"| methods

修复数据集
----------

.. image:: images/1.png
   :alt: 修复数据集示意图 1
   :align: center

.. image:: images/2.png
   :alt: 修复数据集示意图 2
   :align: center
