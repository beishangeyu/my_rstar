#### main 分支
目前 acc 达到 0.8384, 先不要把 eyu 分支合并过来
#### eyu 分支
- 在 main 分支基础上优化了 prompt 格式以及加入了更多 stop token
- 加入了 `gen_subq`, `gen_remain_subq`, `gen_remain_steps` 三个新的动作, 
  - 其中 gen_remain 类型限制只能生成一个节点, 且同一条路径仅能有 1 个用此动作生成的节点
- 等待测试
#### eyu1 分支
- 去除子问题相关动作, 替换为指导模型用特定角度分解问题
#### eyu2 分支
- 将重述问题变得更为具体, 视eval acc而定考虑是否加入代码克隆工具
  - 但是就算加入也会在筛选正确路径时加入, gene部分不会加入代码克隆检查工具
