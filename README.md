#### main 分支
目前 acc 达到 0.8384, 先不要把 eyu 分支合并过来
#### eyu 分支
- 在 main 分支基础上优化了 prompt 格式以及加入了更多 stop token
- 加入了 `gen_subq`, `gen_remain_subq`, `gen_remain_steps` 三个新的动作, 
  - 其中 gen_remain 类型限制只能生成一个节点, 且同一条路径仅能有 1 个用此动作生成的节点
- 等待测试
#### eyu1 分支
- 将细化并固定思考步骤, 考虑使用代码生成领域的思想比如考虑输入输出类型, 考虑边界情况之类的
- 先在 direct answer 部分加入生成测试样例的筛选方法
