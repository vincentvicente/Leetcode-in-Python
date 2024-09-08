# Back Track

## 本质：
N叉决策树的收集，或者遍历
```commandline
def backtrack (可选择列表，当前路径)：
    if 当前路径 符合条件:
        res.add(当前路径)
    
    for 选择 in 可选择列表：
        当前路径.add(选择)
        更新可选择列表
        backtrack()
        当前路径.pop()
        可选择列表回溯
```
## 题目类型
* Combination 排列
* Permutation 组合
![LeetCode algorithm.jpeg](..%2F..%2F..%2F..%2FDownloads%2FLeetCode%20algorithm.jpeg)

