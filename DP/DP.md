## DP核心方法以及思路
基本步骤：
1. 确定dp数组及其下标的含义
2. 确定递推公式
3. dp的初始化
4. 确定遍历顺序

### 结合决策树一步步优化
1. `eg: 斐波那契数列` 
```
def fib(n):
    if n == 1 or n == 2:
        return n
    else:
        return fib(n - 1) * fib(n - 2)
```
#### 时间复杂度高：2^n, 重复计算多（fib(3) = fib(2) + fib(1), fib(4) = fib(3) + fib(2)）

优化1: 思路：减少重复计算，通过memoization记录下之前计算过的fib数
```
哈希表
def fib(n):
    memo = {}
    if n == 1 or n == 2:
        return 1
    if n in memo:
        return memo[n]
    memo[n] = fib(n - 1) + fib(n - 2)
    return memo[n]
```

```
数组
def fib(n):
    dp = [0] * len(n + 1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```

进一步优化：数组
只需要存储两个值去记录
空间复杂度：n -> 1
```
def fib(n):
    dp_1 = 1
    dp_2 = 1
    for i in range(2, len(n)):
        dp_i = dp_1 + dp_2
        dp_1 = dp_2
        dp_2 = dp_i
    return dp_i
```

### 
`eg: integer break`\
拆分给定的整数，使其拆分的积最大

```
def integerBreak(n):
    # dp[i]存储i这个数拆分的最大积
    dp = [0] * (n + 1) # initialize
    for i in range(3, n + 1):
        for j in range(1, i):
            dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
    
    return dp[n]
```
intuition:
需要将整数至少拆分成两个正整数
拆分正整数i
* 情况1：拆成俩数：j, i - j
* 2: 拆成多个数：j, dp[i - j]（存储的是(i - j)可拆分的最大积）
比较两种情况下的较大值，不断更新dp[i]
