from typing import List, Optional


def findTargetSumWays(nums: List[int], target: int) -> int:
    dp = {}

    def backtrack(i, total):
        if i == len(nums):
            return 1 if total == target else 0
        if (i, total) in dp:
            return dp[(i, total)]

        dp[(i, total)] = backtrack(i + 1, total + nums[i]) + backtrack(i + 1, total - nums[i])

        return dp[(i, total)]

    return backtrack(0, 0)


"""
Cannot rob adjacent houses
"""


def rob1(nums: List[int]) -> int:
    rob1, rob2 = 0, 0
    for n in nums:
        tmp = max(rob1 + n, rob2)
        rob1 = rob2
        rob2 = tmp
    return rob2


"""
All houses are arranged in a circle, which means that the first and the last houses are connected
"""


def rob2(nums: List[int]) -> int:
    def helper(nums):
        rob1, rob2 = 0, 0
        for n in nums:
            temp = max(rob1 + n, rob2)
            rob1 = rob2
            rob2 = temp
        return rob2

    return max(nums[0], helper(nums[1:]), helper(nums[:-1]))


"""
All houses are arranged in a binary tree
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def rob3(root: Optional[TreeNode]) -> int:
    def dfs(root):
        if not root:
            return [0, 0]
        leftPairs = dfs(root.left)
        rightPairs = dfs(root.right)
        withRoot = root.val + leftPairs[1] + rightPairs[1]
        withoutRoot = max(leftPairs) + max(rightPairs)

        return [withRoot, withoutRoot]

    return max(dfs(root))


def fibonacci(n):
    if n == 0 or n == 1:
        return 1

    return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci2(n):
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1

    dp = {}
    if n in dp:
        return dp[n]

    dp[n] = fibonacci2(n - 1) + fibonacci2(n - 2)
    return dp[n]


"""
Find the most profit, schedule from startTime to endTime
Using dp to store the previous largest profit
"""


def jobScheduling(self, startTime, endTime, profit):
    intervals = sorted(zip(startTime, endTime, profit))
    cache = {}

    def dfs(i):  # index i is the tracking of the most profit, while j is the nex
        if i == len(intervals):
            return 0

        if i in cache:
            return cache[i]

        res = dfs(i + 1)

        j = i + 1
        while j < len(intervals):
            if intervals[j][0] >= intervals[i][1]:
                break

            j += 1

        cache[i] = res = max(res, intervals[i][2] + dfs(j))
        return res

    return dfs(0)


"""
For an integer array nums, an inverse pair is a pair of integers [i, j],
where 0 <= i < j < nums.length and nums[i] > nums[j].
Given two integers n and k, return the number of different arrays consisting of numbers from 1 to n
such that there are exactly k inverse pairs. Since the answer can be huge, return it modulo 109 + 7.
"""


def kInversePairs(self, n: int, k: int) -> int:
    MOD = 10 ** 9 + 7
    cache = {}  # (n, k) -> count

    def count(n, k):
        if n == 0:
            return 1 if k == 0 else 0

        if k < 0:
            return 0

        if (n, k) in cache:
            return cache[(n, k)]

        cache[(n, k)] = 0
        for i in range(n):
            cache[(n, k)] = (cache[(n, k)] + count(n - 1, k - i)) % MOD

        return cache[(n, k)]

    return count(n, k)


"""
机器人在m*n的grid中行走，只能向右或者向下，求从左上到右下共有多少种走法
二维数组dp
"""


def uniquePaths(self, m: int, n: int) -> int:
    # create two-dimensional array
    dp = [[1] * n for _ in range(m)]

    for r in range(m):
        for c in range(n):
            dp[r][c] = dp[r - 1][c] + dp[r][c - 1]

    return dp[m - 1][n - 1]


## 优化1:从二维到一维

## 数论：左上到右下一共要走 m + n - 2步。向下：m - 1步

"""
网格中存在障碍，无法通过障碍
"""


def uniquePaths2(self, grid):
    ROWS, COLS = len(grid), len(grid[0])
    dp = [[0] * len(COLS) for _ in range(ROWS)]

    for r in range(ROWS):
        if grid[r][0] == 0:
            dp[r][0] = 1

        else:
            break

    for c in range(COLS):
        if grid[0][c] == 0:
            dp[0][c] = 1

        else:
            break

    for r in range(1, ROWS):
        for c in range(1, COLS):
            if grid[r][c] == 1:
                continue

            else:
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1]

    return dp[ROWS - 1][COLS - 1]

def maximizeProduct(n):
    if n == 2:
        return 1

    if n == 3:
        return 2

    if n == 4:
        return 4

    res = 1
    while n > 4:
        res *= 3
        n -= 3

    res *= n
    return n

