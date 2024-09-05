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
    cache = {} # (n, k) -> count

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

