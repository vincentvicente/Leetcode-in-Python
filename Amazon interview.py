from typing import List


## Two sum
def twoSum(nums: List[int], target: int) -> List[int]:
    ## O(n)
    hashMap = {}  # val: index
    for i, num in enumerate(nums):
        diff = target - num
        if diff in hashMap:
            return [hashMap[diff], i]
        hashMap[num] = i

    ## O(n²)
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]


## Best buy and sell stock
def maxProfit(prices: List[int]) -> int:
    l, r = 0, 1
    maxPro = 0

    while r < len(prices):
        if prices[l] < prices[r]:
            pro = prices[r] - prices[l]
            maxPro = max(maxPro, pro)
        else:
            l = r
        r += 1

    return maxPro


## Merge intervals
def mergeIntervals(intervals):
    intervals.sort(key=lambda i: i[0])
    res = [intervals[0]]

    for start, end in intervals[1:]:
        lastEnd = res[-1][1]
        if lastEnd < start:
            res.append([start, end])
        else:
            res[-1][1] = max(lastEnd, end)

    return res
