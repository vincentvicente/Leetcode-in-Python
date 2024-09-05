import heapq
from typing import List

"""
数组在内存空间上是连续存在的，即index - data
py中数组是动态数组，自动调整大小
注意：删除数组中的元素，需要动态调整覆盖，使得被删除的元素后所有的元素往前调整
"""
class Array:
    def __init__(self):
        self.size = 0
        self.data = []

    def add(self, val):
        self.data.append(val)
        self.size += 1

    def remove(self, val):
        if val in self.data:
            self.data.remove(val)
            self.size -= 1

        else:
            return

    def get(self, index):
        if index < 0 or index >= self.size:
            return -1

        return self.data[index]

    ## fixed array def __init__(self, size)
    ## dynamic array


def subarraySum(self, nums: List[int], k: int) -> int:
    res = 0
    curSum = 0
    prefixSums = {0: 1}

    for n in nums:
        curSum += n
        diff = curSum - k
        res += prefixSums.get(diff, 0)
        prefixSums[curSum] = 1 + prefixSums.get(curSum, 0)

    return res


# def groupAnagrams(self, strs: List[str]) -> List[List[str]]:

def threeSum(nums: List[int], target: int) -> int:
    nums.sort()
    for i in range(len(nums) - 2):
        l, r = i + 1, len(nums)
        while l < r:
            sum = nums[i] + nums[l] + nums[r]
            if sum < target:
                l += 1
            elif sum == target:
                return [nums[i], nums[l], nums[r]]
            else:
                r -= 1


def kSum(k, start, target, nums):
    quad, res = [], []
    if k != 2:
        for i in range(len(nums) - k + 1):
            if k > start and nums[i - 1] == nums[i]:
                continue
            quad.append(nums[i])
            kSum(k - 1, i + 1, target - nums[i], nums)
            quad.pop()
        return

    l, r = 0, len(nums) - 1
    while l < r:
        if nums[l] + nums[r] > target:
            r -= 1
        elif nums[l] + nums[r] < target:
            l += 1
        else:
            res.append(quad + [nums[l], nums[r]])

    return res


def lengthOfLongestSubstring(s: str) -> int:
    charSet = set()
    l = 0
    res = 0

    for r in range(len(s)):
        while s[r] in charSet:
            charSet.remove(s[l])
            l += 1
        charSet.add(s[r])
        res = max(res, r - l + 1)

    return res


def merge(intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda i: i[0])
    output = [[intervals[0]]]

    for start, end in intervals[1:]:
        lastEnd = output[-1][1]
        if start <= lastEnd:
            output[-1][1] = max(lastEnd, end)
        else:
            output.append([start, end])

    return output


def longestPalindrome(s: str) -> str:
    res = ""
    resLen = 0

    for i in range(len(s)):
        l, r = i, i
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if (r - l + 1) > resLen:
                res = s[l:r + 1]
                resLen = r - l + 1
            l -= 1
            r += 1

        l, r = i, i + 1
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if (r - l + 1) > resLen:
                res = s[l:r + 1]
                resLen = r - l + 1
            l -= 1
            r += 1

    return res


def maxArea(height: List[int]) -> int:
    l, r = 0, len(height) - 1
    maxArea = 0
    while l < r:
        area = min(height[l], height[r]) * (r - 1)
        maxArea = max(maxArea, area)
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1

    return maxArea


"""
Find the positive integer x that there are exactly x numbers greater or equal to x
in array nums. If not, return -1. 
"""


def specialArray(nums):
    prev, i = -1, 0
    total_right_num = len(nums)

    while i < len(nums):
        if total_right_num == nums[i] or prev < total_right_num < nums[i]:
            return total_right_num

        while i + 1 < len(nums) and nums[i] == nums[i + 1]:
            i += 1

        prev = nums[i]
        i += 1
        total_right_num = len(nums) - 1

    return -1


"""
Sliding window
Given an array of positive integers nums and a positive integer target, return the minimal length of a 
subarray whose sum is greater than or equal to target. If there is no such subarray, return 0 instead.

"""


def minSubArrayLen(self, target: int, nums: List[int]) -> int:
    res = float('inf')
    curSum = 0
    start = 0
    for end in range(len(nums)):
        if curSum < target:
            curSum += nums[end]

        while curSum >= target:
            res = min(res, end - start + 1)
            curSum -= nums[start]
            start += 1

    return res if res != float('inf') else 0

    ## brute force


def minSubArrayLen(self, target: int, nums: List[int]) -> int:
    res = float('inf')
    for start in range(len(nums)):
        curSum = 0
        for end in range(start, len(nums)):
            curSum += nums[end]
            if curSum >= target:
                res = min(res, end - start + 1)
                break
    return res if res != float('inf') else 0


"""
Find the longest subarray which contains only two unique values
"""


# brute force
def totalFruits(self, fruits):
    num_fruits = 0
    for left in range(len(fruits)):
        for right in range(left, len(fruits)):
            basket = set()
            for cur in range(left, right + 1):
                basket.add(fruits[cur])
            if len(basket) <= 2:
                num_fruits = max(num_fruits, right - left + 1)

    return num_fruits


def delElement(self, nums, val):
    # n = len(nums)
    # for i in range(n):
    #     if i == len(n) - 1:
    #         break
    #     if nums[i] == val:
    #         for j in range(i+1, n):
    #             nums[j-1] = nums[j]
    #
    #
    slow, fast = 0, 0

    slow, fast = 0, 0
    while fast < len(nums):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
            fast += 1

        else:
            fast += 1

    while fast < len(nums):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
            fast += 1

        else:
            fast += 1


def trimSpaces(self, s):
    l, r = 0, len(s) - 1
    # 去掉左边多余空格
    while l <= r and s[l] == " ":
        l += 1
    # 去掉右边多余空格
    while l <= r and s[r] == " ":
        r -= 1

    # 减少单词间的多余空格
    res = []
    while l <= r:
        # l所指的是个字母
        if s[l] != " ":
            res.append(s[l])
            l += 1
        # l所指的是空格，但是是有效空格
        elif res[-1] != " ":
            res.append(s[l])
        l += 1

    return res
