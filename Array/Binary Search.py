from typing import List

""" while loop: l <= r, when there are odd number of elements, l might be = r
"""


## 核心：每次减少一半的筛选量，O(logn)


def search(self, nums: List[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] == target:
            return mid

        if nums[mid] >= nums[l]:  # left portion sorted
            if target < nums[l] or target > nums[mid]:  # two scenarios out of bound
                l = mid + 1
            else:
                r = mid - 1

        else:  # right portion sorted
            if target > nums[r] or target < nums[mid]:  # two scenarios out of bound
                r = mid - 1
            else:
                l = mid + 1

    return -1


def isBadVersion(m):  # API
    pass


def firstBadVersion(n: int) -> int:
    l, r = 1, n
    while l <= r:
        m = (l + r) // 2
        if isBadVersion(m):
            if not isBadVersion(m - 1):
                return m
            else:
                r = m - 1
        else:
            l = m + 1

    return l


def findMin(nums: List[int]) -> int:
    res = nums[0]
    l, r = 0, len(nums) - 1
    while l <= r:
        if nums[l] < nums[r]:  # already sorted
            res = min(res, nums[l])
            break

        m = (l + r) // 2
        res = min(res, nums[m])
        if nums[m] >= nums[l]:
            l = m + 1
        else:
            r = m - 1

    return res


class solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        left = self.binSearch(nums, target, True)
        right = self.binSearch(nums, target, False)
        return [left, right]

    def binSearch(self, nums, target, leftBias):
        i = -1
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] > target:
                r = m - 1
            elif nums[m] < target:
                l = m + 1
            else:
                i = m
                if leftBias:
                    r = m - 1
                else:
                    l = m + 1
        return i

    def binarySearch(self, nums, target):
        def findLeftMost(nums, target, mid):
            l, r = 0, mid - 1
            while l <= r:
                m = (l + r) // 2
                if nums[m] < target:
                    l = m + 1
                else:
                    r = m - 1
            return l

        def findRightMost(nums, target, mid):
            l, r = mid + 1, len(nums) - 1
            while l <= r:
                m = (l + r) // 2
                if nums[m] > target:
                    r = m - 1
                else:
                    l = m + 1
            return r

        l, r = 0, len(nums)
        while l <= r:
            m = (l + r) // 2
            if nums[m] > target:
                r = m - 1
            elif nums[m] < target:
                l = m + 1
            else:
                start = findLeftMost(nums, target, m)
                end = findRightMost(nums, target, m)
                return [start, end]
        return [-1, -1]


"""
给定两个sorted数组，在O(log min(m, n))的时间复杂度下返回中位数，空间复杂度为常数
"""


def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    ## 找到长度较短的数组
    A, B = nums1, nums2
    if len(A) > len(B):
        A, B = B, A

    l, r = 0, len(A) - 1
    total = len(A) + len(B)
    half = total // 2

    ## 对长度较短的数组进行二分法搜索
    while True:
        i = (l + r) // 2
        j = half - i - 2
        Aleft = A[i] if i >= 0 else float("-infinity")
        Aright = A[i + 1] if (i + 1) < len(A) else float("infinity")
        Bleft = B[j] if j >= 0 else float("-infinity")
        Bright = B[j + 1] if (j + 1) < len(B) else float("infinity")

        ## 如果满足条件，则对数组长度和进行判断是否为奇数或偶数
        if Aleft <= Bright and Bleft <= Aright:
            if total % 2:
                return min(Aright, Bright)
            return (max(Aleft, Bleft) + min(Aright, Bright)) / 2

        ## 如果不满足条件，则进行指针的移动
        elif Aleft > Bright:
            r = i - 1
        else:
            l = i + 1


def maxValue(self, n: int, index: int, maxSum: int) -> int:
    def calculateSum(maxVal):
        # left portion
        if (index + 1) > maxVal:
            leftSum = ((1 + maxVal) * maxVal) // 2 + (index + 1 - maxVal)

        else:
            leftSum = (maxVal - index + maxVal) * (index + 1) // 2

        # right portion
        if (n - index) > maxVal:
            rightSum = (maxVal + 1) * maxVal // 2 + (n - index - maxVal)

        else:
            rightSum = (maxVal + (maxVal - (n - 1 - index))) * (n - index) // 2

        totalSum = leftSum + rightSum - maxVal
        return totalSum

    ## 当发现符合条件的maxVal时候，仍然要继续二分法，力求最大值。当最后一次合法操作后，继续二分法，此时l = maxVal + 1, l > r，跳出循环，
    ## 此时的r依然保留着最后一次合法操作的值，为最大值。
    l, r = 1, maxSum
    while l <= r:
        maxVal = (l + r) // 2
        if calculateSum(maxVal) > maxSum:
            r = maxVal - 1

        else:
            l = maxVal + 1

    return r
