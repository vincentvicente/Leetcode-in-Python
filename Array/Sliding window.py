from typing import List

"""
明确左右指针指向，然后去动态的移动两个指针。
一般：
右指针指向window的闭端，左指针指向开端
"""

# brute force
# n = len(nums)
# if n < k:
#     return 0
#
# res = 0
# max_num = max(nums)
# for r in range(k - 1, n):
#     l = 0
#     while r - l + 1 >= k:
#         if nums[l:r + 1].count(max_num) >= k:
#             res += 1
#         l += 1
#
# return res
#
# linear scan
"""
卡壳点：理解完算法后，如何移动l指针，使得其指向当前情况下倒数第二个最大值的位置
解决：思考左指针右移后的终止条件
"""


def countSubarrays(self, nums: List[int], k: int) -> int:
    n = len(nums)
    if n < k:
        return 0

    res, l = 0, 0
    max_cnt, max_num = 0, max(nums)
    # r指针动态指向“刚好”满足k时候的index
    for r in range(n):
        if nums[r] == max_num:
            max_cnt += 1
        # l指针指向r情形下，k - 1时候的index
        while (max_cnt > k) or (l <= r and max_cnt == k and nums[l] != max_num):
            if nums[l] == max_num:
                max_cnt -= 1

            l += 1

        if max_cnt == k:
            res += l + 1

    return res
