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

"""
找到s中最短的substring，使得t中的每一个元素都存在在这个substring里
返回这个最短的子字符串，注意不是返回长度
"""

def minWindow(s, t):
    # edge cases
    m, n = len(s), len(t)
    if m < n:
        return ""

    # variables we need
    l, r = 0, 0 # record the dynamic window, l - > shrink the window while find a valid one; r - > expand the window until we find a valid one
    min_len = float('inf')
    window, need = {}, {}
    valid = 0 # record the type in need
    start = 0 # record the curr min window's starting index

    # finish our need map
    for char in t:
        need[char] = need.get(char, 0) + 1

    while r < len(s):
        c = s[r]
        # find a char we need, update the window
        if c in need:
            window[c] = window.get(c, 0) + 1
            # tell if it satisfies one type
            if window[c] == need[c]:
                valid += 1

        # find a valid window, shrink the left pointer to find a smaller one
        while valid == len(need):
            # tell if curr sub is a valid window, if so, update the start point
            if r - l + 1 < min_len:
                min_len = r - l + 1
                start = l

            ch = s[l]
            # char at l is the one in need
            if ch in need:
                window[ch] -= 1
                if window[ch] < need[ch]:
                    valid -= 1

            l += 1

        r += 1

        return "" if min_len == float('inf') else s[start:start + min_len]


"""
给你两个字符串 s1 和 s2 ，写一个函数来判断 s2 是否包含 s1 的排列。如果是，返回 true ；否则，返回 false 。即s2的子字符串是s1的一个排列
"""
def checkInclusion(s1, s2):
    m, n = len(s1), len(s2)
    if m > n:
        return False

    need, window = {}, {}
    l, r = 0, 0
    for ch in s1:
        need[ch] = need.get(ch, 0) + 1
    valid = 0
    while r < n:
        # window needs to expand, move the right pointer
        c = s2[r]
        r += 1
        if c in need:
            window[c] = window.get(c, 0) + 1
            if window[c] == need[c]:
                valid += 1
        # window needs to shrink, move the left pointer
        while r - l >= m:
            if len(need) == valid:
                return True

            cha = s2[l]
            l += 1
            if cha in need:
                if window[cha] == need[cha]:
                    valid -= 1
                window[cha] -= 1

    return False


# fast pointer -> traverse the array
# slow pointer -> modify the index of the array
def compress(self, chars: List[str]) -> int:
    n = len(chars)
    if n == 1:
        return 1

    s, f = 0, 0

    while f < n:
        cur_char = chars[f] # cur char
        # calculate the number of char
        cnt = 0
        while f < n and cur_char == chars[f]:
            cnt += 1
            f += 1

        # modify on the index
        chars[s] = cur_char
        s += 1

        if cnt > 1:
            for c in str(cnt):
                chars[s] = c
                s += 1

    return s








