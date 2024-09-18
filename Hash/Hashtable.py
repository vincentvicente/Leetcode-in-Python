import collections
import heapq
from collections import defaultdict, Counter
from typing import List, Any


def groupAnagrams(self, strs: List[str]):
    res = defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord("a")] += 1

        res[tuple(count)].append(s)

    return res.values()


def intersect(self, nums1: List[int], nums2: List[int]) -> None:
    count1, count2 = Counter(nums1), Counter(nums2)
    res = []
    for key in count1 & count2:
        res.extend([key] * min(count1[key], count2[key]))  # 将可迭代对象中的每个元素分别添加到列表中
    return res


def groupStrings(strings: List[str]):
    mp = defaultdict(list)
    for s in strings:
        key = (tuple(ord(c) - ord(s[0])) % 26 for c in s)
        mp[key].append(s)
    return mp.values()


def romanToInt(self, s: str) -> int:
    dic = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000
    }
    res = 0
    for i in range(len(s)):
        if i + 1 < len(s) and dic[s[i]] < dic[s[i + 1]]:
            res -= dic[s[i]]
        else:
            res += dic[s[i]]

    return res


def integerToRoman(num):
    inToRo = [[1, "I"], [4, "IV"], [5, "V"], [9, "IX"],
              [10, "X"], [40, "XL"], [50, "L"], [90, "XC"],
              [100, "C"], [400, "CD"], [500, "D"], [900, "CM"],
              [1000, "M"]]
    res = ""
    for val, sym in inToRo:
        if num // val:  # 当为0的时候停止
            count = num // val
            num = num % val
            res += (sym * count)

    return res


def RomanToInteger(num):
    RoToIn = {"I": 1, "V": 5, "X": 10,
              "L": 50, "C": 100, "D": 500,
              "M": 1000}
    res = 0
    for i in range(len(num)):
        if i + 1 < len(num) and RoToIn[RoToIn[i]] < RoToIn[RoToIn[i + 1]]:
            res -= RoToIn[RoToIn[i]]
        else:
            res += RoToIn[RoToIn[i]]

    return res


def topKFrequent(nums: List[int], k: int) -> List[int]:
    mp = Counter(nums)
    sorted_values = sorted(mp.items(), key=lambda x: x[1], reverse=True)  # items()返回列表，列表里面是键对值的元组
    # sorted(iterable, key, boolean) (可迭代对象，关键字可创建匿名函数lambda，布尔值reverse排序，默认False)
    res = [key for key, value in sorted_values]
    return res[:k]


def reorganizeString(s: str) -> str:
    count = Counter(s)
    maxHeap = [(-cnt, char) for char, cnt in count.items()]  # list里存储的是元组对(频率的负值，字母)
    heapq.heapify(maxHeap)  # 构造最小堆
    res = ""

    pre = None
    while pre or maxHeap:
        if pre and not maxHeap:  # 说明把pre push回去后，堆里剩下的还是pre，即无法使得两个相邻的不一样
            return ""
        cnt, char = heapq.heappop(maxHeap)
        res += char
        cnt += 1

        if pre:
            heapq.heappush(pre, maxHeap)
            pre = None

        if cnt != 0:
            pre = (cnt, char)


def leastInterval(tasks: List[str], n: int) -> int:
    count = Counter(tasks)
    maxHeap = [-cnt for cnt in count.values()]
    heapq.heapify(maxHeap)
    time = 0
    q = collections.deque()

    while maxHeap or q:
        time += 1
        if maxHeap:
            cnt = heapq.heappop(maxHeap) + 1
            if cnt:
                q.append([cnt, time + n])

        if q and q[0][1] == time:
            heapq.heappush(maxHeap, q.popleft()[0])

    return time


def twoSum(self, nums, target):
    mp = {}
    for i, n in enumerate(nums):
        diff = n - target
        if diff in mp:
            return [i, mp[diff]]

        mp[n] = i


def sortedTwoSum(self, nums):
    res = []
    l, r = 0, len(nums) - 1
    while l < r:
        total = nums[l] + nums[r]
        if total == 0:
            res.append([l, r])
        elif total < 0:
            l += 1
        else:
            r -= 1

    return res


def threeSum(self, nums):
    res = []
    nums.sort()
    for i in range(len(nums)):
        if i > 0 and nums[i - 1] == nums[i]:
            continue

        l, r = i + 1, len(nums) - 1
        while l < r:
            total = nums[i] + nums[l] + nums[r]
            if total == 0:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l + 1]:
                    l += 1
                while l < r and nums[r] == nums[r - 1]:
                    r -= 1

                l += 1
                r -= 1

            elif total > 0:
                r -= 1
            else:
                l += 1

    return res


def judgeAnagrams(self, s, t):
    if len(s) != len(t):
        return False

    cntArr = [0] * 26
    for i, ch in enumerate(s):
        cntArr[ord(ch) - ord('a')] += 1

    for i, ch in enumerate(t):
        cntArr[ord(ch) - ord('a')] -= 1

    for i in range(len(26)):
        if cntArr[i] != 0:
            return False

    return True


def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
    res = 0
    mp = {}
    for a in range(len(A)):
        for b in range(len(B)):
            total = A[a] + B[b]
            mp[total] = mp.get(total, 0)

    for c in range(len(C)):
        for d in range(len(D)):
            res += mp[-(C[c] + D[d])]

    return res


class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next


"""
思维卡壳点：创建好哈希表后，如何跳过连续和为0的节点
误区：不能直接从哈希表的val里面操作
从头遍历
"""


def delConsecutiveZeroSum(self, head):
    dummy = ListNode(0, head)
    mp = {0: dummy}  # prefixSum -> node
    cur = head
    prefixSum = 0
    while cur:
        prefixSum += cur.val
        mp[prefixSum] = cur
        cur = cur.next

    prefix = 0
    cur = dummy
    while cur:
        prefix += cur.val
        # two circumstances:
        # 1. 哈希表此key下的值不止一个节点，意味着找到需要删除的连续节点和为0
        # 2. 如果只有一个节点，即cur == mp[prefix]
        cur.next = mp[prefix].next
        cur = cur.next

    return dummy.next


class KVNode:
    def __init__(self, key, val):
        self.key, self.val = key, val


class ChainingHashMap:
    def __init__(self, capacity):
        self.table = [None] * capacity

    def hash(self, key):
        return key % len(self.table)

    def get(self, key):
        index = self.hash(key)

        if self.table[index] is None:
            return -1

        list = self.table[index]
        for node in list:
            if node.key == key:
                return node.val

    def put(self, key, val):
        index = self.hash(key)

        if self.table[index] is None:
            self.table[index] = []
            self.table[index].append(KVNode(key, val))
            return

        list_ = self.table[index]
        for node in list_:
            if node.key == key:
                node.val = val
                return

        list_.append(KVNode(key, val))

    def remove(self, key):
        index = self.hash(key)

        list_ = self.table[index]
        if list_ is None:
            return

        list_[:] = [node for node in list_ if node.key != key]


class LinearProbingHashMap:
    def __init__(self):
        self.table = [None] * 10

    def hash(self, key):
        return key % len(self.table)

    def put(self, key, val):
        index = self.hash(key)
        node = self.table[index]
        if node is None:
            self.table[index] = KVNode(key, val)
        else:
            while index < len(self.table) and self.table[index] is not None and self.table[index].key != key:
                index += 1
            self.table[index] = KVNode(key, val)

    def get(self, key):
        index = self.hash(key)

        while index < len(self.table) and self.table[index] is not None and self.table[index].key != key:
            index += 1

        if self.table[index] is None:
            return -1

        return self.table[index].val

    def remove(self, key):
        index = self.hash(key)

        while index < len(self.table) and self.table[index] is not None and self.table[index].key != key:
            index += 1

        if self.table[index] is None:
            return

        self.table[index] = None
