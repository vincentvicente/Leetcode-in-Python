"""
python中的heapq模块方法：
heapq.heappush(heap, item)
功能: 将 item 插入到堆中，并保持堆的性质。
复杂度: 平均时间复杂度为 O(log n)，其中 n 是堆的大小。
/默认创建的最小堆
/排序原则：item的值，如果item为列表，元组类型，默认按照第一个值排序
heapq.heappop(heap)
功能: 弹出并返回堆中的最小元素，同时保持堆的性质。
复杂度: 平均时间复杂度为 O(log n)。

heapq.heapify(x)
功能: 将列表 x 原地转换为堆（最小堆）。
复杂度: 平均时间复杂度为 O(n)，其中 n 是列表的大小。
"""
import collections
import heapq


def kMostFrquentNums(self, nums, k):
    if len(nums) == 1:
        return nums

    num_cnt = collections.Counter(nums)
    min_heap = []

    for num, cnt in num_cnt.items():
        heapq.heappush(min_heap, (cnt, num))
        if len(min_heap) > k:
            heapq.heappop()

    return [num for cnt, num in min_heap]
