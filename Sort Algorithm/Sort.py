"""
bubble sort: 每轮将一个最大的元素放在数组末尾
总是从第一个元素开始比较
外层循环：控制轮数（n - 1），即排序好了几个元素
内层循环：负责交换元素，确保每轮将一个最大的元素放在数组末尾
类似于泡泡不断浮出水面的场景
"""


def bubbleSort(arr):
    if len(arr) <= 1:
        return arr
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        for j in range(n - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # 当某轮没有发生交换的时候，意味着已经排好序，提前终止循环
            break
    return arr


"""
insertion sort: 插入排序通过构建有序序列，
对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
"""


def insertionSort(arr):
    if len(arr) <= 1:
        return arr
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

    return arr


def selectionSort(arr):
    if len(arr) <= 1:
        return arr

    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        if min_index != i:
            arr[i], arr[min_index] = arr[min_index], arr[i]

    return arr


"""
LRU设计：缓存
"""
class Node:
    def __init__(self, key, val):
        self.key, self.val = key, val
        self.next = self.pre = None


class LRU:
    def __init__(self, capacity):
        self.cap, self.cache = capacity, {}
        self.left, self.right = Node(0, 0), Node(0, 0)
        self.left.next, self.right.pre = self.right, self.left

    def insert(self, node):
        pre, nxt = node, node.next
        pre.next, nxt.pre = nxt, pre

    def remove(self, node):
        pre, nxt = self.right.pre, self.right
        pre.next = nxt.pre = node
        node.next, node.pre = nxt, pre

    def get(self, key):
        if key in self.cache:
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val
        return -1

    def put(self, key, val):
        if key in self.cache:
            self.remove(self.cache[key])
        self.cache[key] = Node(key, val)
        self.insert(self.cache[key])

        if self.cap < len(self.cache):
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]


class ListNode:
    def __init__(self, val, next):
        self.val, self.next = val, None


def insertionSortLinkedList(head):
    if not head or not head.next:
        return head

    dummy = ListNode(0, head)
    pre, cur = head, head.next
    while cur:
        if cur.val >= pre.val:
            pre, cur = cur, cur.next
            continue

        tmp = dummy
        while tmp.next and tmp.val <= cur.val:
            tmp = tmp.next

        pre.next = cur.next
        cur.next = tmp.next
        tmp.next = cur
        cur = pre.next

    return dummy.next


def findLargest(nums):
    nums.sort()
    i = 0  # track the num in nums
    pre = -1
    total_nums_right = len(nums)

    while i < len(nums):
        if nums[i] == total_nums_right or pre < total_nums_right < nums[i]:
            return total_nums_right

        while i + 1 < len(nums) and nums[i] == nums[i + 1]:
            i += 1

        pre = nums[i]
        i += 1
        total_nums_right = len(nums) - i

    return -1
