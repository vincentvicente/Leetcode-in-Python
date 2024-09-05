from typing import Optional, List

from DFS import ListNode
"""
重点：删除，插入节点
找需要操作的前一个节点以及后一个节点，必要时候需要tmp来指引
dummy：处理头节点必备
原理：删除或添加都是找当前需要处理节点（head）的前一个或后一个，而head本身没有前一个节点
@@ 双指针：pre, cur 和 fast, slow
"""

class ListNode:
    def __init__(self, val = 0, next = None):
        self.val, self.next = val, next

class MyLinkedList:
    def __init__(self):
        self.size, self.head = 0, ListNode()

    def get(self, index):
        if index < 0 or index >= self.size:
            return -1

        else:
            cur = self.head
            for i in range(index):
                cur = cur.next

        return cur.val

    def addHead(self, val):
        newHead = ListNode(val)
        newHead.next = self.head
        self.head = newHead
        self.size += 1

    def addTail(self, val):
        newTail = ListNode(val)
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = newTail
        self.size += 1

    def addAtIndex(self, index, val):
        InsertedNode = ListNode(val)
        if index < 0 or index > self.size:
            return

        dummy = ListNode(0, self.head)
        cur = dummy
        for _ in range(index):
            cur = cur.next

        tmp = cur.next
        cur.next = InsertedNode
        InsertedNode.next = tmp
        self.size += 1

    def delAtIndex(self, index):
        if index < 0 or index >= self.size:
            return

        dummy = ListNode(0, self.head)
        cur = dummy
        for _ in range(index):
            cur = cur.next

        cur.next = cur.next.next
        self.size -= 1

def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    ## recursive method
    if not head:  # 没有节点需要反转
        return None
    newHead = head
    if head.next:  # 有sub problem，即有节点需要反转
        newHead = self.reverseList(head.next)
        head.next.next = head
    head.next = None  # 原来头节点变为尾节点指针指向NULL
    return newHead

    ## iterative method
    # pre, cur = None, head
    # while cur:
    #     nxt = cur.next
    #     cur.next = pre
    #     pre = cur
    #     cur = nxt
    #
    # return pre


def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
    fast, slow = head, head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if slow == fast:
            break

    if not fast or not fast.next:
        return None

    fast = head
    while fast != slow:
        fast = fast.next
        slow = slow.next

    return slow




class ListNode:
    pass


def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()  # 不传参数就是默认 val = 0, next = None
    tail = dummy  # tail指针，指向新链表的尾部
    while l1 and l2:
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next

    if l1:
        tail.next = l1
    elif l2:
        tail.next = l2
    return dummy.next


def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    cur = dummy
    carry = 0
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        # New digit
        val = val1 + val2 + carry
        val = val % 10
        cur.next = ListNode(val)
        carry = val // 10
        # update pointers
        cur = cur.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None

    return dummy.next


def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    left = dummy
    right = head

    while n > 0 and right:
        right = right.next
        n -= 1

    while right:
        right = right.next
        left = left.next

    left.next = left.next.next
    return dummy.next


##Return the head of the linked list after swapping the values of the kth node from the beginning
##and the kth node from the end (the list is 1-indexed).
def swapNodes(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    left = dummy
    right = dummy

    while k > 0 and right:
        right = right.next
        k -= 1
    ##locate the kth node from the beginning, node stored at cur
    cur = right

    ##locate the kth node from the end, node stored at left
    while right:
        right = right.next
        left = left.next

    ##swap cur.val and left.val
    val = left.val
    left.val = cur.val
    cur.val = val

    return dummy.next


def swapPairs(head: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    pre, cur = dummy, head
    ## store pointers
    while cur and cur.next:
        nxtPair = cur.next.next
        second = cur.next

        ## reverse nodes
        second.next = cur
        cur.next = nxtPair
        dummy.next = second

        ## update pointers
        pre = cur
        cur = nxtPair

    return dummy.next


"""
Construct a deep copy of the list. The deep copy should consist of exactly n brand-new nodes, 
where each new node has its value set to the value of its corresponding original node. 
Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers 
in the original list and copied list represent the same list state. 
None of the pointers in the new list should point to nodes in the original list.
"""


def copyRandomList(head: 'Optional[Node]') -> 'Optional[Node]':
    # 原节点作为键，copy的节点作为值
    oldToCopy = {}

    # 只拷贝节点的值
    cur = head
    while cur:
        copy = Node(cur.val)
        oldToCopy[cur] = copy
        cur = cur.next

    # 拷贝节点的指针
    cur = head
    while cur:
        copy = oldToCopy[cur]
        copy.next = oldToCopy[cur.next]
        copy.random = oldToCopy[cur.random]
        cur = cur.next

    return oldToCopy[head]


def reorderList(head: Optional[ListNode]) -> None:
    ##  找到first和second部分
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    ##  second的头节点是slow.next，reverse后second.next指针指向NULL，即为合并后链表的tail pointer
    second = slow.next
    pre = second.next = None
    ##  second部分的节点已经reverse好了
    while second:
        tmp = second.next
        second.next = pre
        pre = second
        second = tmp

    ##  合并first, second
    first, second = head, pre  # 合并后头尾节点
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first = tmp1
        second = tmp2


"""
第一步，创建dummy节点规避头节点问题
第二步，通过for循环找到left位置
第三步，通过for循环去reverse从left到right部分的链表
第四步，修改指针，使得reverse后的链表和其余部分相连接
"""


def reverseBetween(head: Optional[ListNode], l: int, r: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    pre, cur = dummy, head

    for _ in range(l - 1):
        pre, cur = pre.next, cur.next
    lPre = pre

    pre = None
    for _ in range(r - l + 1):
        tmp = cur.next
        cur.next = pre
        pre = cur
        cur = tmp

    lPre.next.next = cur
    lPre.next = pre

    return dummy.next


"""
第一步，判断链表是否为空或者一个节点
第二步，利用mergesort，将链表分成两部分
第三步，利用递归，分别将两部分进行排序
第四步，将排序好的两部分合并
"""


def sortList(self, head):
    if not head or not head.next:
        return head

    left = head
    right = self.getMid(head)
    tmp = right.next
    right.next = None
    right = tmp

    left = self.sortList(left)
    right = self.sortList(right)
    return mergeList(left, right)


def getMid(self, head):
    slow, fast = head, head.next  # 使得无论长度为奇数还是偶数
    while fast and fast.next:  # 确保快指针后还有一个安全的节点，即fast.next不会出现NULL从而导致fast.next.next报错
        slow, fast = slow.next, fast.next.next
    return slow


def mergeList(l1, l2):
    tail = dummy = ListNode()
    while l1 and l2:
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    if l1:
        tail.next = l1
    elif l2:
        tail.next = l2
    return dummy.next


"""
Algorithm used to implement a data structure that follows the constraints of a Least Recently Used (LRU) cache.
"""


class Node:
    def __init__(self, key, val):
        self.key, self.val = key, val
        self.pre = self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = {}
        self.cap = capacity
        self.left, self.right = Node(0, 0), Node(0, 0)
        self.left.next, self.right.pre = self.right, self.left

    def remove(self, node):
        pre, nxt = node.pre, node.next
        pre.next, nxt.pre = nxt, pre

    def insert(self, node):
        pre, nxt = self.right.pre, self.right
        pre.next = nxt.pre = node
        node.pre, node.next = pre, nxt

    def get(self, key: int) -> int:
        if key in self.cache:
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val
        return -1

    def put(self, key: int, val: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])
        self.cache[key] = Node(key, val)
        self.insert(self.cache[key])

        if len(self.cache) > self.cap:
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]


"""
受merge two linked lists启发，拓展到一共要merge k个sorted linked lists。利用mergedLists方法不断去合并lists
"""


class Solution:

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        ## edge cases
        if not lists or len(lists) == 0:
            return None

        while len(lists) > 1:
            mergedList = []
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if (i + 1) < len(lists) else None
                mergedList.append(self.mergeLists(l1, l2))
            lists = mergedList
        return lists[0]

    def mergeLists(self, l1, l2):
        tail = dummy = ListNode()
        while l1 and l2:
            if l1.val < l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        if l1:
            tail.next = l1
        elif l2:
            tail.next = l2
        return dummy.next


"""
Floyd's algorithm: to find a cycle. Fast pointer and slow pointer, when two encounter for the first time,
move fast pointer to the start point, make them both the same speed, when they encounter for the second time,
their position is the start of the cycle
"""


def findDuplicate(self, nums: List[int]) -> int:
    f, s = 0, 0
    while True:
        f = nums[nums[f]]
        s = nums[s]
        if s == f:
            break

    f = 0
    while True:
        f = nums[f]
        s = nums[s]
        if s == f:
            return f


"""
Remove every node which has a node with a greater value anywhere to the right side of it.
Method 1: Monotonic stack
Method 2: Reverse LinkedList
"""


def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
    def reverse(head):
        pre, cur = None, head
        while cur:
            tmp = cur.next
            cur.next = pre
            cur, pre = tmp, cur

        return pre

    head = reverse(head)
    cur = head
    cur_max = cur.val
    while cur.next:
        if cur.next.val < cur_max:
            cur.next = cur.next.next

        else:
            cur_max = cur.next.val
            cur = cur.next

    return reverse(head)

    ## Method 2 Monotonic stack
    stack = []
    curr = head
    while curr:
        while stack and curr.val > stack[-1]:
            stack.pop()
        stack.append(curr.val)
        curr = curr.next

    dummy = ListNode()
    cur = dummy
    for n in stack:
        cur.next = ListNode(n)
        cur = cur.next

    return dummy.next


def insertionSort(arr):
    if len(arr) <= 1:
        return arr

    for i in range(1, len(arr)):
        tmp = arr[i]  # tmp指针存在的意义在于记录下第一次循环前即j = i - 1时候所在索引的数值
        j = i - 1
        # j指针在不断移动，在最后一次循环的时候跳出前更新指针位置，此时在j + 1的位置插入即可
        while j >= 0 and tmp < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = tmp
        return arr


class ListNode:
    def __init__(self, val, next):
        self.val, self.next = val, None


"""
通过插入排序的算法来排序单向链表
通过两个指针来遍历链表：pre, cur
pre以及之前都是已经排序过的链表，比较pre和cur存储的值的大小，来确定是否需要进行插入排序
接着寻找要插入的位置：从头开始找，因为是单向链表，找到后进行指针的切换
"""


def insertionSortList(self, head):
    dummy = ListNode(0, head)
    pre, cur = head, head.next

    while cur:
        if cur.val >= pre.val:
            pre, cur = cur, cur.next
            continue

        # 跳出循环意味着找到需要排序的节点了，得从头开始找位置去插入这个节点
        tmp = dummy
        while tmp.next.val < cur:
            tmp = tmp.next

        # 跳出循环意味着找到需要插入的位置了，即在tmp和tmp.next之间, 注意节点之间插入顺序，需要保证对于每个节点还有reference
        pre.next = cur.next
        cur.next = tmp.next
        tmp.next = cur
        cur = pre.next

    return dummy.next

def delElement(self, head, val):
    dummy = ListNode(head, 0)
    pre, cur = dummy, head
    while cur:
        tmp = cur.next
        if cur.val == val:
            pre.next = tmp
            pre, cur = pre.next, tmp

        else:
            pre, cur = cur, cur.next

    return dummy.next

def removeZeroSumSublists(self, head: Optional[ListNode]) -> Optional[ListNode]:
    # prefixSum
    dummy = ListNode(0)
    dummy.next = head
    start = dummy

    while start:
        end = start.next
        prefixSum = 0
        while end:
            prefixSum += end.val
            if prefixSum == 0:
                start.next = end.next

            end = end.next

        start = start.next

    return dummy.next

    # hashMap {prefixSum:Node}
    # update hashmap
    mp = {}
    dummy = ListNode(0, head)
    cur = dummy
    prefixSum = 0
    while cur:
        prefixSum += cur.val
        mp[prefixSum] = cur
        cur = cur.next

    # locate the zero sum consecutive
    prefix = 0
    cur = dummy
    while cur:
        prefix += cur.val
        cur.next = mp[prefix].next
        cur = cur.next

    return dummy.next


def detectCycle(self, head):
    slow, fast = head, head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow == fast:
            return True

    return False

