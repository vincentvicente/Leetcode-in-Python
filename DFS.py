from typing import List


class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        preMap = {i: [] for i in range(numCourses)}
        for crs, pre in prerequisites:
            preMap[crs].append(pre)

        visit = set()

        def dfs(crs):
            if crs in visit:
                return False
            if not preMap[crs]:
                return True

            visit.add(crs)
            for pre in preMap[crs]:
                if not dfs(pre):
                    return False

            visit.remove(crs)
            preMap[crs] = []
            return True

        for crs in numCourses:
            if not dfs(crs):
                return False

        return True


class Solution(object):
    """
    Given a grid of size m * n, each cell with certain amount of gold. Find the largest amount of gold that one can get
    through it.
    Difficulty: has to brute force every cell, traverse with the path
    """


def getMaximumGold(self, grid):
    ROWS, COLS = len(grid), len(grid[0])

    def dfs(r, c, visit):
        if r < 0 or c < 0 or r == ROWS or c == COLS or (r, c) in visit or not grid[r][c]:
            return 0

        tmp = grid[r][c]  # start from (r, c)
        visit.add((r, c))
        max_gold = 0
        # for dx, dy in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
        #     max_gold = max(max_gold, dfs(r + dx, c + dy, visit))

        # max_gold = max(dfs(r+1, c, visit), dfs(r-1, c, visit), dfs(r, c+1, visit), dfs(r, c-1, visit))
        # visit.remove((r, c))
        # return tmp + max_gold

        tmp = grid[r][c]
        grid[r][c] = 0
        res = 0
        neighbors = [[r + 1, c], [r - 1, c], [r, c + 1], [r, c - 1]]
        for r2, c2 in neighbors:
            res = max(res, tmp + dfs(r2, c2, visit))

        grid[r][c] = tmp
        return res

    res = 0
    for r in range(ROWS):
        for c in range(COLS):
            res = max(res, dfs(r, c, set()))

    return res


class ListNode:
    def __init__(self, val):
        self.val = 0
        self.next = None


def insertionLinked(self, head):
    if not head or not head.next:
        return head

    dummy = ListNode(0, head)
    pre, cur = head, head.next
    while cur:
        if cur.val >= pre.val:
            pre, cur = cur, cur.next
            continue

        tmp = dummy
        while tmp.next and cur.val >= tmp.next.val:
            tmp = tmp.next

        pre.next = cur.next
        cur.next = tmp.next
        tmp.next = cur
        cur = pre.next

    return dummy.next
