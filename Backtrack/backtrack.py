from typing import List

"""
In a n*n board, put n queens in this board in that no queen will attack each other
"""


def solveNQueens(n: int) -> List[List[str]]:
    res = []
    col = set()
    posDiag = set()
    negDiag = set()
    board = [["."] * n for i in range(n)]

    def backtrack(r):
        if r == n:
            copy = ["".join(row) for row in board]
            res.append(copy)
            return

        for c in range(n):
            if c in col or (r + c) in posDiag or (r - c) in negDiag:
                continue

            col.add(c)
            posDiag.add(r + c)
            negDiag.add(r - c)
            board[r][c] = "Q"

            backtrack(r + 1)
            col.remove(c)
            posDiag.remove(r + c)
            negDiag.remove(r - c)
            board[r][c] = "."

    backtrack(0)
    return res


"""
电话号码组合问题，关键点：回溯函数的撤销步骤是否需要
"""


def letterCombinations(digits: str) -> List[str]:
    if not digits:
        return []
    mp = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz"
    }
    res = []

    def backtrack(i, curStr):
        if len(curStr) == len(digits):
            res.append(curStr)
            return

        for c in mp[digits[i]]:
            backtrack(i + 1, curStr + c)

    backtrack(0, "")
    return res


def letterCombinations2(digits: str) -> List[str]:
    if not digits:
        return []
    mp = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz"
    }
    combinations = []

    def backtrack(i, path):
        if len(path) == len(digits):
            combinations.append("".join(path))
            return

        for c in mp[digits[i]]:
            path.append(c)
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return combinations


"""
判断一个课程列表能否都修完
"""


def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
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

        ## 撤销
        visit.remove(crs)
        preMap[crs] = []
        return True

    for crs in range(numCourses):
        if not dfs(crs):
            return False

    return True


"""
输出完整的课程顺序
"""


def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    preMap = {c: [] for c in range(numCourses)}
    for crs, pre in prerequisites:
        preMap[crs].append(pre)

    visit, cycle = set(), set()
    res = []

    def dfs(crs):
        if crs in visit:
            return True
        if crs in cycle:
            return False
        cycle.add(crs)
        for pre in preMap[crs]:
            if not dfs(pre):
                return False

        cycle.remove(crs)
        visit.add(crs)
        res.append(crs)
        return True

    for crs in range(numCourses):
        if not dfs(crs):
            return []

    return res


"""
No duplicate in nums
"""


def subset(self, nums):
    self.res = []

    def backtrack(path, start):
        self.res.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(path, i + 1)
            path.pop()

    path = []
    backtrack(path, 0)
    return self.res


def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
    self.res = []
    candidates.sort()

    def backtrack(curSum, path, start):
        if curSum == target:
            self.res.append(path.copy())

        for i in range(start, len(candidates)):
            if curSum > target:
                break

            if i > start and candidates[i - 1] == candidates[i]:
                continue

            path.append(candidates[i])
            backtrack(curSum + candidates[i], path, i + 1)
            path.pop()

    path = []
    backtrack(0, path, 0)
    return self.res


"""
nums中元素不重复，全排列
"""


def permutation(self, nums):
    self.res = []

    def backtrack(path, visit):
        if len(path) == len(nums):
            self.res.append(path[:])
            return

        for i in range(len(nums)):
            if nums[i] in visit:
                continue

            path.append(nums[i])
            visit.add(nums[i])
            backtrack(path, visit)
            path.pop()
            visit.remove(nums[i])

    path = []
    visit = set()
    backtrack(path, visit)
    return self.res

"""
存在重复元素，全排列
"""
def permutation2(self, nums):
    self.res = []
    self.used = [False] * len(nums)
    nums.sort()

    def backtrack(path):
        if len(path) == len(nums):
            self.res.append(path)

        for i in range(len(nums)):
            if self.used[i]:
                continue

            if i > 0 and nums[i] == nums[i - 1] and not self.used[i - 1]:
                continue

            path.append(nums[i])
            self.used[i] = True
            backtrack(path)
            path.pop()
            self.used[i] = False

    path = []
    backtrack(path)
    return self.res

