from typing import List


class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        l, r = 0, len(matrix) - 1
        while l < r:
            for i in range(r - l):
                t, b = l, r
                topLeft = matrix[t][l + i]
                ## rotate from bottom left to top left
                matrix[t][l + i] = matrix[t][b - i]
                ## rotate from bottom right to bottom left
                matrix[t][b - i] = matrix[r - i][b]
                ## rotate from top right to bottom right
                matrix[r - i][b] = matrix[t + i][r]
                ## rotate from top left to top right
                matrix[t + i][r] = topLeft

            l += 1
            r -= 1


def dailyTemp(t):
    res = [0] * len(t)
    stack = []  # (index, temperature)
    for i, t in enumerate(t):
        while stack and t > stack[-1][1]:
            index, temp = stack.pop()
            res[index] = i - index
        stack.append((i, t))

    return res


def spiralOrder(matrix: List[List[int]]) -> List[int]:
    res = []
    l, r = 0, len(matrix[0])
    t, b = 0, len(matrix)

    while l < r and t < b:
        # get top row
        for i in range(l, r):
            res.append(matrix[t][i])
        t += 1

        # get right column
        for i in range(t, b):
            res.append(matrix[i][r - 1])
        r -= 1

        if not (l < r and t < b):
            break

        # get bottom row
        for i in range(r - 1, l - 1, -1):
            res.append(matrix[b - 1][i])
        b -= 1

        # get left column
        for i in range(b - 1, t - 1, -1):
            res.append(matrix[i][l])
        l += 1

    return res


"""
Find the biggest value of a grid with binary
Method 1: most intuitive: modify the grid, flip the rows first, then the cols. Then count the res
"""


def matrixScore(grid):
    rows, cols = len(grid), len(grid[0])
    # flip the rows
    for r in range(rows):
        if not grid[r][0]:
            for c in range(cols):
                grid[r][c] = 1 if not grid[r][c] else 0

    # flip the cols
    for c in range(cols):
        count_one = 0
        for r in range(rows):
            if not grid[r][c]:
                count_one += 1

        if count_one < (rows - count_one):
            for r in range(rows):
                grid[r][c] = 1 if not grid[r][c] else 0

    # count the result
    res = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]:
                res += grid[r][c] << (cols - 1 - c)

    return res

