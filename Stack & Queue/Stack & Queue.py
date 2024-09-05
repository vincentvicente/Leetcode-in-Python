import collections
from typing import List

"""
FILO structure
"""


class stack:
    def __init__(self):
        self.items = []

    def push(self, val):
        self.items.append(val)

    def pop(self):
        self.items.pop()

    def peek(self):
        return self.items[-1]


"""
Python 的 collections.deque 由一系列固定大小的内存块（blocks）组成。这些块被链接在一起形成一个双向链表。这种结构使得 deque 能够在两端高效地进行插入和删除操作。
在实现上，deque 通过一个指针数组来管理这些块，并且可以通过计算索引快速访问元素。这使得 deque 中的索引访问接近 O(1) 的复杂度。
想象一列火车，每节车厢（内存块）上都有座位（元素）。如果你想找到某个乘客（元素），你可能需要从火车头开始一节一节地走过车厢，直到找到那个乘客。
这种情况下，找到特定元素的时间会随火车的长度增加而增加。
默认格式：
d = deque()
d.append(1)
d.append(2)
print(d) 输出：deque([1, 2])

初始化必须传入可迭代对象
d = deque(1) 错误写法
d = deque([1]) or d = deque((1, ))
"""

def wallsAndGates(rooms: List[List[int]]) -> None:
    rows, cols = len(rooms), len(rooms[0])  # design the coordinates
    visit = set()  # traverse every room without repetition
    q = collections.deque()  # BFS
    for r in range(rows):
        for c in range(cols):
            if rooms[r][c] == 0:  # locate the gates
                q.append((r, c))
                visit.add((r, c))

    def addRooms(r, c):
        if (r < 0 or r == rows or c < 0 or c == cols or rooms[r][c] == -1 or
                (r, c) in visit):  # judge if the room is out of boundary or visited before or if it is an obstacle
            return
        visit.add((r, c))
        q.append((r, c))

    dist = 0
    while q:
        for i in range(len(q)):  # BFS
            r, c = q.popleft()
            rooms[r][c] = dist
            # expand in four directions
            addRooms(r + 1, c)
            addRooms(r - 1, c)
            addRooms(r, c + 1)
            addRooms(r, c - 1)
        dist += 1  # increment 1 every time when finished one BFS


def numIslands(grid: List[List[str]]) -> int:
    ## method1: BFS
    rows, cols = len(grid), len(grid[0])  # 建立坐标
    visit = set()  # 记录已经访问过的位置
    islands = 0  # 记录岛屿数量

    def bfs(r, c):
        q = collections.deque()
        q.append((r, c))
        visit.add((r, c))
        while q:
            row, col = q.popleft()
            directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            for x, y in directions:
                r, c = row + x, col + y
                if (r in range(rows) and c in range(cols) and (r, c) not in visit and
                        grid[r][c] == "1"):  # 范围内，未访问，值为1
                    q.append((r, c))
                    visit.add((r, c))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in visit:
                bfs(r, c)
                visit.add((r, c))
                islands += 1

    return islands


def numIslands2(grid: List[List[str]]) -> int:
    ## method 2: DFS
    rows, cols = len(grid), len(grid[0])
    visit = set()
    direct = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    def isInvalid(r, c):
        return r < 0 or r == rows or c < 0 or c == cols

    def dfs(r, c):
        if isInvalid(r, c) or (r, c) in visit or grid[r][c] == "0":
            return
        visit.add((r, c))
        for dr, dc in direct:
            dfs(r + dr, c + dc)

    islands = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and grid[r][c] not in visit:
                islands += 1
                dfs(r, c)
    return islands


# 1. (DFS) Capture unsurrounded regions (0 -› T)
# 2. Capture surrounded regions (0 -> X)
# 3. Uncapture unsurrounded regions (T -> 0)
def solve(self, board: List[List[str]]) -> None:
    rows, cols = len(board), len(board[0])

    def capture(r, c):
        if (r < 0 or r == rows or c < 0 or c == cols or
                board[r][c] != "O"):
            return
        board[r][c] = "T"
        capture(r + 1, c)
        capture(r - 1, c)
        capture(r, c + 1)
        capture(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == "O" and (r in [0, rows - 1] or c in [0, cols - 1]):
                capture(r, c)

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == "O":
                board[r][c] = "X"

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == "T":
                board[r][c] = "O"


def orangesRotting(grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    direct = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    q = collections.deque()
    fresh, time = 0, 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                fresh += 1
            if grid[r][c] == 2:
                q.append((r, c))

    while q and fresh > 0:
        for i in range(len(q)):
            row, col = q.popleft()
            for x, y in direct:
                r, c = row + x, col + y
                if r < 0 or r == rows or c < 0 or c == cols or grid[r][c] != 1:
                    continue
                grid[r][c] = 2
                fresh -= 1
        time += 1

    return time if fresh == 0 else -1


def isValid(s: str) -> bool:
    stack = []
    closeToOpen = {")": "(", "]": "[", "}": "{"}
    for c in s:
        if c in closeToOpen:
            if stack and stack[-1] == closeToOpen[c]:
                stack.pop()
            else:
                return False
        else:
            stack.append(c)

    return True if not stack else False


def minRemoveToMakeValid(self, s: str) -> str:
    s, stack = list(s), []
    for i, c in enumerate(s):
        if c == "(":
            stack.append(i)

        elif c == ")":
            if stack and s[stack[-1]] == "(":
                stack.pop()
            else:
                stack.append(i)
        else:
            continue

    for i in stack:
        s[i] = ""

    return "".join(s)


def dailyTemperatures(temperatures: List[int]) -> List[int]:  # monotonic stack
    res = [0] * len(temperatures)
    stack = []  # pair(temperature, index)
    for t, i in enumerate(temperatures):
        while stack and t > stack[-1][0]:
            stackT, stackInd = stack.pop()
            res[stackInd] = (i - stackInd)
        stack.append([t, i])

    return res


def twoSum(list, target):
    numMap = {}
    for num, i in enumerate(list):
        complement = target - num
        if complement in numMap:
            return [numMap[complement], i]
        numMap[num] = i

    return -1


def decodeString(s: str) -> str:
    stack = []
    for i, n in enumerate(s):
        if n != "]":
            stack.append(n)
        substr = ""
        while stack[-1] != "[":
            substr = stack.pop() + substr
        stack.pop()

        k = ""
        while stack and stack[-1].isdigit():
            k = stack.pop() + k

        stack.append(int(k) * substr)

    return "".join(stack)


def floodFill(image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
    rows, cols = len(image), len(image[0])
    if image[sr][sc] == color:
        return image
    q = collections.deque([(sr, sc)])

    while q:
        row, col = q.popleft()
        image[row][col] = color
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        for x, y in directions:
            r, c = row + x, col + y
            if (r < 0 or r == rows or c < 0 or
                    r == rows or image[r][c] != image[sr][sc]):
                continue
            q.append((r, c))

    return image


def islandPerimeter(grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    visit = set()

    def dfs(r, c):
        if (r < 0 or r == rows or c < 0 or c == cols or
                grid[r][c] == 0):
            return 1

        if (r, c) in visit:
            return 0

        visit.add((r, c))
        perimeter = dfs(r, c)
        perimeter += dfs(r + 1, c)
        perimeter += dfs(r - 1, c)
        perimeter += dfs(r, c + 1)
        perimeter += dfs(r, c - 1)
        return perimeter

    for r in range(rows):
        for c in range(cols):
            if grid[r][c]:
                return dfs(r, c)


def updateMatrix(mat: List[List[int]]) -> List[List[int]]:
    rows, cols = len(mat), len(mat[0])
    visit = set()
    q = collections.deque()

    for r in range(rows):
        for c in range(cols):
            if mat[r][c] == 0:  # locate all those 0s
                visit.add((r, c))
                q.append((r, c))

    dist = 1
    while q:
        for i in range(len(q)):
            row, col = q.popleft()
            directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            for x, y in directions:
                r, c = row + x, col + y
                if (r in range(rows) and c in range(cols) and
                        (r, c) not in visit):
                    q.append((r, c))
                    visit.add((r, c))
                    mat[r][c] = dist
        dist += 1

    return mat


def sumOfSquares(n):
    sum = 0
    while n:
        n, digit = divmod(n, 10)
        sum += digit ** 2

    return sum


def shortestBridge(grid: List[List[int]]) -> int:
    N = len(grid)
    direct = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    visit = set()

    def isinvalid(r, c):
        return r < 0 or c < 0 or r == N or c == N

    ## locate one island
    def dfs(r, c):
        if isinvalid(r, c) or (r, c) in visit or not grid[r][c]:
            return

        visit.add((r, c))
        for dr, dc in direct:
            dfs(r + dr, c + dc)

    ## calculate the distance
    def bfs(r, c):
        res, q = 0, collections.deque(visit)

        while q:
            for i in range(len(q)):
                r, c = q.popleft()
                for dr, dc in direct:
                    curR, curC = r + dr, r + dc
                    if isinvalid(curR, curC) or (curR, curC) in visit:
                        continue
                    if grid[curR][curC]:
                        return res
                    q.append((curR, curC))
                    visit.add((curR, curC))
            res += 1

    for r in range(N):
        for c in range(N):
            if grid[r][c]:
                dfs(r, c)
                return bfs(r, c)


def inOrderTraversal(root):
    if not root:
        return None

    res = []

    def inorder(root):
        inorder(root.left)
        res.append(root.val)
        inorder(root.right)

    inorder(root)
    return res


def exist(board: List[List[str]], word: str) -> bool:
    rows, cols = len(board), len(board[0])
    path = set()

    def dfs(r, c, i):
        if i == len(word):
            return True
        if r < 0 or r == rows or c < 0 or c == cols or board[r][c] != word[i] or (r, c) in path:
            return False

        path.add((r, c))
        res = dfs(r + 1, c, i + 1) or dfs(r - 1, c, i + 1) or dfs(r, c + 1, i + 1) or dfs(r, c - 1, i + 1)
        path.remove((r, c))
        return res

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    return False


"""
去掉最少数量的括号，使得字符串符合要求，即左括号必须匹配右括号，数量上也得一致
"""


def minRemoveToMakeValid(s: str) -> str:
    removal = set()
    stack = []
    for i, c in enumerate(s):
        if c not in "()":
            continue
        if c == "(":
            stack.append(i)
        elif not stack:
            removal.add(i)
        else:
            stack.pop()

    removal = removal.union(set(stack))
    res = []
    for i, c in enumerate(s):
        if i not in removal:
            res.append(c)

    return "".join(res)


def validParenthesis(s):
    mp = {")": "(", "]": "[", "}": "{"}
    stack = []
    for c in s:
        if c in mp:
            if stack and stack[-1] == mp[c]:
                stack.pop()
            else:
                return False
        else:
            stack.append(c)

    return True if not stack else False


def generateParenthesis(n: int) -> List[str]:
    res = []
    stack = []

    def backtrack(openN, closedN):
        if openN == closedN == n:
            res.append("".join(stack))
            return
        if openN < n:
            stack.append("(")
            backtrack(openN + 1, closedN)
            stack.pop()
        if closedN < openN:
            stack.append(")")
            backtrack(openN, closedN + 1)
            stack.pop()

    backtrack(0, 0)
    return res


def reverseParentheses(self, s: str) -> str:
    def reverseSingleWord(s):
        n = len(s)
        l, r = 0, n - 1
        while l < r:
            s[l], s[r] = s[r], s[l]
            l += 1
            r -= 1

    stack = []
    new_l = []
    s = list(s)
    for ch in s:
        if ch == ")":
            while stack[-1] != "(":
                cha = stack.pop()
                new_l.append(cha)
            stack.pop()
        stack.append(ch)

    return "".join(new_l)


"""
O(n)解决滑动窗口中最大值的问题
Monotonic non-increasing queue
"""


def maxSlidingWindow(nums, k):
    l, r = 0, 0
    res, q = [], collections.deque()

    while r < len(nums):
        while q and q[-1] < nums[r]:
            q.pop()

        q.append(nums[r])

        if r - l + 1 == k:
            res.append(q[0])
            # 思维卡壳点：意味着q里存放了连续的k个元素
            if nums[l] == q[0]:
                q.popleft()
            l += 1

        r += 1

    return res

def asteroidsCollision(asteroids):
    stack = []
    for a in asteroids:
        destroyed = False
        while stack and stack[-1] > 0 > a:
            # destroyed
            if stack[-1] > abs(a):
                destroyed = True
                break
            # destroyed
            elif stack[-1] == abs(a):
                destroyed = True
                stack.pop()
                break
            else:
                stack.pop()

        if not destroyed:
            stack.append(a)

    return stack


