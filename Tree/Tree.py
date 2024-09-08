import collections
from typing import Optional, List

# 递归的本质：
# 函数调用挂起：当一个函数调用另一个函数时，当前函数的执行会暂停，等待被调用的函数执行完毕并返回结果。
# 控制权转移：被调用的函数在执行过程中拥有控制权，当前函数必须等待其执行完毕。
# 回溯继续执行：被调用函数返回后，当前函数恢复执行，并继续执行调用点之后的代码。
# 调用栈：管理函数调用的机制，用于保存函数的执行状态。
# 栈帧：每个函数调用都会生成一个栈帧，保存该函数的局部变量、参数和返回地址。
# 递归：递归调用使用调用栈来保存每次递归的执行状态，确保可以正确地回溯和返回结果。

# 这里使用 TreeNode 类进行操作

"""
树的遍历：
1. DFS
    （1）前序
        i 迭代
        ii 递归
    （2）中序
    （3）后序
2. BFS(level order traversal)
    利用双端队列

3.二叉树的深度（depth）和高度（height）
depth：从根节点到该节点的边数
height：从叶子节点到改节点的边数
        1        <- 深度 0
       / \
      2   3      <- 深度 1
     / \
    4   5        <- 深度 2

        1        <- 高度 2
       / \
      2   3      <- 高度 1
     / \
    4   5        <- 高度 0

"""


class TreeNode:
    # 传左右指针的参数利于提高灵活性，指定节点的左右子节点
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.right = right
        self.left = left


class BinaryTree:
    def __init__(self):
        self.root = TreeNode()


def inOrderTraversal(root):
    res = []
    stack = []
    cur = root
    while cur or stack:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        res.append(cur)
        cur = cur.right
    return res

    # def inorderTraversal(self, root):
    # res = []
    #
    # def inorder(root):
    #     if not root:  # if root is None
    #         return
    #     inorder(root.left)
    #     res.append(root.val)
    #     inorder(root.right)
    #
    # inorder(root)
    # return res


def preorderTraversal(root):
    stack = []
    res = []
    cur = root
    while stack or cur:
        while cur:
            res.append(cur.val)
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        cur = cur.right
    return res


def postorderTraversal(root):
    stack = []
    res = []
    cur = root
    while stack or cur:
        while cur:
            stack.append(cur)
            res.insert(0, cur.val)  # 将节点的值插入到结果列表的开头
            cur = cur.right  # 先遍历右子树
        cur = stack.pop()
        cur = cur.left  # 再遍历左子树
    return res


def levelOrder(self, root):  # BFS
    res = []
    q = collections.deque()
    q.append(root)
    while q:
        qLen = len(q)
        level = []
        for i in range(qLen):
            node = q.popleft()
            if node:  # 判断节点是否为null
                level.append(node.val)
                q.append(node.left)
                q.append(node.right)
        if level:
            res.append(level)
    return res


def isSubtree(self, s: Optional[TreeNode], t: Optional[TreeNode]) -> bool:  # s: root, t: subtree
    if not t: return True
    if not s: return False
    if SameTree(s, t):
        return True
    return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)


def SameTree(self, s, t):
    if not s and not t:
        return True
    if s and t and s.val == t.val:
        return self.SameTree(s.left, t.left) and self.SameTree(s.right, t.right)

    return False


def isValidBST(root):
    def Valid(node, left, right):
        if not node:
            return True
        if not (node.val < left and node.right > right):
            return False
        return Valid(node.left, node.val, left) and Valid(node.right, node.val, right)

    Valid(root, float('-inf'), float('inf'))


def sumNumbers(root):
    def DFS(node, num):
        if not node:
            return 0

        if not node.left and not node.right:
            return num

        num = num * 10 + node.val
        return DFS(node.left, num) + DFS(node.right, num)

    return DFS(root, 0)


def validateBinaryTreeNodes(self, n: int, leftChild: List[int], rightChild: List[int]) -> bool:
    hasParent = set(leftChild + rightChild)
    hasParent.discard(-1)
    if len(hasParent) == n:
        return False

    root = -1
    for i in range(n):
        if i not in hasParent:
            root = i
            break

    visit = set()

    def dfs(i):  # connected and no cycle
        if i == -1:
            return True
        if i in visit:
            return False
        visit.add(i)
        return dfs(leftChild[i]) and dfs(rightChild[i])

    return dfs(root) and len(visit) == n


def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    columnTable = collections.defaultdict(list)
    q = collections.deque([(root, 0)])
    while q:
        node, column = q.popleft()
        if node:
            columnTable[column].append(node.val)
            q.append((node.left, column - 1))
            q.append((node.right, column + 1))

    return [columnTable[x] for x in sorted(columnTable.keys())]  # 以list的形式返回


## index is column and value is the node

## 判断完整二叉树：除了最后一行最后一个儿子可为空，其余都必须有左右儿子
## BFS找到第一个NULL，继续出队列，判断是否为空
def isCompleteTree(root: Optional[TreeNode]) -> bool:
    q = collections.deque([root])
    while q:
        node = q.popleft()
        if node:
            q.append(node.left)
            q.append(node.right)
        else:
            while q:
                if q.popleft():
                    return False

    return True


## 寻找二叉树每行最大值 - intuitive BFS
def largestValues(root: Optional[TreeNode]) -> List[int]:
    q = collections.deque([root])
    res = []
    while q:
        q_len = len(q)
        row_max = q[0].val
        for _ in range(q_len):
            node = q.popleft()
            row_max = max(row_max, node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(row_max)

    return res


## z字型遍历二叉树 - intuitive BFS，key：判断哪行需要reverse排序
def zigzagLevelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    res = []
    q = collections.deque([root] if root else [])

    while q:
        level = []
        for i in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        level = reversed(level) if len(res) % 2 else level
        res.append(level)
    return res


'''
LCA(Lowest Common Ancestor) -> 
'''


def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if not root:
        return None

    def dfs(node, target):
        if not node:
            return False

        if node == target:
            return True

        return dfs(node.left, target) or dfs(node.right, target)

    def LCA(node, p, q):
        if not node or node == p or node == q:
            return node
        l, r = LCA(node.left, p, q), LCA(node.right, p, q)
        if l and r:
            return node
        elif l:
            return l
        else:
            return r

    res = LCA(root, p, q)
    if res == p:
        return p if dfs(p, q) else None
    elif res == q:
        return q if dfs(q, p) else None
    else:
        return res


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        res = []

        def dfs(node):
            if not node:
                res.append("N")
                return
            res.append(str(node.val))
            node.left = dfs(node.left)
            node.right = dfs(node.right)

        dfs(root)
        return ", ".join(res)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        vals = data.split(", ")
        self.i = 0

        def dfs():
            if vals[self.i] == "N":
                self.i += 1
                return
            node = TreeNode(int(vals[self.i]))
            self.i += 1
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()


# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))

def levelOrderTraversal(root):
    if not root:
        return []

    res, q = [], collections.deque([root])
    while q:
        n = len(q)
        level = []
        for _ in range(n):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            level.append(node.val)
        res.append(level)
    return res


"""
The diameter of a binary tree is the longest path between two nodes in the tree, it could pass through
the root node or not path through it
"""


def findDiameter(root):
    res = [0]

    def dfs(node):
        if not node:
            return -1
        left = dfs(node.left)
        right = dfs(node.right)
        res[0] = max(left + right + 2, res[0])
        return max(right, left) + 1

    dfs(root)
    return res[0]


class Solution:
    def distributeCoins(self, root):
        self.result = 0

        def dfs(cur):
            if not cur:
                return [0, 0]

            l_size, l_coins = dfs(cur.left)
            r_size, r_coins = dfs(cur.right)

            size = l_size + r_size + 1
            coins = l_coins + r_coins + cur.val

            self.result += abs(size - coins)
            return [size, coins]

        dfs(root)
        return self.result


class SolutionOfLeetCode:
    """
        Delete all the node in a complete binary tree that has the same value as the target
        Only delete leaf nodes
        With postorder DFS solution
    """

    def delNode(self, root, target):
        if not root:
            return None

        root.left = self.delNode(root.left, target)
        root.right = self.delNode(root.right, target)

        if not root.left and not root.right and root.val == target:
            return None

        return root

    """
        In a full binary tree(with two children for every parent node):
        Every leaf node, with value 0(False) or 1(True)
        Every parent node, with value 2(OR) or 3(AND)
        With postorder DFS solution
    """

    def evaluateTree(self, root: Optional[TreeNode]) -> bool:
        if not root.left:
            return root.val == 1

        if root.val == 2:
            return self.evaluateTree(root.left) or self.evaluateTree(root.right)

        if root.val == 3:
            return self.evaluateTree(root.left) and self.evaluateTree(root.right)


def preOrderTraversal(root):
    stack = [root]
    res = []
    while stack:
        node = stack.pop()
        if node:
            res.append(node.val)
        else:
            continue

        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return res


def postOrderTraversal(root):
    stack = [root]
    res = []
    while stack:
        node = stack.pop()
        if node:
            res.append(node.val)
        else:
            continue
        if node.left:
            stack.append(node.left)

        if node.right:
            stack.append(node.right)

    return res[::-1]


def invertTree(root):
    # preorder
    # if not root:
    #     return None
    #
    # root.left, root.right = root.right, root.left
    # invertTree(root.left)
    # invertTree(root.right)
    #
    # return root

    # bfs
    # if not root:
    #     return None
    # q = collections.deque([root])
    # while q:
    #     node = q.popleft()
    #     node.left, node.right = node.right, node.left
    #     if node.left:
    #         q.append(node.left)
    #     if node.right:
    #         q.append(node.right)
    #
    # return root

    # dfs iterative
    if not root:
        return None

    stack = [root]
    while stack:
        node = stack[-1]
        stack.pop()
        node.left, node.right = node.right, node.left
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return root


def isSymmetric(self, root):
    stack = [root.right, root.left]
    while stack:
        leftNode = stack.pop()
        rightNode = stack.pop()
        if not leftNode and not rightNode:
            continue

        if not leftNode or not rightNode or leftNode.val != rightNode.val:
            return False
        stack.append(rightNode.right)
        stack.append(leftNode.left)
        stack.append(rightNode.left)
        stack.append(leftNode.right)

    return True


"""
LC 定义：根节点深度从1开始
前序
"""


def maxDepthOfBinaryTree(root):
    if not root:
        return 0

    leftDepth = maxDepthOfBinaryTree(root.left)
    rightDepth = maxDepthOfBinaryTree(root.right)
    res = max(leftDepth, rightDepth) + 1
    return res


def findDiameterTree(self, root):
    self.res = 0  # 初始化最大直径为 0

    # 递归遍历树节点，计算每个节点的直径，并更新全局最大直径
    def traverse(node):
        if not node:
            return 0

        # 递归计算左右子树的最大深度
        leftMax = maxDepth(node.left)
        rightMax = maxDepth(node.right)

        # 计算当前节点的直径（左右子树深度之和）
        myDiameter = leftMax + rightMax

        # 更新全局最大直径
        self.res = max(self.res, myDiameter)

        # 继续遍历左右子树
        traverse(node.left)
        traverse(node.right)

    # 计算树的深度
    def maxDepth(root):
        if not root:
            return 0

        # 递归计算左子树和右子树的深度
        leftMax = maxDepth(root.left)
        rightMax = maxDepth(root.right)

        # 返回当前节点的深度
        return 1 + max(leftMax, rightMax)

    # 从根节点开始遍历
    traverse(root)

    # 返回全局最大直径
    return self.res


def inOrder(root):
    # 记录遍历过的元素
    stack = []
    # 指针
    cur = root
    res = []
    while stack or cur:
        while cur:
            stack.append(cur)
            cur = cur.left
        # left leaf node
        cur = stack.pop()
        res.append(cur.val)
        cur = cur.right

    return res


def preOrder(root):
    stack, res = [root], []
    while stack:
        node = stack.pop()
        res.append(node.val)
        if node.right:
            stack.append(node.right)

        if node.left:
            stack.append(node.left)

    return res


def postOrder(root):
    stack, res = [root], []
    while stack:
        node = stack.pop()
        res.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return res[::-1]


def printOutPaths(self, root):
    def traverse(node, path, res):
        if not node:
            return

        path += str(node.val)

        if not node.left and not node.right:
            res.append(path)
            return

        path += "->"
        traverse(node.left, path, res)
        traverse(node.right, path, res)

    res = []
    traverse(root, "", res)
    return res


def findModes(root):
    if not root:
        return []

    traversal = []

    def traverse(node):
        nonlocal traversal
        if not node:
            return
        traversal.append(node.val)
        traverse(node.left)
        traverse(node.right)

    traverse(root)
    mp = collections.Counter(traversal)
    arr = list(mp.items())
    arr = sorted(arr, key=lambda x: x[1], reverse=True)
    res = []
    max_cnt = arr[0][1]
    i = 0
    while i < len(arr):
        if arr[i][1] == max_cnt:
            res.append(arr[i][0])
            i += 1
        else:
            break

    return res


def lowestCommonAncestorBinary(self, root, p, q):
    # 叶子节点
    if not root:
        return None

    if root == p or root == q:
        return root

    left = self.lowestCommonAncestorBinary(root.left, p, q)
    right = self.lowestCommonAncestorBinary(root.right, p, q)

    if left and right:
        return root

    if left and not right:
        return left

    if not left and right:
        return right

    return None

