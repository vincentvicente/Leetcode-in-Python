# 递归的本质：
### 函数调用挂起：当一个函数调用另一个函数时，当前函数的执行会暂停，等待被调用的函数执行完毕并返回结果。
### 控制权转移：被调用的函数在执行过程中拥有控制权，当前函数必须等待其执行完毕。
### 回溯继续执行：被调用函数返回后，当前函数恢复执行，并继续执行调用点之后的代码。
### 调用栈：管理函数调用的机制，用于保存函数的执行状态。
### 栈帧：每个函数调用都会生成一个栈帧，保存该函数的局部变量、参数和返回地址。
### 递归：递归调用使用调用栈来保存每次递归的执行状态，确保可以正确地回溯和返回结果。
```commandline
def functionA():
    print("function A starts")
    functionB()
    print("function A ends")
   
def functionB():
    print("function B starts")
    print("function B ends")
   
functionA (output: "function A starts"
                    "function B starts"
                    "function B ends"
                    "function A ends")
```

# 树的遍历：
## 1. DFS
    （1）前序
        i 迭代
        ii 递归
    （2）中序
    （3）后序
## 2. BFS(level order traversal)
    利用双端队列

## 3.二叉树的深度（depth）和高度（height）LC定义
### depth：从根节点到该节点的边数（从上往下）
        1        <- 深度 1
       / \
      2   3      <- 深度 2
     / \
    4   5        <- 深度 3

### height：从叶子节点到改节点的边数 （从下往上）
        1        <- 高度 3
       / \
      2   3      <- 高度 2
     / \
    4   5        <- 高度 1



