# 哈希表

### 本质：加强版数组 —> `index = self.hash(key)`

```commandline
哈希表伪码逻辑
class MyHashMap:
    def __init__(self):
        self.table = [None] * 1000

    # 增/改，复杂度 O(1)
    def put(self, key, value):
        index = self.hash(key)
        self.table[index] = value

    # 查，复杂度 O(1)
    def get(self, key):
        index = self.hash(key)
        return self.table[index]

    # 删，复杂度 O(1)
    def remove(self, key):
        index = self.hash(key)
        self.table[index] = None

    # 哈希函数，把 key 转化成 table 中的合法索引
    # 时间复杂度必须是 O(1)，才能保证上述方法的复杂度都是 O(1)
    def hash(self, key):
        # ...
        return hash(key) % len(self.table)
```
### 哈希函数
`key` -> `int`:
Java: int hashCode()
* 哈希冲突：不同的key通过hashcode得到相同的index：
    1. 拉链法：self.table 存储链表（放键对值）
    2. 线性探查法：不断去寻找未被占据的位置

* 扩容和负载因子：\
  哈希冲突原因：
    1. 哈希函数设计不好
    2. 哈希表里key-val值过多，self.table.size不够放置
  负载因子：`size/table.length`: size: 键对值
    负载因子大了，自动扩容

### 拉链法
```commandline
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
```

