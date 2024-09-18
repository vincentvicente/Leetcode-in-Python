"""
设计trie: 类似于树的结构
put(word) ->
search(word) -> 是否存在
prefix -> 是否存在
"""

## 难点在于：如何把单词放入树里，何时知道这个词开始结束

class TrieNode:
    def __init__(self):
        self.children, self.endOfWord = {}, False

class Trie:
    def __init__(self):
        self.root = TrieNode()
        
