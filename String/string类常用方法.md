## 1. `join() `

join() 是 Python 中的一个`字符串方法`，它用于将可迭代对象（如列表、元组等）中的元素拼接成一个字符串，
并且在元素之间插入指定的分隔符。

### 语法：

`str.join(iterable)`:

* str：分隔符字符串，表示在拼接元素之间插入的内容。

* iterable：一个可迭代对象，如列表、元组等，里面的元素需要是字符串。

```commandline
words = ['Hello', 'World', 'Python']
result = ', '.join(words)
print(result)  # 输出: "Hello, World, Python"
```

## 2. `map()`

map() 是 Python 的`内置函数`。
它用于将一个函数应用到可迭代对象的每个元素上，并返回一个迭代器。

### 语法

`map(function, iterable)`

* function：要应用的函数，可以是内置函数，也可以是自定义的函数。
* iterable：要操作的可迭代对象（如列表、元组等）。

```commandline
words = ['hello', 'world', 'python']
upper_words = map(str.upper, words)
print(list(upper_words))  # 输出: ['HELLO', 'WORLD', 'PYTHON']

map(str, list) -> 将列表里的元素转化为字符串
function: str()
```

## 3. `lstrip()`

`lstrip()`是字符串的一个内置方法，专门用于移除字符串左侧（开头）的指定字符，
常用于去掉空白字符或特定字符。

### 语法
`str.lstrip([chars])`

* chars：可选参数，指定要移除的字符集合。如果不提供该参数，默认移除空白字符（空格、制表符、换行符等）。

```commandline
s = "00012345"
print(s.lstrip('0'))  # 输出: "12345"
```

