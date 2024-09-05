# Python报错分析
## 类型
### 1. SyntaxError
解释：语法错误。通常在 Python 解释器解析代码时发生，表示代码不符合 Python 的语法规则。
```commandline
if True
    print("Missing colon")
```

### 2. IndentationError
解释：缩进错误。Python 要求代码块必须正确缩进，不符合缩进规范的代码会引发此错误。
```commandline
def my_function():
print("This line is not indented")
```

### 3. NameError
解释：尝试访问一个未定义的变量或函数时引发。
```commandline
print(undeclared_variable)
```

### 4. TypeError
解释：操作或函数应用于错误类型的对象时引发。例如，试图将一个字符串与一个整数相加。
```commandline
result = "string" + 10
```
### 5. ValueError
解释：函数接收到正确类型但不正确值的参数时引发。例如，试图将一个无法转换为整数的字符串传递给 int() 函数。
```commandline
result = int("abc")
```

### 6. IndexError
解释：尝试访问列表，元组或字符串中不存在的索引引发的错误。
```commandline
list = [1,2,3]
print(list[3])
```

### 7. KeyError
解释：尝试访问字典中不存在的键时引发。
```commandline
my_dict = {"a": 1, "b": 2}
print(my_dict["c"])
```

### 8. AttributeError
解释：尝试访问不存在的对象属性时引发。
```commandline
my_list = [1, 2, 3]
my_list.append(4)
my_list.add(5)  # 列表没有 add 方法
```


