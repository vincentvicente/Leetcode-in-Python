## collections.heapq (heapq默认最小堆)

### 1.`heapify`:

`heapq.heapify(x)`

#### 将列表 x 转换为堆，原地重新排列元素。

```commandline
nums = [20, 10, 30]
heapq.heapify(nums)
print(nums)  # 输出: [10, 20, 30]
```

### 2.`heappush`:

`heapq.heappush(heap, item)`

#### 作用：将元素 item 压入堆中。

```commandline
heap = []
heapq.heappush(heap, 10)
heapq.heappush(heap, 5)
heapq.heappush(heap, 30)
print(heap)  # 输出: [5, 10, 30]
```

注意：在 Python 的 heapq 模块中，堆是通过列表来表示的，heapq 默认是一个最小堆（即最小值在堆顶）。
当你将元素添加到堆中时，heapq 会根据元素的值来自动维护堆的性质。
对于元组，heapq 会首先比较元组的第一个元素，
如果第一个元素相等，则比较第二个元素，依次类推。

### 3.`heappop`:

`heapq.heappop(heap)`

#### 从堆中弹出并返回最小的元素（最小堆）。

```commandline
heapq.heappop(heap)  # 输出: 5
print(heap)  # 输出: [10, 30]
```

