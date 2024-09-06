from typing import List


class Solution:
    """
    找到比当前元素大的第一个元素

    """

    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        # brute force
        # n = len(temperatures)
        # if n == 1:
        #     return [0]
        # res = [0] * n
        # for i in range(n - 1):
        #     j = i + 1
        #     while j <= n - 1:
        #         if temperatures[j] > temperatures[i]:
        #             res[i] = j - i
        #             break
        #         else:
        #             j += 1
        #
        # return res

        # Monotonic stack
        n = len(temperatures)
        if n == 1:
            return [0]

        res, stack = [0] * n, []
        # stack stores the index of the temperatures
        for i in range(n):
            while stack and temperatures[stack[-1]] < temperatures[i]:
                index = stack.pop()
                res[index] = i - index

            stack.append(i)

        return res


