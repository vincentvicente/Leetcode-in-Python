# # # # # # ## 求和100以内的偶数/奇数
# # # # # # sum = 0
# # # # # # for i in range(1,100):
# # # # # #     if i % 2 != 0:
# # # # # #         sum += i
# # # # # # print(sum)
# # # # #
# # # # # ## 阶乘
# # # # # def get_factorial(n):
# # # # #     result = 1
# # # # #     while n > 0:
# # # # #         result *= n
# # # # #         n -= 1
# # # # #     return result
# # # # #
# # # # #
# # # # # # print(get_factorial(5))
# # # # #
# # # # # ## 斐波那契数列
# # # # # def Fibonacci_sequence(n):
# # # # #     if n == 1 or n == 2:
# # # # #         result = 1
# # # # #     else:
# # # # #         result = Fibonacci_sequence(n - 2) + Fibonacci_sequence(n - 1)
# # # # #     return result
# # # # #
# # # # #
# # # # # fib = [1, 1]
# # # # # n = 10
# # # # # for i in range(2, n + 1):
# # # # #     fib.append(fib[i - 2] + fib[i - 1])
# # # # # print(fib)
# # # # #
# # # # # #
# # # # # # print(Fibonacci_sequence(6))
# # # # # #
# # # # # # ## 求圆的周长（输入半径）
# # # # # # from math import pi
# # # # # #
# # # # # # r = float(input('Enter the radius: '))
# # # # # # c = 2 * pi * r
# # # # # # s = pi * (r ** 2)
# # # # # # print(f'The circumference of the circle is {c}')
# # # # # # print(f'The area of the circle is {s}')
# # # # #
# # # # # ## 编写程序: 输入三个数，从小到大输出三个数
# # # a = int(input('Enter the first numebr: '))
# # # b = int(input('Enter the second numebr: '))
# # # c = int(input('Enter the third numebr: '))
# # # list = [a, b, c]
# # # new_list = sorted(list)
# # # print(f'The ordered list is: {new_list[0]}, {new_list[1]}, {new_list[2]}')
# # #
# # # # 暂停后输出
# # # import time
# # #
# # # time.sleep(2)
# # # print("Hello World!")
# # #
# # # # # ## lambda：lambda 表达式是 Python 中用于创建匿名函数的工具。它允许你在需要函数对象的任何地方使用简短的函数定义。
# # # # # ## （lambda arguments: expression）
# # # a = lambda x, y: x * y
# # # print(a(3, 4))
# # #
# # #
# # # ## linear search 和 binary search
# # # def linear_search(x, a):
# # #     for k in range(len(a)):
# # #         if a[k] == x:
# # #             return k
# # #     return -1
# # #
# # #
# # # # # ## while 循环写
# # # def linear_search(x, a):
# # #     k = 0
# # #     while k < len(a) and a[k] != x:
# # #         k += 1
# # #     if k == len(a):
# # #         return -1
# # #     else:
# # #         return k
# # #
# # #
# # # # # ##str1 和 str2的公共最长子串，lcs(longest common subsequence)
# # # def lcs(str1, str2):
# # #     res = ''
# # #     left = 0
# # #     for i in range(len(str1) + 1):
# # #         if str1[left:i + 1] in str2:
# # #             res = str1[left:i + 1]
# # #         else:
# # #             left += 1
# # #     return res
# # #
# # #
# # # print(lcs('hellopython', 'goodlopylike'))
# # #
# # #
# # # class Solution:
# # #     def twoSum(self, nums: List[int], target: int) -> List[int]:
# # #         for i in range(len(nums) - 1):
# # #             for j in range(i + 1, len(nums)):
# # #                 if nums[i] + nums[j] == target:
# # #                     return [i, j]
# # #         return []
# # #
# # #
# # # # #
# # # # # ## using recursion to define Palindrome
# # # # # """ PreC: n is a string
# # # # # """
# # # # #
# # # # #
# # # def isPalindrome(n):
# # #     if len(n) == 1:
# # #         return True
# # #     else:
# # #         return n[0] == n[len(n) - 1] and isPalindrome(n[1:len(n) - 1])
# # #
# # #
# # # print(isPalindrome('abcba'))
# # #
# # #
# # # # ## 编写程序，输入三组数据，判断能否构成三角形的三条边。
# # # def isRightTriangle(a, b, c):
# # #     lis = []
# # #     m = a ** 2
# # #     q = b ** 2
# # #     p = c ** 2
# # #     lis.append(m)
# # #     lis.append(q)
# # #     lis.append(p)
# # #     lis.sort()
# # #     if lis[0] + lis[1] == lis[2]:
# # #         return True
# # #     return False
# # #
# # #
# # # print(isRightTriangle(8, 8, 10))
# # #
# # # from copy import copy
# # #
# # #
# # # # 输入一个正整数，输出它的所有质数因子（如180的质数因子为2、2、3、3、5）。
# # # def findPrime(n):
# # #     factor = 2
# # #     res = []
# # #     while n != 2:
# # #         if n % 2 == 0:
# # #             n = n / factor
# # #             res.append(factor)
# # #         else:
# # #             factor += 1
# # #     return res
# # #
# # #
# # # print(findPrime(14))
# # #
# # # list = [2, 3, 4, 5, 6]
# # # new_list = copy(list)
# # # print(new_list)
# # #
# # #
# # # # 猴子吃桃问题：猴子第一天摘下若干个桃子，当即吃了一半，还不瘾，又多吃了一个第二天早上又将剩下的桃子吃掉一半，
# # # # 又多吃了一个。以后每天早上都吃了前一天剩下的一半零一个。到第10天早上想再吃时，见只剩下一个桃子了。求第一天共摘了多少。
# # # def peaches_problem(n, m):
# # #     total = 0
# # #     for i in range(n):
# # #         total += (2 * m + 1)
# # #     return total
# # #
# # #
# # # print(peaches_problem(10, 1))
# # #
# # #
# # # # 请输入星期几的第一个字母来判断一下是星期几，如果第一个字母一样，则继续判断第二个字母。
# # # # 星期一 Monday、星期二Tuesday、星期三 Wednesday、星期四 Thursday、星期五 Friday、星期六 Saturday、星期日Sunday.
# # # def which_day(s):
# # #     dic = {'M': 'Monday', 'T': 'Tuesday', 'W': 'Wednesday', 'T': 'Thursday',
# # #            'Sa': 'Saturday', 'Su': 'Sunday'}
# # #     if s in dic:
# # #         return dic[s]
# # #     return None
# # #
# # #
# # # print(which_day('S'))
# # #
# # #
# # # # 给定俩变量，进行交换
# # # def swap_variables(m, n):
# # #     med = m
# # #     m = n
# # #     n = med
# # #     return m, n
# # #
# # #
# # # print(swap_variables(2, 5))
# # #
# # #
# # # # 编写程序，生成随机数。(random - [0,1) return a float number /randint - [a,b] return an integer /randrange - [a,b,c) c - step )
# # #
# # # # 输入摄氏度，将其转为华氏度。
# # # # 输入华氏度，将其转沩摄氏度。
# # # # 华氏温度与摄氏温度转换公式为：
# # # # 华氏温度=摄氏温度×1.8+32。
# # # def Celsius_Fahrenheit(m, n):
# # #     to_fa = m * 1.8 + 32
# # #     to_ce = int((n - 32) / 1.8)
# # #     return to_fa, to_ce
# # #
# # #
# # # print(Celsius_Fahrenheit(32, 80))
# # #
# # #
# # # ## 给定一个正整数，请你判断这个数是不是快乐数。
# # # # 快乐数：对于一个正整数，每次把他替换为他每个位置上的数字的平方和，如果这个数能变为1则是快乐数，如果不可能变成1则不是快乐数。
# # # # 例如：正整数19
# # # # 转换过程为1*1+9*9=82，8*8+2*2=68，6*6+8*8=100,1*1+0*0+0*0=1，所以他是快乐数。
# # # def new_sum(n):
# # #     sum_digit = 0
# # #     while n > 0:
# # #         sum_digit += ((n % 10) ** 2)
# # #         n = n // 10
# # #     return sum_digit
# # #
# # #
# # # def isHappyNum(n):
# # #     while n > 6:
# # #         n = new_sum(n)
# # #     if n == 1:
# # #         return True
# # #     return False
# # #
# # #
# # # print(isHappyNum(7))
# # #
# # #
# # # # 给你一个大小为n的字符串数组strs，其中包含
# # # # n个字符串，编写一个函数来查找字符串数组中的最长公共前缀，返回这个公共前缀。
# # # # 输入：
# # # # ["abca", "abc", "abca", "abc", "abcc"]
# # # # 输出：
# # # # "abc"
# # # def lcs(strs):
# # #     n = len(strs)
# # #     if n == 0:
# # #         return ''
# # #     for i in range(len(strs[0])):
# # #         temp = strs[0][i]
# # #         for j in range(1, n):
# # #             if i == len(strs[j]) or temp != strs[j][i]:
# # #                 return strs[0][: i]
# # #     return strs[0]
# # #
# # #
# # # print(lcs(['abc', 'abcde', 'abcddd']))
# # #
# # #
# # # # 给定一个长度为n的无序数组，包含正数、负数和0，请从中找出3个数，使得乘积最大，返回这个乘积。
# # # def max_product(list):
# # #     list.sort()
# # #     return max(list[-3] * list[-2] * list[-1], list[0] * list[1] * list[-1])
# # #
# # #
# # # print(max_product([-5, -3, -1, 0, 1, 5, 75, 32, 64]))
# #
# # ## 给定一个数组，请你实现将所有0移动到数组末尾并且不改变其他数字的相对顺序。
# # def remove_zeroes(arr):
# #     non_zero_index = 0  # 指针指向非0数
# #     for i in range(len(arr)):
# #         if arr[i] != 0:
# #             arr[non_zero_index] = arr[i]  # 让指针指向非0的数
# #             if i != non_zero_index:  # 如果非0位置和i位置不一致，将i位置的数换成0
# #                 arr[i] = 0
# #             non_zero_index += 1
# #     return arr
# #
# #
# # print(remove_zeroes([1, 2, 3, 0, 53, 9, 4, 0]))
# #
# # 在柠檬水摊上，每一杯柠檬水的售价为5美元。
# # 顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯。
# # 每位顾客只买一杯柠檬水，然后向你付5美元、10美元或20美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付5美元。
# # 注意，一开始你手头没有任何零钱。
# # 如果你能给每位顾客正确找零，返回true，否则返回 false
# def canChanged(bills):
#     five = 0
#     ten = 0
#     for bill in bills:
#         if bill == 5:
#             five += 1
#         elif bill == 10:
#             if five > 0:
#                 five -= 1
#                 ten += 1
#             else:
#                 return False
#         else:
#             if five > 0 and ten > 0:
#                 ten -= 1
#                 five -= 1
#             elif five >= 3:
#                 five -= 3
#             else:
#                 return False
#     return True
#
#
# print(canChanged([5, 5, 5, 10, 20]))

# 给定两个字符串，判断其中一个字符串是否为另一个字符串的置换。
# 置换的意思是，通过改变顺序可以使得两个字符串相等。
# def isSwap(a, b):
#     a.sort()
#     b.sort()
#     return a == b

## 冒泡排序
# def bubbleSort(list):
#     for i in range(len(list)):
#         for j in range(len(list) - 1 - i):
#             if list[j] > list[j + 1]:
#                 list[j], list[j + 1] = list[j + 1], list[j]
#     return list
#
#
# print(bubbleSort([6, 53, 1, 73, 16]))

## 选择排序
# def selection_sort(list):
#     for i in range(len(list) - 1):
#         min_index = i
#         for j in range(i+1, len(list)):
#             if list[min_index] > list[j]:
#                 min_index = j
#         list[i], list[min_index] = list[min_index], list[i]
#     return list
#
#
# print(selection_sort([1,5,6,82,2,53]))
