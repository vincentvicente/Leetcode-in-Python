class Solution:
    def minimumDeletions(self, s: str) -> int:
        n = len(s)
        B = [0] * n
        cnt_a, cnt_b = 0, 0
        # count number of b before index i
        for i in range(n):
            if s[i] == "b":
                cnt_b += 1

            B[i] = cnt_b

        res = n
        for i in range(n - 1, -1, -1):
            res = min(res, B[i] + cnt_a)
            if s[i] == "a":
                cnt_a += 1

        return res
