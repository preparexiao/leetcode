class Solution:
    def order(self, s: tuple) -> tuple:#左边为奇数右边为偶数
        start=0
        end=len(s)-1
        while True:
            print(s)
            print("循环",start,end)
            if start == end or start + 1 == end:
                return s
            if s[start] % 2 == 1 and s[end] % 2 == 0:
                start=start+1
                end=end-1
            elif s[start] % 2 == 0 and s[end] % 2 == 0:
                end=end-1
                # print(s)
                # print(start,end)
                temp=s[start]
                s[start]=s[end]
                s[end]=temp
            elif s[start] % 2 == 0 and s[end] % 2 == 1:
                temp=s[start]
                s[start]=s[end]
                s[end]=temp
                start = start + 1
                end = end - 1
            elif s[start] % 2 == 1 and s[end] % 2 == 1:
                start=start+1
test1=[33,42,32,5,6,8]
test2=[33]
test3=[33,42,32,5,6,8,9]
ss=Solution()
print(ss.order(test1))
print(ss.order(test2))
print("test3")
print(ss.order(test3))


