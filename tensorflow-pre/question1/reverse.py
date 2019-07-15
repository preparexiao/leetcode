#coding:utf-8
class Solution:
    def adapt_to_n(self,s:str,n:int)->int:
        if n<101:
            return self.longestPalindrome(s)
        elif n<1001:
            return self.longestPalindrome(s)
        else:
            return self.longestPalindrome(s)
    def longestPalindrome(self, s: str) -> str:
        if len(s)==0:
            return ""
        lenth=1
        rs=s[0]
        re=""
        t=0
        # print(len(s))
        for i in range(len(s)):
            # print("-------",i)
            t=1
            re=s[i]
            if i>0 and i+1<len(s):
                # print(i-1,i+1)
                # print("判断左右相等")
                if s[i-1]==s[i+1]:
                    # print("相等")
                    t=1
                    for k in range(1, i + 1):
                        if i + k < len(s):
                            if s[i - k] == s[i + k]:
                                # print(i-k,i+k, s[i - k], s[i + k])
                                t = t + 2
                                re = s[i - k:i + k + 1]
                                # print(re)
                            else:
                                break
            if t>lenth:
                # print("换",i,t)
                lenth=t
                rs=re
            t = 0
            if i + 1 < len(s):
                # print("判断相邻相等")
                if s[i]==s[i+1]:
                    # print("相等")
                    # if i==2:
                    #     print("i=2",s[i],s[i+1])
                    t=2
                    re = s[i:i+2]
                    for k in range(1, i + 1):
                        # print("k",k)
                        if i + k +1< len(s):
                            # print(s[i-1],s[i+k+1])
                            if s[i-k] == s[i + k+1]:
                                # print("k循环相等","(",s[i-1],s[i + k+1],")",i-1,)
                                t = t + 2
                                re = s[i - k:i + k + 2]
                                # print(re)
                            else:
                                break

            if t>lenth:
                # print("换2", i,t)
                lenth=t
                rs=re

        return rs
    def longestPalindrome_1000_to_10000(self, s: str,start,end) -> int:
        if len(s)==0:
            return 0
        length=1
        if start==end:
            return 1
        elif start>end:
            return 0
        elif s[start]==s[end]:
            # print(s[start],s[end])
            return self.longestPalindrome_1000_to_10000(s,start+1,end-1)+2
        else:
            start_1=self.longestPalindrome_1000_to_10000(s,start+1,end)
            end_1=self.longestPalindrome_1000_to_10000(s,start,end-1)
            if start_1>end_1:
                return start_1
            else:
                return end_1

ss=Solution()
# file=open("input")
# out=open("output",'w')
# length=int(file.readline().strip())
# test=file.readline()
# result=ss.adapt_to_n(test,length)
# print(result)
# out.write(str(result))
# print("--------END--------")
test1="babadada"
# test2=""
# test3="a"
# test4="aAAa"
# test5="aAAAAA"
print(ss.longestPalindrome(test1))
# print(ss.longestPalindrome(test2))
# print(ss.longestPalindrome(test3))
# print(ss.longestPalindrome(test4))
# print(ss.longestPalindrome(test5))
#
#
