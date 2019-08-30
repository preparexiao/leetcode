
class Solution:
    def longestPalindrome(self, s: str) -> int:
        lenth=0
        rs=s
        re=""
        # print(len(s))
        for i in range(len(s)):
            # print("-------",i)
            t=1
            re=s[i]
            if i>0 and i+1<len(s):
                # print(i-1,i+1)
                if s[i-1]==s[i+1]:
                    for k in range(1, i + 1):
                        # print(k)
                        if i + k < len(s):
                            if s[i - k] == s[i + k]:
                                t = t + 2
                                re = s[i - k:i + k + 1]
                                # print(re)
            elif i + 1 < len(s):
                if s[i]==s[i+1]:
                    t=2
                    re = s[i:i+2]
                    for k in range(1, i + 1):
                        # print(k)
                        if i + k +1< len(s):
                            # print(s[i-1],s[i+k+1])
                            if s[i-1] == s[i + k+1]:
                                t = t + 2
                                re = s[i - k:i + k + 2]
                                print(re)

            else:
                    break
            if t>lenth:
                lenth=t
                rs=re
        return lenth

ss=Solution()
file=open("input_1")
length=int(file.readline().strip())
print("length",length)
print(ss.longestPalindrome(file.readline()))
print("--------END--------")
test1="bb"
test2=""
test3="Ab3bd"
print(ss.longestPalindrome(test1))
print(ss.longestPalindrome(test2))
print(ss.longestPalindrome(test3))


