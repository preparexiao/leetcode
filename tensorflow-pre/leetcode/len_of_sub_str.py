

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        a = [0 for k in range(128)]
        num=0
        if len(s)==0:
            # print("len")
            return 0
        for i in range(len(s)):
            t=0
            # print(ord(s[i]))
            # print("-------",s[i])
            for j in range(i,len(s)):
                if a[ord(s[j])] == 0:
                    # print(s[j])
                    t = t + 1
                    a[ord(s[j])] = 1
                else:
                    # print("temp_end")
                    if t > num:
                        num = t
                    a = [0 for k in range(128)]
                    break
            if t>num:
                num=t
        return num

a="abcabcbb"
b=" "
print(isinstance(a,str))
ss=Solution()
print(ss.lengthOfLongestSubstring(a))

