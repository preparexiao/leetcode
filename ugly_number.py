class Solution:
    def judge(self,x)->bool:
        for i in range(2,x):
            if x%i==0:
                return False
        return True
    def uglynumber(self) -> int:#第1500个丑数
        print("test")
        count=0
        result=[]
        number=1
        result.append(1)
        while True:
            # print("循环",count,len(result))
            number=number+1
            if self.judge(number):
                result.append(number)
            count=count+1
            if len(result)==1500:
                break
        return result[1499]

ss=Solution()
# print(ss.judge(7))
print(ss.uglynumber())




