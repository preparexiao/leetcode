import math
nums1 = [1, 2]
nums2 = [3, 4]
class Solution:
    def findMedianSortedArrays(self) -> float:
        i=0
        j=0
        k=0
        if ((len(nums1)+len(nums2))%2==0):
            mid = int((len(nums1) + len(nums2)) / 2) - 1
            mid_ = int((len(nums1) + len(nums2)) / 2)
        else:
            mid = int((len(nums1) + len(nums2)+1) / 2) - 1
            mid_=mid
        mid_num=0
        mid_num2=0
        len1=len(nums1)
        len2=len(nums2)
        # import pdb
        # pdb.set_trace()
        print(mid)
        print(mid_)
        while True:
            if k<mid:
                print()
                if not i<len1:
                    j=j+1
                elif not j<len2:
                    i=i+1
                elif nums1[i]<nums2[j]:
                    i=i+1
                else:
                    j=j+1
                k=k+1
            elif k==mid:
                if not i<len1:
                    mid_num = nums2[j]
                    mid_num2 = mid_num
                    j=j+1
                elif not j<len2:
                    mid_num = nums1[i]
                    mid_num2 = mid_num
                    i=i+1
                elif nums1[i]<nums2[j]:
                    mid_num=nums1[i]
                    mid_num2=mid_num
                    i=i+1
                else:
                    mid_num=nums2[j]
                    mid_num2=mid_num
                    j=j+1
                k=k+1
            elif k==mid_:
                if not i<len1:
                    mid_num2 = nums2[j]
                    j=j+1
                elif not j<len2:
                    mid_num2 = nums1[i]
                    i=i+1
                elif nums1[i]<nums2[j]:
                    mid_num2=nums1[i]
                    i=i+1
                else:
                    mid_num2=nums2[j]
                    j=j+1
                k=k+1
            else:
                break
        return (mid_num+mid_num2)/2.0

        return 0.0
nums1 = [1, 3]
nums2 = [2]
ss=Solution()
print(ss.findMedianSortedArrays())
