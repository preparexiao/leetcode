# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1
        val = l1.val + l2.val
        ansnode = ListNode(val % 10)
        ansnode.next = self.addTwoNumbers(l1.next, l2.next)

        if val >= 10:
            ansnode.next = self.addTwoNumbers(ListNode(1), ansnode.next)
        return ansnode
x=ListNode(1)
y=ListNode(2)
ss=Solution()
ss.addTwoNumbers(x,y)