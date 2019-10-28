from collections import defaultdict
n = int(input())
arr = str(input())
check = defaultdict(lambda:0)
def func():
    if check['o'] and check['n'] and check['e']:
        print(1, end=" ")
        check['o'] -= 1
        check['n'] -= 1
        check['e'] -= 1
        return 1
    if check['z'] and check['e'] and check['r'] and check['o']:
        print(0, end=" ")
        check['z'] -= 1
        check['e'] -= 1
        check['r'] -= 1
        check['o'] -= 1
        return 1
    return 0
for i in arr:
    check[i] += 1
while func():
    continue