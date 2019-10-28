
n = int(input())
arr = list()
for i in range(n):
    arr.append(list(map(int, input().split())))
def func2(ret, x, i):
    for y, j in enumerate(ret):
        if arr[x][y] != i*j:
            return 0
    return 1
def func(ret, x):
    print(ret)
    if x==n:
        return ret
    for i in range(1, max(arr[x]) + 1):
        if func2(ret, x, i):
            temp = func(ret+[i], x+1)
            if temp:
                return temp
    return 0
for i in func([], 0):
    print(i, end=" ")