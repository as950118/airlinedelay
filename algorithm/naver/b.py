import math
maxnum = 10**12
sqrt_maxnum = int(math.sqrt(maxnum))
def solution(n):
    arr = set() #중복을 피하기위해
    for i in range(1, sqrt_maxnum):
        j = i+1
        temp = i*j
        while temp<maxnum:
            arr.add(temp)
            j += 1
            temp *= j
    arr = list(arr)
    arr.sort()

    return arr[n-1]