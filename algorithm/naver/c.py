import sys
sys.setrecursionlimit(10**8)
dist = [0] #시간, cook_times
seq = [[] for i in range(1001)] #순서, order
dp = [-1 for i in range(1001)] #방문 검사를 위해

def func(k, cur):
    #더이상 필요한게 없다면
    if not seq[k]:
        return dist[k], cur

    #이미 한일이라면
    if dp[k] != -1:
        return dp[k], cur

    #해야할일들을 모두 검사
    for next_k in seq[k]:
        temp, temp_cur = func(next_k, cur+1)
        if dp[k] < temp:
            dp[k] = temp
            cur = temp_cur

    #현재 하는 일이 필요로 하는 시간
    dp[k] += dist[k]

    #필요 시간과 몇번단계를 지나왔는지
    return dp[k], cur

def solution(cook_times, order, k):
    global dist

    #편의상 1부터 사용하기 위해
    dist += cook_times
    #건물 순서 mapping 시켜주기
    for a, b in order:
        seq[b].append(a)

    #recursion을 위한 함수
    ret = func(k, 0)

    #결과출력
    return([ret[1], ret[0]])