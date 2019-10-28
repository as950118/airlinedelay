def check(p):
    n_open=0
    n_close=0
    while 1:
        for i in p:
            if i=='(':
                n_open += 1
            else:
                n_close += 1
            if n_open<n_close:
                return 0
        return 1

def func(p):
    n_open=0
    n_close=0
    ret = ''
    while 1:
        for i in p:
            if i == '(':
                ret += i
                n_open += 1
            else:
                ret += i
                n_close += 1
            if n_open == n_close:
                return ret, p[len(ret):]
        return p, ""


def solution(p):
    if check(p):
        return p
    if len(p)==2:
        u, v = p, ''
    else:
        u, v = func(p)
    if check(u):
        if check(v):
            return "{0}{1}".format(u,v)
        else:
            return "{0}{1}".format(u,solution(v))
    else:
        if check(v):
            return "({0}){1}".format(v, u[::-1][1:-1])
        else:
            return "({0}){1}".format(solution(v), u[::-1][1:-1])
    answer = ''
    return answer

print(solution('(()())()'))
print(solution(')('))
print(solution('()))((()'))