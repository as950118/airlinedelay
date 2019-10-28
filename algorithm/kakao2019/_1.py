def func2(s, cur, jmp, flag):
    if len(s) <= cur+jmp:
        if flag != 1:
            return str(flag) + s[cur:]
        else:
            return s[cur:]
    if s[cur:cur+jmp] == s[cur+jmp:cur+jmp+jmp]:
        return func2(s, cur + jmp, jmp, flag+1)
    else:
        if flag != 1:
            return str(flag) + s[cur-jmp:cur] + func2(s,cur+jmp, jmp, 1)
        else:
            return s[cur] + func2(s, cur + 1, jmp, 1)

def func(s, cur, jmp, flag):
    if len(s) <= cur+jmp:
        if flag != 1:
            return len(str(flag)) + len(s) - cur
        else:
            return len(s) - cur
    if s[cur:cur+jmp] == s[cur+jmp:cur+jmp+jmp]:
        return func(s, cur + jmp, jmp, flag+1)
    else:
        if flag != 1:
            return len(str(flag)) + jmp + func(s,cur+jmp, jmp, 1)
        else:
            return 1 + func(s, cur + 1, jmp, 1)

def solution(s):
    len_s = len(s)
    answer = len_s
    for i in range(1, (len_s+1//2) + 1):
        if s[0:i] == s[i:i+i]:
            print(i,func2(s,0,i,1))
            answer = min(answer, func(s,0,i,1))
    return answer

print(solution('aaaaaaaaaabbbbbbbbbbaaaaaaaaaabbbbbbbbbb'))
print(solution('ababcdcdababcdcd'))
print(solution('abcabcdede'))
print(solution('abcabcabcabcdededededede'))
print(solution('xababcdcdababcdcd'))