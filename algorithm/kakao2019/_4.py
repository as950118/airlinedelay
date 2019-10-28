import re

def solution(words, queries):
    #queries = [re.compile(''.join(['.' if d=='?' else d for d in q ])+'$') for q in queries]
    #queries = [re.compile('.{%d}%s' %(q.count('?'), q[q.index('?')+q.count('?'):]) if q.index('?')==0 else '%s.{%d}' %(q[:q.index('?')], q.count('?')) + '$') for q in queries]
    #answer = [sum([1 for w in words if q.match(w)]) for q in queries]
    #answer = [0 for i in range(len(queries))]
    queries = [(q.index('?'), q.count('?'), q[q.count('?'):] if q.index('?')==0 else q[:len(q)-q.count('?')]) for q in queries]
    answer = [sum([1 if len(w)==len(q)+c and i==0 and w[c:]==q or len(w)==len(q)+c and i!=0 and w[:len(q)]==q else 0 for w in words]) for i,c,q in queries]
    return answer
print(solution(["frodo", "front", "frost", "frozen", "frame", "kakao"], ["fro??", "????o", "fr???", "fro???", "pro?"]))