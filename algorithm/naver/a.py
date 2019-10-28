def solution(records):
    temp = []
    answer = []
    #R, S, D로 나누어서 판단
    for record in records:
        if record[0] == "R":
            #메일주소는 8번째부터 나옴
            mail = record[8:]
            temp.append(mail)
            continue

        if record[0] == "S":
            #영구보관함에 저장하고 초기화
            answer += temp
            temp = []
            continue

        if record[0] == "D":
            #임시보관함에 있다면 제거
            if temp:
                temp.pop()
            #아니라면 그냥 진행
            else:
                pass
            continue
    return answer