def func(key, lock, len_k, len_l):
    for i in range(-len_k + 1, len_l):
        for j in range(-len_k + 1, len_l):
            ret = 1
            for k in range(len_k):
                for l in range(len_k):
                    if 0 <= k + i < len_l and 0 <= l + j < len_l:
                        if key[k][l] == lock[k + i][l + j]:
                            ret = 0
                            break
                if not ret:
                    break
            else:
                for k in range(len_l):
                    for l in range(len_l):
                        if i <= k < i + len_k and j <= l < j + len_k:
                            continue
                        elif lock[k][l] == 0:
                            ret = 0
                            break
                    if not ret:
                        break
                else:
                    return 1
    return 0


def solution(key, lock):
    answer = 0
    len_k, len_l = len(key), len(lock)
    rotate_key = [[key[i][j] for j in range(len_k)] for i in range(len_k)]

    # 4방향
    for i in range(4):
        answer = func(rotate_key, lock, len_k, len_l)
        rotate_key = [[rotate_key[len_k - 1 - j][i] for j in range(len_k)] for i in range(len_k)]
        if answer:
            break
    return answer