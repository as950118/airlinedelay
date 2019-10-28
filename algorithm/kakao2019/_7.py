def solution(board):
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    answer = 0
    len_b = len(board)
    visit = {(0, 1, 0)}
    queue = [(0, 1, 0)]

    while 1:
        if (len_b - 1, len_b - 1, 0) in visit or (len_b - 1, len_b - 1, 1) in visit:
            break
        answer += 1

        # dijkstra?
        for _ in range(len(queue)):
            x, y, d = queue.pop(0)
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx - d and nx < len_b and 0 <= ny - 1 + d and ny < len_b:
                    if board[nx][ny] == 0 and board[nx - d][ny - 1 + d] == 0 and (nx, ny, d) not in visit:
                        visit.add((nx, ny, d))
                        queue.append((nx, ny, d))
            if d == 0:
                if x + 1 < len_b and board[x + 1][y] == 0 and board[x + 1][y - 1] == 0:
                    if (x + 1, y - 1, 1) not in visit:
                        visit.add((x + 1, y - 1, 1))
                        queue.append((x + 1, y - 1, 1))
                    if (x + 1, y, 1) not in visit:
                        visit.add((x + 1, y, 1))
                        queue.append((x + 1, y, 1))
                if x - 1 >= 0 and board[x - 1][y] == 0 and board[x - 1][y - 1] == 0:
                    if (x, y - 1, 1) not in visit:
                        visit.add((x, y - 1, 1))
                        queue.append((x, y - 1, 1))
                    if (x, y, 1) not in visit:
                        visit.add((x, y, 1))
                        queue.append((x, y, 1))
            else:
                if y + 1 < len_b and board[x - 1][y + 1] == 0 and board[x][y + 1] == 0:
                    if (x - 1, y + 1, 0) not in visit:
                        visit.add((x - 1, y + 1, 0))
                        queue.append((x - 1, y + 1, 0))
                    if (x, y + 1, 0) not in visit:
                        visit.add((x, y + 1, 0))
                        queue.append((x, y + 1, 0))
                if y - 1 >= 0 and board[x - 1][y - 1] == 0 and board[x][y - 1] == 0:
                    if (x - 1, y, 0) not in visit:
                        visit.add((x - 1, y, 0))
                        queue.append((x - 1, y, 0))
                    if (x, y, 0) not in visit:
                        visit.add((x, y, 0))
                        queue.append((x, y, 0))
    return answer