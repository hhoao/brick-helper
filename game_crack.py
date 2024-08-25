import os
import shutil
import time
import cv2
import numpy as np
import algorithms as algorithms


def get_matrix(image_path, row, column, crop_width, crop_height, generate_image=False):
    img1 = cv2.imread(image_path)
    height = img1.shape[0]
    width = img1.shape[1]
    dx = height / row
    dy = width / column
    images = [[] for _ in range(row)]
    flatten = []

    for i in range(row):
        for j in range(column):
            x = int(dx * i)
            y = int(dy * j)
            next_x = int(x + dx)
            next_y = int(y + dy)
            clip = algorithms.round_clip(img1[x:next_x, y:next_y], crop_width, crop_height)
            images[i].append(clip)
            flatten.append(clip)

    category = []
    i = 0
    while i < len(flatten):
        j = i + 1
        category.append(flatten[i])
        while j < len(flatten):
            if algorithms.is_similarity(flatten[i], flatten[j]):
                del flatten[j]
            else:
                j += 1
        i += 1

    print("Category Count: %s" % len(category))

    root_path = os.getcwd()
    target_path = root_path + "/target"
    category_path = target_path + "/category"
    grouped_path = target_path + "/grouped"
    if generate_image:
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
            os.makedirs(target_path)
            os.makedirs(category_path)
            os.makedirs(grouped_path)

    matrix = np.zeros((row, column), int)

    for i, category_item in enumerate(category):
        if generate_image:
            cv2.imwrite(category_path + "/" + str(i) + ".png", category_item)
            dest_dir = grouped_path + "/" + str(i)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

    for k in range(row):
        for j in range(column):
            for i, category_item in enumerate(category):
                img = images[k][j]
                if algorithms.is_similarity(img, category_item):
                    if generate_image:
                        dest_dir = grouped_path + "/" + str(i)
                        cv2.imwrite(dest_dir + "/" + str((k + 1) * column + (j + 1)) + ".png", img)
                    matrix[k][j] = i
    return matrix


def find_end_point(matrix, x, y, direction):
    while True:
        dx = x + direction[0]
        dy = y + direction[1]
        if 0 <= dx < len(matrix) and 0 <= dy < len(matrix[0]):
            if matrix[dx][dy] == -1:
                end_point = [x, y]
                break
            else:
                x = dx
                y = dy
        else:
            end_point = [x, y]
            break
    return end_point


def find_nearest_remote_point(matrix, x, y, direction):
    while True:
        dx = x + direction[0]
        dy = y + direction[1]
        if 0 <= dx < len(matrix) and 0 <= dy < len(matrix[0]):
            if matrix[dx][dy] == -1:
                x = dx
                y = dy
            else:
                nearest_remote_point = [x, y]
                break
        else:
            nearest_remote_point = [x, y]
            break
    return nearest_remote_point


def get_direction_distance(x, y, matrix, direction):
    end_point = find_end_point(matrix, x, y, direction)
    nearest_remote_point = find_nearest_remote_point(matrix, end_point[0], end_point[1], direction)
    return [nearest_remote_point[0] - end_point[0], nearest_remote_point[1] - end_point[1]]


def is_valid(matrix, dx, dy):
    return 0 <= dx < len(matrix) and 0 <= dy < len(matrix[0])


def find_same_block(matrix, x, y, nx, ny, block):
    for direction in directions:
        dx = nx + direction[0]
        dy = ny + direction[1]

        while True:
            if dx == x and dy == y:
                break
            if is_valid(matrix, dx, dy):
                if matrix[dx][dy] == -1:
                    dx += direction[0]
                    dy += direction[1]
                elif matrix[dx][dy] == block:
                    return [dx, dy]
                else:
                    break
            else:
                break
    return None


def move_block(matrix, x, y, dx, dy, direction):
    if dx == 0 and dy == 0:
        return
    end_point = find_end_point(matrix, x, y, direction)
    current_x = end_point[0]
    current_y = end_point[1]
    while current_x != x or current_y != y:
        matrix[current_x + dx][current_y + dy] = matrix[current_x][current_y]
        matrix[current_x][current_y] = -1
        current_x = current_x - direction[0]
        current_y = current_y - direction[1]


def is_blocked(matrix, x, y, same_block, direction):
    if same_block is not None and (same_block[0] - x == 0 or same_block[1] - y == 0):
        nx = x
        ny = y
        while True:
            nx += direction[0]
            ny += direction[1]
            if nx == same_block[0] and ny == same_block[1]:
                break
            if is_valid(matrix, nx, ny):
                if matrix[nx][ny] == -1:
                    continue
                else:
                    return True
            else:
                break
    return False


def try_move_block(matrix, x, y, dx, dy, direction):
    block = matrix[x][y]

    if direction[0] == 0:
        to_x = False
        real_distance = dy
    else:
        to_x = True
        real_distance = dx

    if real_distance > 0:
        start = 0
        end = real_distance + 1
    else:
        start = real_distance
        end = 1

    for i in range(start, end):
        if to_x:
            move_x = i
            move_y = 0
        else:
            move_x = 0
            move_y = i
        same_block = find_same_block(matrix, x, y, x + move_x, y + move_y, block)
        blocked = is_blocked(matrix, x, y, same_block, direction)
        if blocked:
            continue

        if same_block is not None:
            matrix[same_block[0]][same_block[1]] = -1
            matrix[x][y] = -1
            move_block(matrix, x, y, move_x, move_y, direction)
            return same_block
    return None


def is_end(total_steps, current_steps):
    return current_steps >= total_steps


# 消去两个方块的条件：
# 1. 一个方块的上下左右有相同的方块
# 2. 一个方块往上下左右移动后的位置的上下左右有相同的方块
#
# 移动的条件：
# 如果要将一个方块向一个方向移动x格距离，那么该方块在那个方向连续的几个方块的那个方向必须有x格空位置
#
# 推箱子规则：
# 将一个方块朝着一个方向移动x格距离，那么该方块该方向的相连的所有方块都需要移动x距离
def game_start(matrix):
    row = len(matrix)
    column = len(matrix[0])
    total_steps = int(row * column / 2)
    current_steps = 0
    while current_steps < total_steps:
        for x in range(len(matrix)):
            for y in range(len(matrix[0])):
                if matrix[x][y] == -1:
                    continue
                for direction in directions:
                    block = matrix[x][y]
                    if block == -1:
                        break
                    distance = get_direction_distance(x, y, matrix, direction)

                    same_block = try_move_block(matrix, x, y, distance[0], distance[1], direction)
                    if same_block is not None:
                        current_steps += 1
                        print_matrix(matrix, current_steps, [x, y], same_block)
                        time.sleep(0)
                        if is_end(total_steps, current_steps):
                            return


def print_split_line(width, start="", end=""):
    print(start, end="")
    for i in range(width):
        print("——", end="\t")
    print(end)
    pass


def print_serial_number(width, start, end=""):
    print(start, end="\t\t|\t")
    for j in range(width):
        print(j, end="\t")
    print(end)
    pass


def print_matrix(matrix, current_steps, point1=None, point2=None):
    print()
    if point2 is not None and point2 is not None:
        print("Step: %s, Source Block: %s, Target Block: %s" %
              (current_steps, point1, point2))
    else:
        print("Step: %s " % current_steps)
    if not is_print_matrix:
        return
    print_split_line(len(matrix[0]) + 2, start="-  ", end="-")
    print_serial_number(len(matrix[0]), start="|", end="|")
    print_split_line(len(matrix[0]) + 2, start="|  ", end="|")
    for i in range(len(matrix)):
        print("| ", i, end="\t|\t")
        for j in range(len(matrix[0])):
            print(matrix[i][j], end="\t")
        print("|")
    print_split_line(len(matrix[0]) + 2, start="-  ", end="-")


def main():
    image_path = "pictures/scene.png"
    row = 14
    column = 10
    crop_width = 3
    crop_height = 3
    matrix = get_matrix(image_path, row, column, crop_height, crop_width, True)
    print_matrix(matrix, 0)
    game_start(matrix)


if __name__ == "__main__":
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    is_print_matrix = False
    main()
