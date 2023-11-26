
import random
import os
import msvcrt  # For Windows-specific keyboard input


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    width = 20
    height = 10
    snake = [[4, 4]]
    food = [random.randint(1, width - 1), random.randint(1, height - 1)]
    score = 0

    while True:
        clear_screen()

        for y in range(height):
            for x in range(width):
                if [x, y] in snake:
                    print("*", end=" ")
                elif x == food[0] and y == food[1]:
                    print("F", end=" ")
                else:
                    print(" ", end=" ")
            print()

        key = msvcrt.getch().decode('utf-8')
        new_head = snake[0][:]  # Copy the head position

        if key == 'w' and snake[0][1] > 0:
            new_head[1] -= 1
        elif key == 's' and snake[0][1] < height - 1:
            new_head[1] += 1
        elif key == 'a' and snake[0][0] > 0:
            new_head[0] -= 1
        elif key == 'd' and snake[0][0] < width - 1:
            new_head[0] += 1

        snake.insert(0, new_head)

        if snake[0] == food:
            score += 1
            food = [random.randint(1, width - 1),
                    random.randint(1, height - 1)]
        else:
            snake.pop()

        if len(snake) != len(set(tuple(x) for x in snake)):
            print("Game Over. Your Score:", score)
            break


if __name__ == '__main__':
    main()
