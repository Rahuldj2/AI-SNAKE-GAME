
import time
import random
import pygame
import matplotlib.pyplot as plt
"""
This file implements the Snake Game using a greedy best-first search algorithm to move the snake towards the food.
The snake's movement is determined by calculating the direction that leads to the shortest path towards the food,
ignoring collisions with itself. The game is played on a grid of size 15x15, and the snake's initial position is at
(7, 7). The game keeps track of the score, high score, average score, and the number of games played. The snake
grows by eating the food, and the game ends if the snake collides with the border or itself. The game is displayed
using the Pygame library, and the score plot is generated using the Matplotlib library.
"""


# Defining the grid size 15x15
GRID_SIZE = 15
GRID_WIDTH = 450  # Adjusted for 15x15 grid
GRID_HEIGHT = 450  # Adjusted for 15x15 grid

# Calculate the unit size based on the grid size
UNIT_SIZE = GRID_WIDTH // GRID_SIZE

game_numbers = []
scores = []


# Set the initial food placement
def random_food(snake):
    while True:
        x = random.randint(0, GRID_SIZE - 1) * UNIT_SIZE
        y = random.randint(0, GRID_SIZE - 1) * UNIT_SIZE
        if (x, y) not in snake:
            return (x, y)

delay = 0.1

# Creating the snake as a list of coordinates
snake = [(UNIT_SIZE * 7, UNIT_SIZE * 7)]

# Initialize the snake direction
snake_direction = "right"

# Score
score = 0
high_score = 0
game_num = 1
avg_score = 0
sum_score = 0

pygame.init()
screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
pygame.display.set_caption("Snake Game")

# Snake food
food = random_food(snake)

# Define a function to calculate the direction to move towards the food
#This follows a greedy approach and takes shortest path towards food ignoring collisions with self
def move_towards_food(snake, food):
    head_x, head_y = snake[0]
    food_x, food_y = food
    if head_x < food_x:
        return "right"
    elif head_x > food_x:
        return "left"
    #since pygame coordinate system is top to bottom increasing y and left to right X
    elif head_y < food_y:
        return "down"
    else:
        return "up"

# Font for the score display
font = pygame.font.Font(None, 36)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            plt.plot(list(map(int, game_numbers)), list(map(int, scores)))
            plt.xlabel('Game Number')
            plt.ylabel('Score')
            plt.title('Snake Game Score Plot')
            plt.show()
            exit()

    # Calculate the direction to move towards the food
    snake_direction = move_towards_food(snake, food)

    # Update snake position
    if snake_direction == "right":
        new_head = (snake[0][0] + UNIT_SIZE, snake[0][1])
    elif snake_direction == "left":
        new_head = (snake[0][0] - UNIT_SIZE, snake[0][1])
    elif snake_direction == "up":
        new_head = (snake[0][0], snake[0][1] - UNIT_SIZE)
    elif snake_direction == "down":
        new_head = (snake[0][0], snake[0][1] + UNIT_SIZE)

    snake.insert(0, new_head)

    # Check for a collision with the food
    if snake[0] == food:
        food = random_food(snake)
        score += 1
        if score > high_score:
            high_score = score
    else:
        snake.pop()

    # collision with the border
    if (
        snake[0][0] < 0
        or snake[0][0] >= GRID_WIDTH
        or snake[0][1] < 0
        or snake[0][1] >= GRID_HEIGHT
    ):
        # Reset the game
        game_numbers.append(game_num)
        scores.append(score)
        time.sleep(1)
        snake = [(UNIT_SIZE * 7, UNIT_SIZE * 7)]
        snake_direction = "right"
        sum_score += score
        score = 0
        avg_score = sum_score/game_num
        game_num += 1
        

    # Check for head collision with the body
    if snake[0] in snake[1:]:
        # Reset the game
        game_numbers.append(game_num)
        scores.append(score)
        time.sleep(1)
        snake = [(UNIT_SIZE * 7, UNIT_SIZE * 7)]
        snake_direction = "right"
        sum_score += score
        score = 0
        avg_score = sum_score/game_num
        game_num += 1

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the grid
    for x in range(0, GRID_WIDTH, UNIT_SIZE):
        pygame.draw.line(screen, (64, 64, 64), (x, 0), (x, GRID_HEIGHT))
    for y in range(0, GRID_HEIGHT, UNIT_SIZE):
        pygame.draw.line(screen, (64, 64, 64), (0, y), (GRID_WIDTH, y))

    # Draw the snake
    for segment in snake:
        pygame.draw.rect(screen, (0, 255, 0), (segment[0], segment[1], UNIT_SIZE, UNIT_SIZE))

    # Draw the food
    pygame.draw.rect(screen, (255, 0, 0), (food[0], food[1], UNIT_SIZE, UNIT_SIZE))

    # Display the score
    font = pygame.font.Font(None, 24)
    score_text = font.render(f"Game Num: {game_num}  Score: {score}  High Score: {high_score}  Avg Score: {avg_score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    pygame.display.update()

    time.sleep(delay)
