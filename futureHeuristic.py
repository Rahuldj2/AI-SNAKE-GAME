import time
import random
import pygame
import matplotlib.pyplot as plt

"""
# A* algorithm with future planning has been used in this code to improve the snake game.
# It is used to calculate the direction for the snake to move towards the food.
The future_snake part in the code is an improvement over the A* algorithm because it considers
potential future collisions with itself. By simulating the snake's movement and checking if the move 
would lead to a collision with itself in subsequent iterations, it avoids making moves that would 
result in the snake trapping itself.

This approach takes into account the snake's current position, the potential future score increase, and
the distance to the food. It calculates a weighted distance for each possible move, where the weight is
determined by the sum of the Manhattan distance to the food and the future score increase.
"""

# Define the grid size
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

delay = 0.001

# Create the snake as a list of coordinates
snake = [(UNIT_SIZE * 7, UNIT_SIZE * 7)]

# Initialize the snake direction
snake_direction = "right"

# Score
score = 0
high_score = 0
game_num=1
avg_score = 0
sum_score = 0
# Update the UI
pygame.init()
screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
pygame.display.set_caption("Snake Game")

# Snake food
food = random_food(snake)

# Define a function to calculate the direction to move towards the food
import heapq
# Define a function to calculate the direction to move towards the food
# Define a function to calculate the direction to move towards the food
# Define a function to calculate the direction to move towards the food
def move_towards_food(snake, food):
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # Get the current head position
    head_x, head_y = snake[0]

    # Calculate the distance to the food using the Manhattan distance (sum of horizontal and vertical distances)
    distance_to_food = manhattan_distance((head_x, head_y), food)

    # Create a list of possible moves
    possible_moves = []

    for direction in ["up", "down", "left", "right"]:
        if direction == "up":
            new_head = (head_x, head_y - UNIT_SIZE)
        elif direction == "down":
            new_head = (head_x, head_y + UNIT_SIZE)
        elif direction == "left":
            new_head = (head_x - UNIT_SIZE, head_y)
        elif direction == "right":
            new_head = (head_x + UNIT_SIZE, head_y)

        # Check if the new head position is valid
        if (
            0 <= new_head[0] < GRID_WIDTH
            and 0 <= new_head[1] < GRID_HEIGHT
            and new_head not in snake[1:]
        ):
            # Estimate the potential future score increase
            future_score_increase = int(
                manhattan_distance(new_head, food) < distance_to_food
            )

            # Consider collisions with itself in subsequent iterations
            future_snake = snake[:]
            future_snake.insert(0, new_head)

            # Check if the move would lead to a collision with itself
            if not any(
                future_snake[0] == part for part in future_snake[1:]
            ):
                score_weighted_distance = (
                    manhattan_distance(new_head, food) + future_score_increase
                )
                possible_moves.append((direction, score_weighted_distance))

    # Sort the possible moves by the sum of distance to food and self-collision avoidance
    if not possible_moves:
        return None

    possible_moves.sort(key=lambda move: move[1])

    # Return the direction of the first move in the sorted list
    return possible_moves[0][0]


# Font for the score display
font = pygame.font.Font(None, 36)

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            plt.plot(list(map(int, game_numbers)), list(map(int, scores)))
            plt.xlabel('Game Number')
            plt.ylabel('Score')
            plt.title('Snake Game Score Plot A * future heuristic')
            plt.show()
            pygame.quit()
            exit()

    # Calculate the direction to move towards the food
    snake_direction = move_towards_food(snake, food)
    if (snake_direction is None):
        game_numbers.append(game_num)
        scores.append(score)
        print(score)
        time.sleep(1)
        snake = [(UNIT_SIZE * 7, UNIT_SIZE * 7)]
        snake_direction = "right"
        sum_score += score
        score = 0
        avg_score = sum_score/game_num
        game_num += 1


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

    # Check for a collision with the border
    if (
        snake[0][0] < 0
        or snake[0][0] >= GRID_WIDTH
        or snake[0][1] < 0
        or snake[0][1] >= GRID_HEIGHT
    ):
        # Reset the game
        game_numbers.append(game_num)
        scores.append(score)
        print(score)
        time.sleep(1)
        snake = [(UNIT_SIZE * 7, UNIT_SIZE * 7)]
        snake_direction = "right"
        sum_score += score
        score = 0
        avg_score = sum_score/game_num
        game_num += 1

    # Check for head collision with the body
    if snake[0] in snake[1:]:
        game_numbers.append(game_num)
        scores.append(score)
        print(score)
        # Reset the game
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
    # score_text = font.render(f"Score: {score} High Score: {high_score}", True, (255, 255, 255))
    # score_text = font.render(f"Game Num: {game_num} Score: {score} High Score: {high_score}", True, (255, 255, 255))
    # screen.blit(score_text, (10, 10))

    font = pygame.font.Font(None, 24)
    score_text = font.render(f"Game Num: {game_num}  Score: {score}  High Score: {high_score}  Avg Score: {avg_score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))
    pygame.display.update()

    # Delay for smooth movement
    time.sleep(delay)
