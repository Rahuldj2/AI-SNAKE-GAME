import time
import random
import pygame
import heapq

# Define the grid size
GRID_SIZE = 15
GRID_WIDTH = 450  # Adjusted for 15x15 grid
GRID_HEIGHT = 450  # Adjusted for 15x15 grid

# Calculate the unit size based on the grid size
UNIT_SIZE = GRID_WIDTH // GRID_SIZE

# Set the initial food placement
def random_food(snake):
    while True:
        x = random.randint(0, GRID_SIZE - 1) * UNIT_SIZE
        y = random.randint(0, GRID_SIZE - 1) * UNIT_SIZE
        if (x, y) not in snake:
            return (x, y)

delay = 0.4

# Create the snake as a list of coordinates
snake = [(UNIT_SIZE * 7, UNIT_SIZE * 7)]

# Initialize the snake direction
snake_direction = "right"

# Score
score = 0
high_score = 0

# Update the UI
pygame.init()
screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
pygame.display.set_caption("Snake Game")

# Snake food
food = random_food(snake)

# Define a function to calculate the direction to move towards the food
def a_star_search(snake, food):
    def heuristic(node, goal):
        # Manhattan distance as the heuristic
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    open_set = []
    closed_set = set()

    start = snake[0]
    goal = food

    heapq.heappush(open_set, (heuristic(start, goal), start, []))

    while open_set:
        _, current, path = heapq.heappop(open_set)

        if current == goal:
            if len(path) > 0:
                return path[0]
            else:
                return snake_direction

        closed_set.add(current)

        neighbors = [
            (current[0] + UNIT_SIZE, current[1]),
            (current[0] - UNIT_SIZE, current[1]),
            (current[0], current[1] + UNIT_SIZE),
            (current[0], current[1] - UNIT_SIZE),
        ]

        for neighbor in neighbors:
            if (
                0 <= neighbor[0] < GRID_WIDTH
                and 0 <= neighbor[1] < GRID_HEIGHT
                and neighbor not in snake
                and neighbor not in closed_set
            ):
                neighbor_heuristic = heuristic(neighbor, goal)
                heapq.heappush(open_set, (neighbor_heuristic, neighbor, path + [neighbor]))

    return snake_direction

# Font for the score display
font = pygame.font.Font(None, 36)

# Initialize new_head with a dummy value before the game loop
new_head = (snake[0][0] + UNIT_SIZE, snake[0][1])

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # Calculate the direction to move towards the food
    snake_direction = a_star_search(snake, food)

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

    # ... (rest of your existing code)

    # Check for a collision with the border
    if (
        snake[0][0] < 0
        or snake[0][0] >= GRID_WIDTH
        or snake[0][1] < 0
        or snake[0][1] >= GRID_HEIGHT
    ):
        # Reset the game
        time.sleep(1)
        snake = [(UNIT_SIZE * 7, UNIT_SIZE * 7)]
        snake_direction = "right"
        score = 0

    # Check for head collision with the body
    if snake[0] in snake[1:]:
        # Reset the game
        time.sleep(1)
        snake = [(UNIT_SIZE * 7, UNIT_SIZE * 7)]
        snake_direction = "right"
        score = 0

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
    score_text = font.render(f"Score: {score} High Score: {high_score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    pygame.display.update()

    # Delay for smooth movement
    time.sleep(delay)
