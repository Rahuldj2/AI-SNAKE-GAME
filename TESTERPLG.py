import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(None, 36)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRID_SIZE = 15
GRID_WIDTH = 450
GRID_HEIGHT = 450
UNIT_SIZE = GRID_WIDTH // GRID_SIZE
BLOCK_SIZE = 30
SPEED = 15

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - UNIT_SIZE, self.head.y),
                      Point(self.head.x - (2 * UNIT_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - UNIT_SIZE) // UNIT_SIZE) * UNIT_SIZE
        y = random.randint(0, (self.h - UNIT_SIZE) // UNIT_SIZE) * UNIT_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - UNIT_SIZE or pt.x < 0 or pt.y > self.h - UNIT_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill((0, 0, 0))

        # Draw the grid
        for x in range(0, self.w, UNIT_SIZE):
            pygame.draw.line(self.display, WHITE, (x, 0), (x, self.h))
        for y in range(0, self.h, UNIT_SIZE):
            pygame.draw.line(self.display, WHITE, (0, y), (self.w, y))

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, UNIT_SIZE, UNIT_SIZE))
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x + 4, pt.y + 4, UNIT_SIZE - 8, UNIT_SIZE - 8))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, UNIT_SIZE, UNIT_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [10, 10])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += UNIT_SIZE
        elif self.direction == Direction.LEFT:
            x -= UNIT_SIZE
        elif self.direction == Direction.DOWN:
            y += UNIT_SIZE
        elif self.direction == Direction.UP:
            y -= UNIT_SIZE

        self.head = Point(x, y)

# Main game loop
if __name__ == "__main__":
    game = SnakeGameAI(GRID_WIDTH, GRID_HEIGHT)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = [1, 0, 0, 0]
        elif keys[pygame.K_RIGHT]:
            action = [0, 1, 0, 0]
        elif keys[pygame.K_UP]:
            action = [0, 0, 1, 0]
        elif keys[pygame.K_DOWN]:
            action = [0, 0, 0, 1]
        else:
            action = [0, 0, 0, 0]

        reward, game_over, score = game.play_step(action)

        if game_over:
            game.reset()
