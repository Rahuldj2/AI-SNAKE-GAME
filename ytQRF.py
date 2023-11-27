import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
# font = pygame.font.Font('arial.ttf', 25)
font = pygame.font.Font(None, 24)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
#This object will keep track of coordinates of the snake and the food particle
Coordinate_obj = namedtuple('Coordinate_obj', 'x, y')

GRID_SIZE = 15
UNIT_SQUARE_SIZE = 30  # Increased block size for better visibility
GAME_FRAME_RATE = 20

high_score = 0
game_num = 1
avg_score = 0
sum_score = 0

class SnakeReinforce:
    #cool working
    def __init__(self, w=GRID_SIZE * UNIT_SQUARE_SIZE, h=GRID_SIZE * UNIT_SQUARE_SIZE):
        pygame.display.set_caption('Snake Game Q learning')
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.game_num=1
        self.avg_score=0
        self.sum_score=0
        self.high_score=0
        self.sum_score=0
        self.score=0
        self.reset()
    
    #this function resets the game in case the snake collides with corner or with itself
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Coordinate_obj(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Coordinate_obj(self.head.x - UNIT_SQUARE_SIZE, self.head.y),
            Coordinate_obj(self.head.x - (2 * UNIT_SQUARE_SIZE), self.head.y)
        ]
        self.sum_score += self.score
        self.score = 0
        self.food = None
        self._randomly_put_food()
        self.LoopCheck = 0
        self.avg_score = self.sum_score/self.game_num
        self.game_num += 1

    #this function places the food in a random location
    def _randomly_put_food(self):
        x = random.randint(0, (self.w - UNIT_SQUARE_SIZE) // UNIT_SQUARE_SIZE) * UNIT_SQUARE_SIZE
        y = random.randint(0, (self.h - UNIT_SQUARE_SIZE) // UNIT_SQUARE_SIZE) * UNIT_SQUARE_SIZE
        self.food = Coordinate_obj(x, y)
        if self.food in self.snake:
            self._randomly_put_food()
        
    def snake_play(self, action):
        self.LoopCheck += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.get_collision_bool() or self.LoopCheck > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        #means the snake has eaten food
        if self.head == self.food:
            self.score += 1
            if self.score > self.high_score:
                self.high_score = self.score
            reward = 10
            self._randomly_put_food()
        else:
            self.snake.pop()

        self._refresh_user_interface()
        self.clock.tick(GAME_FRAME_RATE)
        return reward, game_over, self.score
    
    def get_collision_bool(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
    
    def _refresh_user_interface(self):
        self.display.fill((0, 0, 0))
        
        for coordinate_point in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(coordinate_point.x, coordinate_point.y, UNIT_SQUARE_SIZE, UNIT_SQUARE_SIZE))
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(coordinate_point.x + 4, coordinate_point.y + 4, 12, 12))
        
        pygame.draw.rect(self.display, (200,0,0), pygame.Rect(self.food.x, self.food.y, UNIT_SQUARE_SIZE, UNIT_SQUARE_SIZE))
        
        # Draw grid lines
        for i in range(0, self.w, UNIT_SQUARE_SIZE):
            pygame.draw.line(self.display, (64, 64, 64), (i, 0), (i, self.h))
        for j in range(0, self.h, UNIT_SQUARE_SIZE):
            pygame.draw.line(self.display, (64, 64, 64), (0, j), (self.w, j))

        # text = font.render("Score: " + str(self.score), True, WHITE)
        text=font.render(f"Game Num: {self.game_num}  Score: {self.score}  High Score: {self.high_score}  Avg Score: {self.avg_score}", True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    
    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Right turn R -> D -> L -> U
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Left turn R -> U -> L -> D
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += UNIT_SQUARE_SIZE
        elif self.direction == Direction.LEFT:
            x -= UNIT_SQUARE_SIZE
        elif self.direction == Direction.DOWN:
            y += UNIT_SQUARE_SIZE
        elif self.direction == Direction.UP:
            y -= UNIT_SQUARE_SIZE

        # Align the head position to the grid
        x = (x // UNIT_SQUARE_SIZE) * UNIT_SQUARE_SIZE
        y = (y // UNIT_SQUARE_SIZE) * UNIT_SQUARE_SIZE
            
        self.head = Coordinate_obj(x, y)

# Example of using the modified SnakeReinforce class