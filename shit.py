import math
import random

import numpy as np
import pygame

# I know this code looks like the unperformant thing you've ever seen
# I kinda want to see the difference between the worst code vs somewhat good code


class Position:
    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        if not isinstance(value, tuple):
            raise TypeError("Position must be a float.")
        self._pos = value

    @property
    def x(self):
        return self._pos[0]

    @x.setter
    def x(self, value):
        if not isinstance(value, float):
            raise TypeError("Position must be a float.")
        self._pos = (value, self._pos[1])

    @property
    def y(self):
        return self._pos[1]

    @y.setter
    def y(self, value):
        if not isinstance(value, float):
            raise TypeError("Position must be a float.")
        self._pos = (self._pos[0], value)


class Dot(Position):
    def __init__(self, pos: tuple[float, float], mass: float):
        self.pos = pos
        self.vel = (0, 0)
        self.mass = mass


# 1. Initialize Pygame modules
pygame.init()

size = (600, 600)

# 2. Create the window display
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Objects")

# 3. Create a clock object to track and limit frame rate
clock = pygame.time.Clock()

# Boolean flag to keep the loop running
running = True

objects = []

G = 6.674 * 10**-11

G *= 10**2  # made gravity stronger awards


def clamp(value, least, most):
    return max(least, min(value, most))


for _ in range(40):
    objects.append(Dot((random.random(), random.random()), random.random()))

# --- THE MAIN GAME LOOP ---
while running:
    # PHASE A: Process Input (Event Handling)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Clicking the 'X' button
            running = False

    # PHASE B: Update Game Logic
    # (e.g., move characters, check for collisions, score points)
    # F=Gm1m2/r^2
    for object in objects:  # shouldve used a quadtree award
        forcevector = [0, 0]  # shouldve turned this into a np object award
        for object2 in objects:  # O(n^2) and iterate over our own object award. Couldve been turned into O(n^2-n) award
            dist = math.sqrt(
                (object.x - object2.x) ** 2 + (object.y - object2.y) ** 2
            )  # shouldve used the vectorized formula for gravity award
            if dist < 0:
                # Could have them combine but it would fuck with the iterator
                continue  # crashes when objects are inside eachother award
            force = (
                G * object.mass * object2.mass / (dist**2)
            )  # literaly sqrt then square award
            angle = math.atan2(
                object2.y - object.y, object2.x - object.x
            )  # only code that requires an iq above 70 award
            forcevector = [
                forcevector[0] + force * math.cos(angle),
                forcevector[1] + force * math.sin(angle),
            ]
        object.vel = (
            object.vel[0] + forcevector[0] * object.mass,
            object.vel[1] + forcevector[1] * object.mass,
        )
        object.pos = (
            object.x + object.vel[0],
            object.y + object.vel[1],
        )
        if object.x < 0 or object.x > 1:
            object.x = clamp(object.x, float(0), float(1))
            object.vel = (0, object.vel[1])

        if object.y < 0 or object.y > 1:
            object.y = clamp(object.y, float(0), float(1))
            object.vel = (
                object.vel[0],
                0,
            )  # literaly just shit code and unreadable and unperformant jesus shut the site down

    # The reason ai cutoff date is 2025
    arr = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for object in objects:
        arr[round(object.x * (size[0] - 1)), round(object.y * (size[1] - 1))] = (
            255,
            255,
            255,
        )

    pygame.surfarray.blit_array(screen, arr)

    # (e.g., draw your player or shapes here)

    pygame.display.flip()  # Update the full display Surface to the screen

    # Maintain constant speed (60 Frames Per Second)
    clock.tick(60)

# Clean up and exit cleanly
pygame.quit()
