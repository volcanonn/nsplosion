import math

import numpy as np
import pygame

G = 6.674 * 10**-11

G *= 10**2


class Dot:
    def __init__(self, mass: float):
        self.vel = np.array([0.0, 0.0])
        self.mass = mass


BODIES = 40
WIDTH, HEIGHT = 600, 600
FPS = 60

rng = np.random.default_rng()
positions = rng.uniform(0, 1, size=(BODIES, 2))
vectorized_dot = np.vectorize(Dot)
dots = vectorized_dot(rng.uniform(0.5, 0.7, size=BODIES)).tolist()

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
pixel_data = np.zeros((WIDTH, HEIGHT), dtype=np.uint32)

dt = 1 / FPS
white = 0xFFFFFFFF

running = True
clock = pygame.time.Clock()


def clamp(value, least, most):
    return max(least, min(value, most))


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pixel_data.fill(0)
    # F=Gm1m2/r^2
    # np.random.randint(0, 256, size=(WIDTH, HEIGHT), dtype=np.uint32)

    for pos1, body1 in zip(positions, dots):
        force = np.array([0.0, 0.0])
        for pos2, body2 in zip(positions, dots):
            if body1 == body2:
                continue

            displacement = pos1 - pos2
            distance = np.linalg.norm(displacement)

            if distance < 0.05:
                continue

            force -= (G * body1.mass * body2.mass * displacement) / (distance**3)

        acceleration = force / body1.mass
        body1.vel += acceleration * dt
        pos1 += body1.vel * dt

        if 0 > pos1[0] or pos1[0] > 1:
            pos1[0] = clamp(pos1[0], 0.0, 1.0)
            body1.vel[0] *= -0.5

        if 0 > pos1[1] or pos1[1] > 1:
            pos1[1] = clamp(pos1[1], 0.0, 1.0)
            body1.vel[1] *= -0.5

    pixel_positions = (positions * np.array([WIDTH - 1, HEIGHT - 1])).astype(int)
    pixel_data[pixel_positions[:, 0], pixel_positions[:, 1]] = white

    pygame.surfarray.blit_array(screen, pixel_data)

    pygame.display.flip()

    clock.tick(FPS)

# Clean up and exit cleanly
pygame.quit()
