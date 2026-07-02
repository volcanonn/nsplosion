import numpy as np
import pygame

G = 6.674 * 10**-11

G *= 10**5


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
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.RESIZABLE)
# Use actual surface size, not requested size
WIDTH, HEIGHT = screen.get_size()
pixel_data = np.zeros((WIDTH, HEIGHT), dtype=np.uint32)

white = 0xFFFFFFFF

dt = 1 / FPS
running = True
clock = pygame.time.Clock()


def clamp(value, least, most):
    return max(least, min(value, most))


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.VIDEORESIZE:
            # Just update dimensions - DON'T call set_mode()
            WIDTH, HEIGHT = event.size

    # Always sync to actual surface size
    actual_w, actual_h = screen.get_size()
    if actual_w != WIDTH or actual_h != HEIGHT:
        WIDTH, HEIGHT = actual_w, actual_h

    # Reallocate if size changed
    if pixel_data.shape != (WIDTH, HEIGHT):
        pixel_data = np.zeros((WIDTH, HEIGHT), dtype=np.uint32)
    pixel_data.fill(0)
    # F=Gm1m2/r^2
    # np.random.randint(0, 256, size=(WIDTH, HEIGHT), dtype=np.uint32)

    forces = np.zeros((BODIES, 2), dtype=np.float64)

    for i, j in list(zip(*np.triu_indices(BODIES, 1))):
        pos1 = positions[i]
        pos2 = positions[j]
        body1 = dots[i]
        body2 = dots[j]

        displacement = pos1 - pos2
        distance = np.linalg.norm(displacement)

        if distance < 0.01:
            continue

        force = (G * body1.mass * body2.mass * displacement) / (distance**3)

        forces[i] -= force
        forces[j] += force

    for force, pos, body in zip(forces, positions, dots):
        acceleration = force / body.mass
        body.vel += acceleration * dt
        pos += body.vel * dt

        if 0 > pos[0] or pos[0] > 1:
            pos[0] = clamp(pos[0], 0.0, 1.0)
            body.vel[0] *= -0.5

        if 0 > pos[1] or pos[1] > 1:
            pos[1] = clamp(pos[1], 0.0, 1.0)
            body.vel[1] *= -0.5

    pixel_positions = (positions * np.array([WIDTH - 1, HEIGHT - 1])).astype(int)
    pixel_data[pixel_positions[:, 0], pixel_positions[:, 1]] = white

    pygame.surfarray.blit_array(screen, pixel_data)

    pygame.display.flip()

    clock.tick(FPS)

# Clean up and exit cleanly
pygame.quit()
