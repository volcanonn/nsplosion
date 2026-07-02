import numpy as np
import taichi as ti

G = 6.674 * 10**-11

G *= 10**5

ti.init(arch=ti.gpu)

BODIES = 40
WIDTH, HEIGHT = 600, 600
FPS = 60

pos = ti.Vector.field(2, dtype=ti.f32, shape=BODIES)
vel = ti.Vector.field(2, dtype=ti.f32, shape=BODIES)
mass = ti.Vector.field(1, dtype=ti.f32, shape=BODIES)

dt = 1 / FPS


@ti.kernel
def init_particles():
    for i in pos:
        pos[i] = ti.Vector([ti.random() * 0.8 + 0.1, ti.random() * 0.8 + 0.1])
        vel[i] = ti.Vector([(ti.random() - 0.5) * 2.0, (ti.random() - 0.5) * 2.0])


GRAVITY = 1
DT = 1


# 5. Parallel simulation step running entirely on the GPU
@ti.kernel
def substep():
    for i in pos:
        # Apply physics formulas: v = v + a*dt, x = x + v*dt
        vel[i] += GRAVITY * DT
        pos[i] += vel[i] * DT

        # Simple 2D boundary collision checking
        for d in ti.static(range(2)):
            if pos[i][d] < 0.05:
                pos[i][d] = 0.05
                vel[i][d] *= -0.8  # Bounce with energy loss

            if pos[i][d] > 0.95:
                pos[i][d] = 0.95
                vel[i][d] *= -0.8


rng = np.random.default_rng()
positions = rng.uniform(0, 1, size=(BODIES, 2))
positions = rng.uniform(0, 1, size=(BODIES, 2))
vectorized_dot = np.vectorize(Dot)
dots = vectorized_dot(rng.uniform(0.5, 0.7, size=BODIES)).tolist()


@ti.kernel
def gravity(pixels: ti.template()):
    width = pixels.shape[0]
    height = pixels.shape[1]
    black = ti.cast(ti.Vector([0.0, 0.0, 0.0]), ti.f32)
    white = ti.cast(ti.Vector([1.0, 1.0, 1.0]), ti.f32)
    for i, j in pixels:
        pixels[i, j] = black

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

        pixel_positions = (positions * np.array([width - 1, height - 1])).astype(int)
        pixels[pixel_positions[:, 0], pixel_positions[:, 1]] = white


# 6. Create the modern GGUI Window and Canvas
window = ti.ui.Window("Taichi 2D Simulation", (1280, 720), vsync=True)
canvas = window.get_canvas()
gui = window.get_gui()

pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# Main simulation and rendering loop
while window.running:
    gravity(pixels)
    canvas.set_image(pixels)
    window.show()
