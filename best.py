import taichi as ti

G = 6.674 * 10**-11

G *= 10**5

ti.init(arch=ti.gpu)

NUM_PARTICLES = 10000
DT = 1e-3
GRAVITY = ti.Vector([0.0, -9.8])

pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
vel = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
mass = ti.Vector.field(1, dtype=ti.f32, shape=NUM_PARTICLES)


@ti.kernel
def init_particles():
    for i in pos:
        pos[i] = ti.Vector([ti.random() * 0.8 + 0.1, ti.random() * 0.8 + 0.1])
        vel[i] = ti.Vector([(ti.random() - 0.5) * 2.0, (ti.random() - 0.5) * 2.0])


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


# Run initial setup
init_particles()

# 6. Create the modern GGUI Window and Canvas
window = ti.ui.Window("Taichi 2D Simulation", res=(600, 600))
canvas = window.get_canvas()

# Main simulation and rendering loop
while window.running:
    # Run multiple physics steps per render frame for numerical stability
    for _ in range(10):
        substep()

    # Clear screen to a dark grey background
    canvas.set_background_color((0.1, 0.1, 0.1))

    # Render all 10,000 particles instantly using the canvas circles API
    canvas.circles(pos, radius=0.003, color=(0.2, 0.7, 1.0))

    # Draw frame to the screen
    window.show()
