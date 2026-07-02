import taichi as ti

G = 0.0001

BODIES = 100
WIDTH, HEIGHT = 600, 600
FPS = 60


ti.init(arch=ti.gpu)

pos = ti.Vector.field(2, dtype=ti.f32)
vel = ti.Vector.field(2, dtype=ti.f32)
forces = ti.Vector.field(2, dtype=ti.f32)
mass = ti.field(dtype=ti.f32)

ti.root.dense(ti.i, BODIES).place(pos, vel, forces, mass)

dt = 1 / FPS


@ti.kernel
def init_bodies():
    for i in range(BODIES):
        pos[i] = ti.Vector([ti.random() * 0.5 + 0.25, ti.random() * 0.5 + 0.25])

        vel[i] = ti.Vector([0.0, 0.0])
        forces[i] = ti.Vector([0.0, 0.0])
        mass[i] = ti.random() * 0.5 + 0.5


NUM_PAIRS = (BODIES * (BODIES - 1)) // 2


pair_i = ti.field(ti.i32, shape=NUM_PAIRS)
pair_j = ti.field(ti.i32, shape=NUM_PAIRS)


@ti.kernel
def init_pairs():
    for i in range(BODIES):
        k_start = i * BODIES - i * (i + 1) // 2
        for j in range(i + 1, BODIES):
            idx = k_start + (j - i - 1)
            pair_i[idx] = i
            pair_j[idx] = j


@ti.kernel
def compute_forces():
    for idx in range(NUM_PAIRS):
        i = pair_i[idx]
        j = pair_j[idx]
        d = pos[j] - pos[i]
        dist_sq = d.dot(d)
        softening = 0.01
        inv_dist = 1.0 / ti.sqrt(dist_sq + softening**2)
        inv_dist3 = inv_dist * inv_dist * inv_dist
        f = G * mass[i] * mass[j] * d * inv_dist3
        ti.atomic_add(forces[i], f)
        ti.atomic_sub(forces[j], f)


@ti.kernel
def gravity(pixels: ti.template()):
    white = ti.cast(ti.Vector([1.0, 1.0, 1.0]), ti.f32)
    dims = ti.Vector([pixels.shape[0] - 1, pixels.shape[1] - 1])

    for i in forces:
        f = forces[i]
        m = mass[i]

        acc = f / m
        vel[i] += acc * dt
        pos[i] += vel[i] * dt

        forces[i] = ti.Vector([0.0, 0.0])

        for d in ti.static(range(2)):
            if pos[i][d] < 0.01:
                pos[i][d] = 0.01
                vel[i][d] *= -0.8

            if pos[i][d] > 0.99:
                pos[i][d] = 0.99
                vel[i][d] *= -0.8

        pixel_positions = ti.cast(pos[i] * dims, ti.i32)
        pixels[pixel_positions[0], pixel_positions[1]] = white


# 6. Create the modern GGUI Window and Canvas
window = ti.ui.Window("Taichi 2D Simulation", (1280, 720), vsync=True)
canvas = window.get_canvas()
gui = window.get_gui()

pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

init_bodies()
init_pairs()
while window.running:
    pixels.fill(ti.Vector([0.0, 0.0, 0.0]))
    compute_forces()
    gravity(pixels)
    canvas.set_image(pixels)
    window.show()
