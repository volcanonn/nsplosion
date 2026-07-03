import math

import taichi as ti

# 1. Initialize for AMD/Intel/Nvidia GPUs using the Vulkan backend
ti.init(arch=ti.vulkan)

# --- Physics & Simulation Constants (Natural Units) ---
# Distance: 1.0 = 1 AU
# Mass: 1.0 = 1 Solar Mass
# Time: 1.0 = 1 Earth Year
G = 4.0 * math.pi**2  # Gravitational constant in AU^3 / (M_sun * Year^2)
DT = 0.02  # 0.02 Years per frame (~7.3 days). Fast but stable.
THETA = 0.7  # Barnes-Hut opening angle
EPS = 1.0  # Softening factor in AU (1 AU prevents all NaN explosions)

BODIES = 1000  # Number of particles
MAX_NODES = BODIES * 8
MAX_DEPTH = 20

# Internal render buffer
RENDER_W, RENDER_H = 1280, 720

# Map physical bounds ([-60, 60] AU)
MAP_SIZE = 120.0
HALF_MAP = MAP_SIZE / 2.0

# 2. Define the structural node layout
BHNode = ti.types.struct(
    mass=ti.f32,
    center_of_mass=ti.math.vec2,
    children=ti.types.vector(4, ti.i32),
    is_leaf=ti.i32,
    particle_idx=ti.i32,
    center=ti.math.vec2,
    half_size=ti.f32,
)

# 3. Flat Memory Allocations
node_pool = BHNode.field(shape=MAX_NODES)
node_counter = ti.field(ti.i32, shape=())

particle_pos = ti.Vector.field(2, dtype=ti.f32, shape=BODIES)
particle_vel = ti.Vector.field(2, dtype=ti.f32, shape=BODIES)
particle_mass = ti.field(dtype=ti.f32, shape=BODIES)


@ti.kernel
def init_bodies():
    # Create a physically accurate spinning galaxy disk
    M_TOTAL = 10000.0  # Total mass of the system in Solar Masses
    R_MAX = 50.0  # Outer radius of the disk in AU

    for i in range(BODIES):
        # Distribute particles radially using sqrt for uniform area density
        r = ti.sqrt(ti.random()) * R_MAX + 1.0
        theta = ti.random() * 2.0 * math.pi

        pos = ti.Vector([r * ti.cos(theta), r * ti.sin(theta)])
        particle_pos[i] = pos

        # Calculate stable orbital velocity for a uniform density disk
        # v = sqrt(G * M_enclosed / r), where M_enclosed ~ M_total * (r / R_max)^2
        m_enc = M_TOTAL * (r / R_MAX) * (r / R_MAX)
        v_mag = ti.sqrt(G * m_enc / r)

        # Tangential velocity
        particle_vel[i] = ti.Vector([-v_mag * ti.sin(theta), v_mag * ti.cos(theta)])
        particle_mass[i] = ti.random() * 0.5 + 0.5  # 0.5 to 1.0 Solar Masses


@ti.kernel
def reset_tree():
    node_counter[None] = 1
    node_pool[0].mass = 0.0
    node_pool[0].center_of_mass = ti.Vector([0.0, 0.0])
    node_pool[0].children = ti.Vector([-1, -1, -1, -1])
    node_pool[0].is_leaf = 1
    node_pool[0].particle_idx = -1
    node_pool[0].center = ti.Vector([0.0, 0.0])
    node_pool[0].half_size = HALF_MAP


@ti.func
def get_quadrant(pos, center):
    q = 0
    if pos[0] > center[0]:
        q += 1
    if pos[1] > center[1]:
        q += 2
    return q


@ti.func
def allocate_quad_block(parent_center, parent_half_size) -> ti.i32:
    first_child_idx = ti.atomic_add(node_counter[None], 4)
    ret_val = first_child_idx

    if first_child_idx + 4 >= MAX_NODES:
        ret_val = -1
    else:
        new_half_size = parent_half_size * 0.5

        for i in ti.static(range(4)):
            idx = first_child_idx + i
            node_pool[idx].mass = 0.0
            node_pool[idx].center_of_mass = ti.Vector([0.0, 0.0])
            node_pool[idx].children = ti.Vector([-1, -1, -1, -1])
            node_pool[idx].is_leaf = 1
            node_pool[idx].particle_idx = -1

            dx = new_half_size if (i % 2 == 1) else -new_half_size
            dy = new_half_size if (i >= 2) else -new_half_size

            node_pool[idx].center = parent_center + ti.Vector([dx, dy])
            node_pool[idx].half_size = new_half_size

    return ret_val


@ti.func
def insert_particle(p_idx):
    pos = particle_pos[p_idx]
    mass_p = particle_mass[p_idx]

    curr_idx = 0
    depth = 0
    done = 0

    while done == 0:
        node_pool[curr_idx].mass += mass_p
        node_pool[curr_idx].center_of_mass += pos * mass_p

        node = node_pool[curr_idx]

        if node.is_leaf == 1:
            if node.particle_idx == -1:
                node_pool[curr_idx].particle_idx = p_idx
                done = 1
            else:
                if depth >= MAX_DEPTH:
                    done = 1
                else:
                    existing_p_idx = node.particle_idx
                    node_pool[curr_idx].particle_idx = -1
                    node_pool[curr_idx].is_leaf = 0

                    first_child = allocate_quad_block(node.center, node.half_size)
                    if first_child == -1:
                        done = 1
                    else:
                        for i in ti.static(range(4)):
                            node_pool[curr_idx].children[i] = first_child + i

                        e_pos = particle_pos[existing_p_idx]
                        e_mass = particle_mass[existing_p_idx]
                        child_q = get_quadrant(e_pos, node.center)
                        child_idx = first_child + child_q

                        node_pool[child_idx].particle_idx = existing_p_idx
                        node_pool[child_idx].mass = e_mass
                        node_pool[child_idx].center_of_mass = e_pos * e_mass

                        new_q = get_quadrant(pos, node.center)
                        curr_idx = first_child + new_q
                        depth += 1
        else:
            child_q = get_quadrant(pos, node.center)
            curr_idx = node.children[child_q]
            depth += 1


@ti.kernel
def build_tree():
    ti.loop_config(serialize=True)
    for i in range(BODIES):
        insert_particle(i)


@ti.func
def calculate_bh_force(p_idx: ti.i32) -> ti.math.vec2:
    pos = particle_pos[p_idx]
    force = ti.Vector([0.0, 0.0])

    stack = ti.Vector([0] * 64, ti.i32)
    stack_ptr = 0

    stack[stack_ptr] = 0
    stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1
        curr_node_idx = stack[stack_ptr]

        skip = 0
        if curr_node_idx == -1:
            skip = 1

        if skip == 0:
            node = node_pool[curr_node_idx]
            if node.mass == 0.0:
                skip = 1

        if skip == 0:
            node = node_pool[curr_node_idx]
            com = node.center_of_mass / node.mass
            diff = com - pos
            dist_sq = diff.norm_sqr() + EPS
            dist = ti.sqrt(dist_sq)

            node_size = node.half_size * 2.0

            if node.is_leaf == 1 or (node_size / dist) < THETA:
                if node.particle_idx != p_idx:
                    inv_dist3 = 1.0 / (dist_sq * dist)
                    force += G * node.mass * diff * inv_dist3
            else:
                for i in ti.static(range(4)):
                    child_idx = node.children[i]
                    if child_idx != -1:
                        stack[stack_ptr] = child_idx
                        stack_ptr += 1

    return force


@ti.kernel
def integrate_and_render(pixels: ti.template()):
    white = ti.cast(ti.Vector([1.0, 1.0, 1.0]), ti.f32)
    w = pixels.shape[0]
    h = pixels.shape[1]

    for i in range(BODIES):
        f = calculate_bh_force(i)
        acc = f / particle_mass[i]

        particle_vel[i] += acc * DT
        particle_pos[i] += particle_vel[i] * DT

        # Hard boundaries at edge of map so tree logic doesn't break
        for d in ti.static(range(2)):
            if particle_pos[i][d] < -HALF_MAP:
                particle_pos[i][d] = -HALF_MAP
                particle_vel[i][d] *= -0.8
            if particle_pos[i][d] > HALF_MAP:
                particle_pos[i][d] = HALF_MAP
                particle_vel[i][d] *= -0.8

        # Map physics space [-60, 60] to pixel space [0, w/h]
        # y is inverted so it doesn't look upside down
        px = ti.cast((particle_pos[i][0] / MAP_SIZE + 0.5) * w, ti.i32)
        py = ti.cast((0.5 - particle_pos[i][1] / MAP_SIZE) * h, ti.i32)

        # Draw a 2x2 block for better visibility
        for dx in ti.static(range(2)):
            for dy in ti.static(range(2)):
                if 0 <= px + dx < w and 0 <= py + dy < h:
                    pixels[px + dx, py + dy] = white


# --- Main Window Loop ---
# Taichi windows are implicitly resizable by Hyprland/Wayland.
window = ti.ui.Window(
    "Taichi Barnes-Hut N-Body (Real Units)", (RENDER_W, RENDER_H), vsync=True
)
canvas = window.get_canvas()
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(RENDER_W, RENDER_H))

init_bodies()

while window.running:
    pixels.fill(ti.Vector([0.0, 0.0, 0.0]))

    reset_tree()
    build_tree()
    integrate_and_render(pixels)

    canvas.set_image(pixels)
    window.show()
