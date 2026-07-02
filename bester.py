import taichi as ti

x = ti.field(ti.f32)
block = ti.root.pointer(ti.ij, (4, 4))
pixel = block.bitmasked(ti.ij, (2, 2))
pixel.place(x)
