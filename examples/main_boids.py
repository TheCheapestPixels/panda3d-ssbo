# Welcome to the Boids example, the first application around here that
# does something *cool*.
#
# First let's try dumping data into pstats, because "How fast is it?" is
# quite an important questions with systems like this.
from panda3d.core import PStatClient
PStatClient.connect()
PStatClient.mainTick()


from panda3d.core import Vec3

from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import DirectSlider

from p3d_ssbo.gltypes import GlVec3
from p3d_ssbo.gltypes import GlUInt
from p3d_ssbo.gltypes import Struct
from p3d_ssbo.gltypes import Buffer
from p3d_ssbo.algos.raw_glsl import RawGLSL
from p3d_ssbo.algos.copy import Copy
from p3d_ssbo.algos.random_number_generator import MurmurHash
from p3d_ssbo.algos.spatial_hash import SpatialHash
from p3d_ssbo.algos.spatial_hash import PivotTable
from p3d_ssbo.algos.spatial_hash import PairwiseAction
from p3d_ssbo.algos.bitonic_sort import BitonicSort
from p3d_ssbo.tools.ssbo_particles import SSBOParticles


# Just a quick no-frills setup of the engine...
ShowBase()
base.accept('escape', base.task_mgr.stop)
base.set_frame_rate_meter(True)


# The camera shall rotate around the cube...
gimbal = base.render.attach_new_node("gimbal")
gimbal.set_pos(0.5, 0.5, 0.5)
base.cam.reparent_to(gimbal)
base.cam.set_y(-2.5)
def update_camera(task):
    gimbal.set_h(task.time)  # * 20.0)
    return task.cont
base.task_mgr.add(update_camera)


# FIXME: Square this away somewhere better.
from panda3d.core import CullBinManager
bin_mgr = CullBinManager.get_global_ptr()


# Okay, here the actually interesting things begin. So, what *is* this
# program?
# We're implementing boids, an algorithm published by Craig Reynolds in
# 1986, which was (and is) an important starting point to simulating the
# behavior of flocks of birds, schools of fish, and groups of people. It
# does so by calculating for each boid (a data-laden point in space)
# which other boids are in its vicinity, and whether they are visible to
# it; If they are, the other boid's influence on this one are calculated
# so that this boid maintains or improves:
# * Cohesion: Try to fly close to the other boid,
# * Separation: ...but not too close,
# * Alignment: ...and try to fly in the same direction.
#
# As the complexity is, in a naive implementation, O(n**2) (as each boid
# has to consider each other boid)
num_elements = 2**12  # Usually has to be a multiple of 32, but because of bitonic sort, it has to be a power of 2 equal or greater than 64.
grid_res = (16, 16, 16)  # Per-axis number of cells in the spatial hash grid. Product has to be a multiple of 32.
grid_vol = (1.0, 1.0, 1.0)  # Spatial volume that is covered by the spatial hash grid.

boids = Struct(
    'Boid',
    GlVec3('pos'),
    GlVec3('nextPos'),
    #GlVec3('dir'),
    GlUInt('hashIdx'),
)
pivot = Struct(
    'Pivot',
    GlUInt('start'),
    GlUInt('len'),
)
data_buffer = Buffer(
    'dataBuffer',
    boids('boids', num_elements),
    pivot('pivot', grid_res[0] * grid_res[1] * grid_res[2]),
    #boids('boids', num_elements, unbounded=True),
)


# The compute shaders
rng = MurmurHash(
    data_buffer,
    ('boids', 'pos'),
    #('boids', 'dir'),
)
spatial_hash = SpatialHash(
    data_buffer,
    ('boids', 'pos', 'hashIdx'),  # array, position field, hash field
    grid_vol,  # Hash grid volume
    grid_res,  # Hash grid resolution
)
sorter = BitonicSort(
    data_buffer,
    ('boids', 'hashIdx'),
)
pivot = PivotTable(
    data_buffer,
    ('boids', 'hashIdx'),  # What to build the pivot table for.
    ('pivot', 'start', 'len'),  # The pivot table, and where it stores start and length.
)
declarations = """
uniform float radius;
float dt = min(osg_DeltaFrameTime, 1./30.); // n.b. application must run at 30fps minimum!

// Value accumulators for the boid.
uint otherVecs = 0;
vec3 cohesion = vec3(0);
vec3 separation = vec3(0);

float sepRadius = 0.05;
vec3 minSpeed = vec3(0.);  // n.b.: Speed times dt usually; omitted here because of 0 value
vec3 maxSpeed = vec3(2.5 * dt);
vec3 vel = vec3(0.05,0.,0.) * dt; // prevailing motion
"""[1:-1]
processing = """
  // `a` is the current boid, `b` the nearby boid.
  vec3 toBoid = b.pos - a.pos;
  float dist = length(toBoid);
  if (dist <= radius) {
    otherVecs++;

    // Cohesion
    // cohesion += toBoid;
    cohesion += normalize(toBoid) * pow(20./dist, 6.);

    // Separation
    separation -= pow(15./dist, 3.);
    // separation += normalize(-toBoid) * max(0, sepRadius - length(toBoid));

    // Alignment: FIXME
  }
"""[1:-1]
combining = """
  // After looping over all nearby boids.
  vec3 pos = boids[boidIdx].pos;
  if (otherVecs > 0) {
    // vec3 move = ((cohesion + 3.0 * separation) / 4.0) / otherVecs;
    vec3 move = (cohesion + separation);
    move = clampVec(move, minSpeed, maxSpeed);
    vel += move;
    boids[boidIdx].nextPos = min(max(pos + vel * dt, vec3(0)), vec3(1));
  } else {
    // boids[boidIdx].nextPos = pos + vel * dt;
    boids[boidIdx].nextPos = mod(pos +  vel * dt, vec3(1));
  }
"""[1:-1]
mover = PairwiseAction(
    data_buffer,
    'boids',
    'pivot',
    declarations,
    processing,
    combining,
    src_args=dict(
        gridRes=grid_res,
        gridVol=grid_vol,
    ),
    shader_args=dict(radius=0.15),
    debug=True,
)
movement_actualizer = Copy(
    data_buffer,
    (('boids', 'nextPos'), ('boids', 'pos')),
)

# UI
def set_radius(*args, **kwargs):
    r = slider_radius['value']
    mover.set_shader_arg('radius', r)
    print(f"Detection radius: {r}")
slider_radius = DirectSlider(
    parent=base.a2dTopLeft,
    frameSize=(0, 1, -0.03, 0.03),
    pos=(0.02+0.45, 0, -0.05),
    text="Detection radius",
    text_scale=0.05,
    text_pos=(-0.25, -0.015),
    range=(0.0, 1.0),
    value=mover.shader_args['radius'],
    pageSize=0.01,
    command=set_radius,
)


points = SSBOParticles(base.render, data_buffer, ('boids', 'pos'))


def setup_debug():
    for shader in [rng, spatial_hash, sorter, pivot, mover]:
        shader.dispatch()
        data = base.win.gsg.get_engine().extract_shader_buffer_data(
            data_buffer.ssbo,
            base.win.gsg,
        )
def setup_prod():
    rng.dispatch()
    stages = [
        ("cmp_spatial_hash", spatial_hash),
        ("cmp_sort_spatial_hashes", sorter),
        ("cmp_pivot_table", pivot),
        ("cmp_mover", mover),
        ("cmp_mover_2", movement_actualizer),
    ]
    for idx, (bin_name, shader) in enumerate(stages):
        bin_mgr.add_bin(bin_name, CullBinManager.BT_fixed, -20+idx)
        shader.attach(points.get_np(), bin_name)
        print(f"Shader {bin_name} attached.")
setup_prod()


#data = base.win.gsg.get_engine().extract_shader_buffer_data(
#    data_buffer.ssbo,
#    base.win.gsg,
#)
#py_data = data_buffer.unpack(data)
#for element in py_data[0]:
#    print(element[0])


base.run()
