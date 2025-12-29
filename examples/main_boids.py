# Welcome to the Boids example, the first application around here that
# does something *cool*.
#
# First let's try dumping data into pstats, because "How fast is it?" is
# quite an important questions with systems like this.
from panda3d.core import PStatClient
PStatClient.connect()
PStatClient.mainTick()

from panda3d.core import CullBinManager

from direct.showbase.ShowBase import ShowBase

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
from p3d_ssbo.algos import boids as boids_module
from p3d_ssbo.tools.ssbo_particles import SSBOParticles


# Just a quick no-frills setup of the engine...
ShowBase()
base.accept('escape', base.task_mgr.stop)
base.set_frame_rate_meter(True)
base.set_background_color(0.6, 0.6, 0.9)


# The camera shall rotate around the cube...
gimbal = base.render.attach_new_node("gimbal")
gimbal.set_pos(0.5, 0.5, 0.5)
base.cam.reparent_to(gimbal)
base.cam.set_y(-2.0)


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
# has to consider each other boid), we need something to cut down on the
# number of boids that we look at for each of our boids. For this, we
# use a spatial hash grid, which will be explained below. For now: It's
# a cuboid grid of cells that covers a volume of space, and all our
# boids will be within it.


# So, how many boids do we want? So far, most shaders assume that we
# will use a multiple of 32, but the bitonic sort implementation
# requires a power of 2 that is at least 2**6 = 64.
num_elements = 2**14


# How large will the volume covered by the spatial hash grid be? Note
# that so far that means that it will stretch from the oriin to the
# provided coordinate, while being axis-aligned.
grid_vol = (1.0, 1.0, 1.0)
# And into how many cells do we split our grid? The product of the cells
# per axis also has to be a multiple of 32, and for some as of yet
# undetermined reason, 64**3 = 262144 seems to be the maximum. However,
# more cells doesn't automatically mean better performance; This is a
# parameter that has to be tuned.
grid_res = (16, 16, 16)
# How far will our boids be able to sense? The smaller, the better,
# though currently there are no further gains once the radius becomes
# smaller than the shortest edge of a cell
perception_radius = 0.02


# With some rather abstract configuration now covered, let's get coding
# for real, starting with the data structures that we need. A boid has a
# position where it is, and a direction (technically a velocity, as it
# has direction and speed). Due to the algorithms that we will use, it
# also has the index of the grid cell that it is in, and stores the
# position and direction that it will have; The latter prevents race
# conditions, as otherwise a boid would update itself before another one
# has calculated its interactions with it, leading to the next step's
# data being used instead of the current step's.
boids = Struct(
    'Boid',
    GlVec3('pos'),
    GlVec3('dir'),
    GlVec3('nextPos'),
    GlVec3('nextDir'),
    GlUInt('hashIdx'),
)
# A pivot table is basically a list of references that indicate where in
# the list of boids the contents of a grid cell begin, and how many
# boids there are in the cell.
pivot = Struct(
    'Pivot',
    GlUInt('start'),
    GlUInt('len'),
)
# These two taken together already form our boid system. This is also
# the point where we could feed some initial data. That would look
# somehat like this:
#
# ```python
# boid_0 = (
#     (0.1, 0.2, 0.3),  # position
#     (0.0, 0.1, 0.0),  # direction
#     (0.0, 0.0, 0.0),  # nextPos
#     (0.0, 0.0, 0.0),  # nextDir
#     0,                # hashIdx
# )
# boids_data = [boid_0, ...]
# pivot_table_data = [(start_0, len_0), (start_1, len_1), ...]
# buffer_data = (boids_data, pivot_table_data)
# ```
data_buffer = Buffer(
    'dataBuffer',
    boids('boids', num_elements),
    pivot('pivot', grid_res[0] * grid_res[1] * grid_res[2]),
    # This would feed in the initial data:
    #initial_data=buffer_data,
    # The following line is left here for hunting bugs.
    #boids('boids', num_elements, unbounded=True),
)


# We've got the data structure, now we need to fill it. For that, we
# just use random data within that is uniformly distributed within the
# given ranges. We will be dispatching (running) this shader only once,
# before running the actual boid simulation.
rng = MurmurHash(
    data_buffer,
    ('boids', 'pos', 0.2, 0.8),
    ('boids', 'dir', -0.2, 0.2),
)
# Every shader that now follows will be run once per frame while the
# grid's bounding box is visible. This first shader calculates which
# grid cell the boid is in, the numeric ID of which is the boid's
# spatial hash.
spatial_hash = SpatialHash(
    data_buffer,
    ('boids', 'pos', 'hashIdx'),  # array, position field, hash field
    grid_vol,  # Hash grid volume
    grid_res,  # Hash grid resolution
)
# Then we sort the array of boids by their spatial hashes.
sorter = BitonicSort(
    data_buffer,
    ('boids', 'hashIdx'),
)
# Now we create the pivot table data. The nth element of the pivot table
# stores where in the boid array the first boid with hash n is, and how
# many there are.
pivot = PivotTable(
    data_buffer,
    ('boids', 'hashIdx'),  # What array to build the pivot table for,
    # and which element to use as the key.
    ('pivot', 'start', 'len'),  # The pivot table, and where it stores
    # start and length.
)
# With all this data in place, we can now do the calculations, so here
# is a shader that we stick bits of GLSL into.
# * The first bit is the declaration of variables that we will need
#   during the calculation, and is accordingly called "declarations".
# * The second bit is the calculation itself, which sits in the center
#   of a loop, which provides us with "this" and "that" boid. This part
#   is called "processing".
# * The third goes at the end, after the loop, doing the final
#   calculations on the accumulated values, and also adding factors that
#   are based only at the boid itself. It is called "combining".
# The aforementioned loop calculates roughly which grid cells may be
# within the reach of the boid, look those up in the pivot table, then
# iterates over the boids of each of them, and finally passes them to
# the processing code.
#
# So, where *is* the actual GLSL code implementing boids? That part is
# left as an exercise to the reader.. Just kidding. As it turned out,
# boids start out pretty easy, and then you want to add *all* the
# features, so it becomes more sensible to spin that code off into its
# own module, and keep experimenting with it there. Furthermore, all the
# infrastructure that we are setting up here also allows for other kinds
# of particle systems; You could make a LaPlacian water simulation with
# it, or molecular dynamics, or even a gravity simulation (though for
# that, Barnes-Hut would be more appropriate than a spatial hash). So
# separating out the specific simulation from the simulation
# infrastructure seemed to make sense.
#
# So, do we send you hunting for the code in the `boids` module's
# source code? No, we'll use this opportunity to demonstrate a neat
# feature of these algorithms: Whenever you pass `debug=True` to them,
# they will output their generated code, and the boids code will show
# you where its three parts are.
mover = PairwiseAction(
    data_buffer,
    'boids',
    'pivot',
    boids_module.declarations,
    boids_module.processing,
    boids_module.combining,
    src_args=dict(
        gridRes=grid_res,
        gridVol=grid_vol,
    ),
    shader_args=dict(radius=perception_radius),
    debug=True,
)
# One all boids have finished calculating their next position and
# direction, it is safe to make them their current positions and
# directions. And with that, we are *done*... with the math part.
movement_actualizer = Copy(
    data_buffer,
    (('boids', 'nextPos'), ('boids', 'pos')),
    (('boids', 'nextDir'), ('boids', 'dir')),
)


# User interfaces are also not made of the stuff that is relevant here,
# and are quite specific to an application and its setup, so I've spun
# this off into the boids module as well.
boids_module.make_ui(mover)

# This is a generic visualization of Vec3 data in an array in an SSBO.
# It creates a mesh containing as many glPoints as the array has
# elements, but does not store any vertex data. During rendering (which
# happens because a bounding box is also created), it then reads that
# data from the SSBO.
points = SSBOParticles(base.render, data_buffer, ('boids', 'pos'))


# As mentioned before, we just dispatch the randum number generator
# once, then we are done with it.
rng.dispatch()
# For the other shaders, however, we have to create a bunch of render
# bins, so tat they are guaranteed to be invoked in the correct order.
bin_mgr = CullBinManager.get_global_ptr()
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


# Theoretically we do have the option to download the SSBO data back to
# the CPU side of things, and convert it into a Python data structure.
# While anything but efficient, this allows you to debug or store data
# from the live system, or even use Panda3D to write non-graphical GPU
# applications that do a lot of math for you.
#
# The code to download and decode the data would look like this:
#
# ```python
# data = base.win.gsg.get_engine().extract_shader_buffer_data(
#     data_buffer.ssbo,
#     base.win.gsg,
# )
# py_data = data_buffer.unpack(data)
# ```
#
# The structure of the data is the same as would have been fed into
# `Buffer(initial_data=..)`, but would consist solely of tuples.


# And with that, let's enjoy some murmurations.
base.run()
