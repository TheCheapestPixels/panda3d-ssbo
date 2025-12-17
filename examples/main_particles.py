# TODO
#
# * Refactor Struct and SSBO to account for the fullness of std430.
# * SSBO.extract_data()
# * Give RNG distributions.
# * Changing inputs at runtime, e.g. seed for RNG.
# * Make workgroup size settable.
# * Bitonic sort: Make it work on other sizes than power-of-2s.
# * Add spatial hash grids.
# * ShaderWrapper.detach()
# * Evaluate only every nth frame

import random

from panda3d.core import PStatClient
from panda3d.core import Vec3

from direct.showbase.ShowBase import ShowBase

from p3d_ssbo.gltypes import GlVec3
from p3d_ssbo.gltypes import Struct
from p3d_ssbo.gltypes import Buffer
from p3d_ssbo.algos.random_number_generator import PermutedCongruentialGenerator
from p3d_ssbo.algos.random_number_generator import MurmurHash
from p3d_ssbo.algos.bitonic_sort import BitonicSort
from p3d_ssbo.tools.ssbo_particles import SSBOParticles


ShowBase()
gimbal = base.render.attach_new_node("gimbal")
gimbal.set_pos(0.5, 0.5, 0.5)
base.cam.reparent_to(gimbal)
base.cam.set_y(-2.5)
base.accept('escape', base.task_mgr.stop)
PStatClient.connect()
base.set_frame_rate_meter(True)


# The data
num_elements = 2**10
print(f"Configured for {num_elements} elements.")
def make_particle_buffer():
    particle_positions = [
        ((random.random(), random.random(), random.random()), )
        for _ in range(num_elements)
    ]
    particle_buffer_data = (particle_positions, )
    return particle_buffer_data


struct = Struct(
    'Particle',
    GlVec3('position'),
)
data_buffer = Buffer(
    'dataBuffer',
    struct('particles', num_elements),
    #initial_data=make_particle_buffer(),
)
print("SSBO created.")


# Visualization
points = SSBOParticles(base.render, data_buffer, ('particles', 'position'))
print("Visualization created.")


# The compute shaders
from panda3d.core import CullBinManager
bin_mgr = CullBinManager.get_global_ptr()
bin_mgr.add_bin("cmp_rng_particles", CullBinManager.BT_fixed, -10)


rng = MurmurHash(
#rng = PermutedCongruentialGenerator(
    data_buffer,
    ('particles', 'position'),
    #debug=True,
)
print("Shaders created.")


#rng.attach(points.get_np(), bin_name="cmp_rng_particles", task=((), {}))
rng.dispatch()
print("Shaders dispatched.")


def update_camera(task):
    gimbal.set_h(task.time * 20.0)
    return task.cont
base.task_mgr.add(update_camera)


print("Starting application...")
base.run()
