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
base.cam.set_pos(0.5, -2.0, 0.5)
base.accept('escape', base.task_mgr.stop)
PStatClient.connect()
base.set_frame_rate_meter(True)


# The data
num_elements = 2**16
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
#from panda3d.core import CullBinManager
#compute_bin = CullBinManager.get_global_ptr().add_bin("preliminary_compute_pass", CullBinManager.BT_fixed, 0)
rng = MurmurHash(
#rng = PermutedCongruentialGenerator(
    data_buffer,
    ('particles', 'position'),
    debug=True,
)
print("Shaders created.")


rng.attach(points.get_np(), task=((), {}))
#rng.dispatch()
print("Shaders dispatched.")


# particle_setupper = Shim(
#     ssbo,
#     "",
#     "particles[gl_GlobalInvocationID.x].direction = normalize(particles[gl_GlobalInvocationID.x].position - 0.5);",
#     (num_elements // 32, 1, 1),
# )


# mover_header = "uniform float dt;"
# mover_body = """
#   Particle p = particles[gl_GlobalInvocationID.x];
#   vec3 newPos = p.position + p.direction * 1.0/60.0/10.0;//dt;
#   float distFromCenter = length(newPos - 0.5);
#   float newLength = fract(distFromCenter);
#   newPos = normalize(newPos - 0.5) * newLength + 0.5;
#   particles[gl_GlobalInvocationID.x].position = newPos;
# """
# particle_mover = Shim(
#     ssbo,
#     mover_header,
#     mover_body,
#     (num_elements // 32, 1, 1),
#     debug=True,
# )


# hash_spatially = """const float gridSize = 32.0;
# 
# uint pcg_hash(uint input) {
#   uint state = input * 747796405u + 2891336453u;
#   uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
#   return (word >> 22u) ^ word;
# }
# 
# uint spatialHash (vec3 pos) {
#   uvec3 gridCell = floor(pos * gridSize);
#   uint cellCode = gridCell.x + gridCell.y * int(gridSize) + gridCell.z * int(pow(gridSize, 2));
#   uint hash = mod(pcg_hash(cellCode), 1000);
#   return hash;
# }
# """
# hash_particle_positions = Shim(
#     ssbo,
#     hash_spatially,
#     "particles[gl_GlobalInvocationID.x].hash = spatialHash(particles[gl_GlobalInvocationID.x].position * gridSize);",
#     (num_elements // 32, 1, 1),
# )
# sort_particle_hashes = BitonicSort(ssbo, ('particles', 'hash'))
# 
# 
# rng.dispatch()
# particle_setupper.dispatch()
# 
# 
# np = points.get_np()
# particle_mover.attach(np, bin_sort=0)
# hash_particle_positions.attach(np, bin_sort=1)
# sort_particle_hashes.attach(np, bin_sort=2)
# # zero_out_hash_key_table
# # fill_hash_key_table
# # boid_particles.attach(np, bin_sort=2)
# 
# # Data extraction
# #data = base.win.gsg.get_engine().extract_shader_buffer_data(
# #    ssbo.get_buffer(),
# #    base.win.gsg,
# #)
# 
# 
# def update_shader_inputs(task):
#     particle_mover.update(dt=globalClock.dt)
#     return task.cont
# base.task_mgr.add(update_shader_inputs, sort=-10)
# 
# 
# camera_gimbal = base.render.attach_new_node("")
# camera_gimbal.set_pos(0.5, 0.5, 0.5)
# base.cam.reparent_to(camera_gimbal)
# base.cam.set_y(-2.5)
# def rotate_camera(task):
#     camera_gimbal.set_h(camera_gimbal.get_h() + globalClock.dt * 10.0)
#     return task.cont
# base.task_mgr.add(rotate_camera)


print("Starting application...")
base.run()
