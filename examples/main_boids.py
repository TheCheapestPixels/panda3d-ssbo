from panda3d.core import PStatClient
print("pstats imported")
PStatClient.connect()
PStatClient.mainTick()
print("pstats connected")
from panda3d.core import Vec3

from direct.showbase.ShowBase import ShowBase

from p3d_ssbo.gltypes import GlVec3
from p3d_ssbo.gltypes import GlUInt
from p3d_ssbo.gltypes import Struct
from p3d_ssbo.gltypes import Buffer
from p3d_ssbo.algos.raw_glsl import RawGLSL
from p3d_ssbo.algos.random_number_generator import MurmurHash
from p3d_ssbo.algos.spatial_hash import SpatialHash
from p3d_ssbo.algos.bitonic_sort import BitonicSort
from p3d_ssbo.tools.ssbo_particles import SSBOParticles


ShowBase()
base.accept('escape', base.task_mgr.stop)
base.set_frame_rate_meter(True)


gimbal = base.render.attach_new_node("gimbal")
gimbal.set_pos(0.5, 0.5, 0.5)
base.cam.reparent_to(gimbal)
base.cam.set_y(-2.5)
def update_camera(task):
    gimbal.set_h(task.time * 20.0)
    return task.cont
base.task_mgr.add(update_camera)


from panda3d.core import CullBinManager
bin_mgr = CullBinManager.get_global_ptr()


# The data
num_elements = 2**19  # Usually has to be a multiple of 32, but because of bitonic sort, it has to be a power of 2 equal or greater than 64.
grid_res = (4, 4, 4)  # Per-axis number of cells in the spatial hash grid. Product has to be a multiple of 32.
boids = Struct(
    'Boid',
    GlVec3('pos'),
    #GlVec3('dir'),
    GlUInt('hashIdx'),
)
pivot = Struct(
    'Pivot',
    GlUInt('start'),
    GlUInt('len'),
)
print("Creating data buffer...")
data_buffer = Buffer(
    'dataBuffer',
    boids('boids', num_elements),
    pivot('pivot', grid_res[0] * grid_res[1] * grid_res[2])
)
print("Data buffer created.")


# The compute shaders
rng = MurmurHash(
    data_buffer,
    ('boids', 'pos'),
    #('boids', 'dir'),
)
spatial_hash = SpatialHash(
    data_buffer,
    ('boids', 'pos', 'hashIdx'),  # array, position field, hash field
    (1.0, 1.0, 1.0),  # Hash grid volume
    grid_res,  # Hash grid resolution
)
sorter = BitonicSort(
    data_buffer,
    ('boids', 'hashIdx'),
)
pivot_start_source = """
  uint idx = uint(gl_GlobalInvocationID.x);
  uint hash = boids[idx].hashIdx;
  uint diff;
  if (idx == 0) {
    // This is the first element with the lowest index, and thus we'll
    // have to write it. So, how much further than the non-existant 
    // index -1 are we?
    diff = hash + 1;
  } else {
    // A regular element. How many cell indices further are we than the
    // last boid was? (If we're still in the same cell, that's 0.)
    uint hashLeft = boids[idx-1].hashIdx;
    diff = hash - hashLeft;
  }
  // If this is such an cell-index-advancing element, we write its boid
  // index into the pivot table as the start of a run. Since we may have
  // advanced by *several* cell indices, we'll also need to set the
  // start of the skipped ones; We set it to this element, so the
  // runlength for them will be 0.
  if (diff > 0) {
    for (uint pivotIdx = hash; pivotIdx > hash - diff; pivotIdx--) {
      pivot[pivotIdx].start = idx;
    }
  }
  // If this is the last boid, we may need to set the start of any
  // remaining cell indices, and we set them to "just beyond the end of
  // the list of boids."
  if (idx+1 == boids.length()) {
    for (uint pivotIdx = hash + 1; pivotIdx < pivot.length(); pivotIdx++) {
      pivot[pivotIdx].start = boids.length();
    }
  }
"""[1:-1]
pivot_start =  RawGLSL(
    data_buffer,
    'boids',
    "",
    pivot_start_source,
    #debug=True,
)
pivot_length_source = """
  uint pivotIdx = uint(gl_GlobalInvocationID.x);
  uint start = pivot[pivotIdx].start;
  uint end;
  if (pivotIdx == pivot.length() - 1) {
    end = boids.length();
  } else {
    end = pivot[pivotIdx + 1].start;
  }
  pivot[pivotIdx].len = end - start;
"""[1:-1]
pivot_length =  RawGLSL(
    data_buffer,
    'pivot',
    "",
    pivot_length_source,
)
mover_source = """
  uint boidIdx = uint(gl_GlobalInvocationID.x);
  uint pivotIdx = boids[boidIdx].hashIdx;
  Pivot p = pivot[pivotIdx];
  vec3 pos = boids[boidIdx].pos;

  uint otherVecs = 0;
  vec3 sumOfDists;
  for (uint idx = p.start; idx < p.start + p.len; idx++) {
    if (idx != boidIdx) {
      otherVecs++;
      sumOfDists = boids[idx].pos - pos;
    }
  }
  if (otherVecs > 0) {
    boids[boidIdx].pos += sumOfDists / otherVecs * 0.2;
  }
"""[1:-1]
mover = RawGLSL(
    data_buffer,
    'boids',  # Array for determining the number of invocations
    "",  # source code of functions
    mover_source,  # source code of the main function
)

print("Creating particles...")
points = SSBOParticles(base.render, data_buffer, ('boids', 'pos'))
print("Particles created.")
def setup_debug():
    for shader in [rng, spatial_hash, sorter, pivot_start, pivot_length, mover.attach]:
        shader.dispatch()
        data = base.win.gsg.get_engine().extract_shader_buffer_data(
            data_buffer.ssbo,
            base.win.gsg,
        )
def setup_prod():
    #rng.dispatch()
    stages = [
        #("cmp_spatial_hash", spatial_hash),
        #("cmp_sort_spatial_hashes", sorter),
        #("cmp_pivot_table_starts", pivot_start),
        #("cmp_pivot_table_length", pivot_length),
        #("cmp_mover", mover),
    ]
    for idx, (bin_name, shader) in enumerate(stages):
        bin_mgr.add_bin(bin_name, CullBinManager.BT_fixed, -20+idx)
        shader.attach(points.get_np(), bin_name)
        print(f"Shader {bin_name} attached.")
print("Shaders definedy.")
setup_prod()
print("Shaders set up.")

#data = base.win.gsg.get_engine().extract_shader_buffer_data(
#    data_buffer.ssbo,
#    base.win.gsg,
#)
#py_data = data_buffer.unpack(data)
#import pdb; pdb.set_trace()
#for element in py_data[0]:
#    print(element[0])

print("Starting main loop...")
base.run()
