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
num_elements = 2**10  # Usually has to be a multiple of 32, but because of bitonic sort, it has to be a power of 2 equal or greater than 64.
grid_res = (4, 4, 4)  # Per-axis number of cells in the spatial hash grid. Product has to be a multiple of 32.
grid_vol = (1.0, 1.0, 1.0)
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
print("Creating data buffer...")
data_buffer = Buffer(
    'dataBuffer',
    boids('boids', num_elements),
    pivot('pivot', grid_res[0] * grid_res[1] * grid_res[2]),
    #boids('boids', num_elements, unbounded=True),
)
print("Data buffer created.")


# The compute shaders
rng = MurmurHash(
    data_buffer,
    ('boids', 'pos'),
    #('boids', 'dir'),
    debug=True,
)
spatial_hash = SpatialHash(
    data_buffer,
    ('boids', 'pos', 'hashIdx'),  # array, position field, hash field
    grid_vol,  # Hash grid volume
    grid_res,  # Hash grid resolution
    debug=True,
)
sorter = BitonicSort(
    data_buffer,
    ('boids', 'hashIdx'),
    debug=True,
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
    debug=True,
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
    debug=True,
)
mover_funcs = """
ivec3 cellIdxToCell (uint cellIdx, ivec3 res) {
  ivec3 cell = ivec3(mod(cellIdx, res.x),
                     mod(floor(cellIdx / res.x), res.y),
                     floor(cellIdx / (res.x * res.y)));
  return cell;
}

uint cellToCellIdx (ivec3 cell, ivec3 res) {
  uint cellIdx = cell.x + 
                 cell.y * res.x + 
                 cell.z * res.x * res.y;
  return cellIdx;
}

vec3 clampVec(vec3 v, float minL, float maxL) {
  float vL = length(v);
  float targetL = clamp(vL, minL, maxL);
  v *= targetL / vL;
  return v;
}
"""[1:-1]
mover_source = """
  float radius = 0.15;
  float sepRadius = 0.05;
  float minSpeed = 0.0;  // FIXME: Speed times dt
  float maxSpeed = 0.05 * (1.0/60.0);  // FIXME: Replace by dt

  // Which boid are we processing? Where is it?
  uint boidIdx = uint(gl_GlobalInvocationID.x);
  vec3 pos = boids[boidIdx].pos;

  // And where, in terms of spatial hash cell, are we?
  uint cellIdx = boids[boidIdx].hashIdx;
  ivec3 res = ivec3({{gridRes[0]}}, {{gridRes[1]}}, {{gridRes[2]}});
  vec3 vol = vec3({{vol[0]}}, {{vol[1]}}, {{vol[2]}});
  vec3 cellSize = vec3(vol / res);
  ivec3 cell = cellIdxToCell(cellIdx, res);

  // The radius, how many cells does it cover?
  // From where to where will we scan the cell grid?
  ivec3 reach = ivec3(ceil(vec3(radius) / cellSize));
  ivec3 lower = cell - reach;
  lower = max(lower, ivec3(0));
  lower = min(lower, res);
  ivec3 upper = (cell + reach);
  upper = max(upper, ivec3(0));
  upper = min(upper, res);

  // Value accumulators for the boid.
  uint otherVecs = 0;
  vec3 cohesion = vec3(0);
  vec3 separation = vec3(0);

  uint scanIdx;
  // For each cell that is considered relevant (because its volume is
  // less than radius away from the boid's cell), ...
  for (int x=lower.x; x<=upper.x; x++) {
    for (int y=lower.y; y<=upper.y; y++) {
      for (int z=lower.z; z<=upper.z; z++) {
        // ...consider all boids in it, ...
        scanIdx = cellToCellIdx(ivec3(x, y, z), res);
        Pivot p = pivot[scanIdx];
        for (uint idx = p.start; idx < p.start + p.len; idx++) {
          if (idx != boidIdx) {  // Don't consider yourself, boid!
            vec3 toBoid = boids[idx].pos - pos;
            // ...and if the boid is in range, ...
            if (length(toBoid) <= radius) {
              // ...then add the boid-boid calculation to the pile.
              otherVecs++;
              // Cohesion
              cohesion += toBoid;
              separation += normalize(-toBoid) * max(0, sepRadius - length(toBoid));
              // Alignment: FIXME
            }
          }
        }
      }
    }
  }
  // We can also do rules that do not involve other boids.
  // e.g. the walls repel the boid.

  if (otherVecs > 0) {
    vec3 move = ((cohesion + 3.0 * separation) / 4.0) / otherVecs;
    move = clampVec(move, minSpeed, maxSpeed);
    boids[boidIdx].nextPos = min(max(pos + move, vec3(0)), vec3(1));
  } else {
    boids[boidIdx].nextPos = pos;
  }
"""[1:-1]
mover = RawGLSL(
    data_buffer,
    'boids',  # Array for determining the number of invocations
    mover_funcs,  # source code of functions
    mover_source,  # source code of the main function
    src_args=dict(
        gridRes=grid_res,
        vol=grid_vol,
    ),
    debug=True,
)
movement_actualizer_source = """
  uint idx = gl_GlobalInvocationID.x;
  boids[idx].pos = boids[idx].nextPos;
"""[1:-1]
movement_actualizer = RawGLSL(
    data_buffer,
    'boids',
    "",
    movement_actualizer_source,
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
    rng.dispatch()
    stages = [
        ("cmp_spatial_hash", spatial_hash),
        ("cmp_sort_spatial_hashes", sorter),
        ("cmp_pivot_table_starts", pivot_start),
        ("cmp_pivot_table_length", pivot_length),
        ("cmp_mover", mover),
        ("cmp_mover_2", movement_actualizer),
    ]
    for idx, (bin_name, shader) in enumerate(stages):
        bin_mgr.add_bin(bin_name, CullBinManager.BT_fixed, -20+idx)
        shader.attach(points.get_np(), bin_name)
        print(f"Shader {bin_name} attached.")
print("Shaders defined.")
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
#base.task_mgr.step()
