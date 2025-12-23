from jinja2 import Template

from panda3d.core import BoundingVolume
from panda3d.core import NodePath
from panda3d.core import ComputeNode
from panda3d.core import Shader
from panda3d.core import ShaderAttrib


spatial_hash_template = """#version 430
layout (local_size_x = 32, local_size_y = 1) in;

{{ssbo}}

uvec3 resolution = uvec3({{res[0]}}, {{res[1]}}, {{res[2]}});
vec3 volume = vec3({{vol[0]}}, {{vol[1]}}, {{vol[2]}});
vec3 edges = volume / resolution;

uint spatialHash ({{type}} pos) {
  uvec3 cellV = uvec3(floor(pos / edges));
  uint cell = cellV.x + 
              cellV.y * resolution.x + 
              cellV.z * resolution.x * resolution.y;
  return cell;
}

void main() {
  uint idx = uint(gl_GlobalInvocationID.x);
  {{array}}[idx].{{hash}} = spatialHash({{array}}[idx].{{key}});
}
"""


class SpatialHash:
    def __init__(self, ssbo, target, volume, resolution, debug=False):
        target_array, target_pos, target_hash = target
        struct = ssbo.get_field(target_array)
        dims = struct.get_num_elements()
        assert len(dims) == 1, "Just 1D arrays for now."

        pos_type = struct.get_field(target_pos).glsl_type_name
        if pos_type in ('vec2', 'uvec2'):  # 2D space
            assert len(volume) == 2
            assert len(resolution) == 2
        elif pos_type in ('vec3', 'uvec3'):  # 3D space
            assert len(volume) == 3
            assert len(resolution) == 3
        else:
            assert False, "Unsupported position type"

        render_args = dict(
            ssbo=ssbo.full_glsl(),
            array=target_array,
            key=target_pos,
            hash=target_hash,
            type=pos_type,
            vol=volume,
            res=resolution,
        )
        template = Template(spatial_hash_template)
        source = template.render(**render_args)
        if debug:
            for line_nr, line_txt in enumerate(source.split('\n')):
                print(f"{line_nr:4d}  {line_txt}")
        shader = Shader.make_compute(Shader.SL_GLSL, source)
        workgroups = (dims[0] // 32, 1, 1)
        self.ssbo = ssbo
        self.shader = shader
        self.workgroups = workgroups

    def dispatch(self):
        np = NodePath("dummy")
        np.set_shader(self.shader)
        np.set_shader_input(self.ssbo.glsl_type_name, self.ssbo.ssbo)
        sattr = np.get_attrib(ShaderAttrib)
        base.graphicsEngine.dispatch_compute(
            self.workgroups,
            sattr,
            base.win.get_gsg(),
        )

    def attach(self, np, bin_name):
        cn = ComputeNode(self.__class__.__name__)
        cn.add_dispatch(self.workgroups)
        cnnp = np.attach_new_node(cn)

        cnnp.set_shader(self.shader)
        cnnp.set_shader_input(self.ssbo.glsl_type_name, self.ssbo.ssbo)

        cnnp.set_bin(bin_name, 0)
        cn.set_bounds_type(BoundingVolume.BT_box)
        cn.set_bounds(np.get_bounds())
        self.cnnp = cnnp


pivot_start_source = """
#version 430

layout (local_size_x = 32, local_size_y = 1) in;

{{ssbo}}

void main() {
  uint idx = uint(gl_GlobalInvocationID.x);
  uint key = {{list_field}}[idx].{{list_key}};
  uint diff;
  if (idx == 0) {
    // This is the first element with the lowest index, and thus we'll
    // have to write it. So, how much further than the non-existant 
    // index -1 are we?
    diff = key + 1;
  } else {
    // A regular element. How many cell indices further are we than the
    // last boid was? (If we're still in the same cell, that's 0.)
    uint keyLeft = {{list_field}}[idx-1].{{list_key}};
    diff = key - keyLeft;
  }
  // If this is such an cell-index-advancing element, we write its boid
  // index into the pivot table as the start of a run. Since we may have
  // advanced by *several* cell indices, we'll also need to set the
  // start of the skipped ones; We set it to this element, so the
  // runlength for them will be 0.
  if (diff > 0) {
    for (uint pivotIdx = key; pivotIdx > key - diff; pivotIdx--) {
      {{table_field}}[pivotIdx].start = idx;
    }
  }
  // If this is the last boid, we may need to set the start of any
  // remaining cell indices, and we set them to "just beyond the end of
  // the list of boids."
  if (idx+1 == {{list_field}}.length()) {
    for (uint pivotIdx = key + 1; pivotIdx < pivot.length(); pivotIdx++) {
      {{table_field}}[pivotIdx].{{table_start}} = {{list_field}}.length();
    }
  }
}
"""[1:]


pivot_length_source = """
#version 430

layout (local_size_x = 32, local_size_y = 1) in;

{{ssbo}}

void main() {
  uint pivotIdx = uint(gl_GlobalInvocationID.x);
  uint start = pivot[pivotIdx].start;
  uint end;
  if (pivotIdx == pivot.length() - 1) {
    end = boids.length();
  } else {
    end = pivot[pivotIdx + 1].start;
  }
  pivot[pivotIdx].len = end - start;
}
"""[1:]


class PivotTable:
    def __init__(self, ssbo, key, table, debug=False):
        self.ssbo = ssbo

        list_field, list_key = key
        list_struct = ssbo.get_field(list_field)
        list_dims = list_struct.get_num_elements()
        table_field, table_start, table_len = table
        table_struct = ssbo.get_field(table_field)
        table_dims = table_struct.get_num_elements()

        # Start shader
        render_args_start = dict(
            ssbo=ssbo.full_glsl(),
            list_field=list_field,
            list_key=list_key,
            table_field=table_field,
            table_start=table_start,
        )
        template_start = Template(pivot_start_source)
        source_start = template_start.render(**render_args_start)
        if debug:
            for line_nr, line_txt in enumerate(source_start.split('\n')):
                print(f"{line_nr+1:4d}  {line_txt}")
        shader_start = Shader.make_compute(Shader.SL_GLSL, source_start)
        workgroups_start = (list_dims[0] // 32, 1, 1)
        self.shader_start = shader_start
        self.workgroups_start = workgroups_start

        # Length shader
        render_args_length = dict(
            ssbo=ssbo.full_glsl(),
        )
        template_length = Template(pivot_length_source)
        source_length = template_length.render(**render_args_length)
        if debug:
            for line_nr, line_txt in enumerate(source_length.split('\n')):
                print(f"{line_nr+1:4d}  {line_txt}")
        shader_length = Shader.make_compute(Shader.SL_GLSL, source_length)
        workgroups_length = (table_dims[0] // 32, 1, 1)
        self.shader_length = shader_length
        self.workgroups_length = workgroups_length

    def dispatch(self):
        np = NodePath("dummy")
        np.set_shader(self.shader_start)
        np.set_shader_input(self.ssbo.glsl_type_name, self.ssbo.ssbo)
        sattr = np.get_attrib(ShaderAttrib)
        base.graphicsEngine.dispatch_compute(
            self.workgroups_start,
            sattr,
            base.win.get_gsg(),
        )

        np = NodePath("dummy")
        np.set_shader(self.shader_length)
        np.set_shader_input(self.ssbo.glsl_type_name, self.ssbo.ssbo)
        sattr = np.get_attrib(ShaderAttrib)
        base.graphicsEngine.dispatch_compute(
            self.workgroups_length,
            sattr,
            base.win.get_gsg(),
        )

    def attach(self, np, bin_name):
        cn_s = ComputeNode(self.__class__.__name__ + "_start")
        cn_s.add_dispatch(self.workgroups_start)
        cnnp_s = np.attach_new_node(cn_s)

        cnnp_s.set_shader(self.shader_start)
        cnnp_s.set_shader_input(self.ssbo.glsl_type_name, self.ssbo.ssbo)

        cnnp_s.set_bin(bin_name, 0)
        cn_s.set_bounds_type(BoundingVolume.BT_box)
        cn_s.set_bounds(np.get_bounds())
        self.cnnp_s = cnnp_s

        cn_l = ComputeNode(self.__class__.__name__ + "_length")
        cn_l.add_dispatch(self.workgroups_length)
        cnnp_l = np.attach_new_node(cn_l)

        cnnp_l.set_shader(self.shader_length)
        cnnp_l.set_shader_input(self.ssbo.glsl_type_name, self.ssbo.ssbo)

        cnnp_l.set_bin(bin_name, 1)
        cn_l.set_bounds_type(BoundingVolume.BT_box)
        cn_l.set_bounds(np.get_bounds())
        self.cnnp_l = cnnp_l
