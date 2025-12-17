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
