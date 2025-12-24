from jinja2 import Template

from panda3d.core import BoundingVolume
from panda3d.core import NodePath
from panda3d.core import ComputeNode
from panda3d.core import Shader
from panda3d.core import ShaderAttrib


copy_template = """#version 430
layout (local_size_x = 32, local_size_y = 1) in;

{{ssbo}}

void main() {
  uint idx = gl_GlobalInvocationID.x;
  {{target_array}}[idx].{{target_field}} = {{source_array}}[idx].{{source_field}};
}
"""


class Copy:
    def __init__(self, ssbo, copy, debug=False):
        ((source_array, source_field), (target_array, target_field)) = copy
        struct = ssbo.get_field(source_array)
        dims = struct.get_num_elements()
        render_args = dict(
            ssbo=ssbo.full_glsl(),
            source_array=source_array,
            source_field=source_field,
            target_array=target_array,
            target_field=target_field,
        )
        template = Template(copy_template)
        source = template.render(**render_args)
        if debug:
            for line_nr, line_txt in enumerate(source.split('\n')):
                print(f"{line_nr+1:4d}  {line_txt}")
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
