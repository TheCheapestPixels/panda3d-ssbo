from jinja2 import Template

from panda3d.core import BoundingVolume
from panda3d.core import NodePath
from panda3d.core import ComputeNode
from panda3d.core import Shader
from panda3d.core import ShaderAttrib


raw_code_template = """#version 430
layout (local_size_x = 32, local_size_y = 1) in;

{{ssbo}}

{{funcs}}

void main() {
{{main}}
}
"""


class RawGLSL:
    def __init__(self, ssbo, target_array,
                 funcs_source, main_source,
                 debug=False, src_args=None, shader_args=None):
        struct = ssbo.get_field(target_array)
        dims = struct.get_num_elements()
        if src_args == None:
            src_args = dict()
        render_args = dict(
            ssbo=ssbo.full_glsl(),
            funcs=funcs_source,
            main=main_source,
        )
        template = Template(raw_code_template)
        assembled_source = template.render(**render_args)
        source = Template(assembled_source).render(**src_args)
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
