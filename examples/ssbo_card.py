from jinja2 import Template

from panda3d.core import CardMaker
from panda3d.core import Shader


vertex_source = """#version 430
uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 vertex;
in vec2 texcoord;
out vec2 v_texcoord;

void main() {
  gl_Position = p3d_ModelViewProjectionMatrix * vertex;
  v_texcoord = texcoord;
}
"""

fragment_template = """#version 430
uniform sampler2D p3d_Texture0;

in vec2 v_texcoord;

out vec4 p3d_FragColor;

{{ssbo}}

void main() {
  int idx = int(floor(v_texcoord.x * float({{array}}.length())));
  p3d_FragColor = vec4({{array}}[idx].{{key}}, 0, 0, 1);
}
"""


class SSBOCard:
    def __init__(self, parent, data_buffer, array_and_key):
        array_name, key = array_and_key
        render_args = dict(
            ssbo=data_buffer.full_glsl(),
            array=array_name,
            key=key,
        )
        template = Template(fragment_template)
        fragment_source = template.render(**render_args)
        vis_shader = Shader.make(
            Shader.SL_GLSL,
            vertex=vertex_source,
            fragment=fragment_source,
        )
        cm = CardMaker('card')
        card = parent.attach_new_node(cm.generate())
        card.set_shader(vis_shader)
        card.set_shader_input(
            data_buffer.glsl_type_name,
            data_buffer.ssbo,
        )
        self.card = card

    def get_np(self):
        return self.card
