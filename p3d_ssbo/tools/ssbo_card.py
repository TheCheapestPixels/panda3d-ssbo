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
  float value = {{array}}[idx].{{key}};
  if (value >= v_texcoord.y) {
    // 'bar chart' style

    p3d_FragColor = vec4(1.);
    // 'heat map' style
    // p3d_FragColor = vec4(1.0 - value, value, 0., 1.);

  } else {
    p3d_FragColor = vec4(0, 0, 0, 1);
  }
}
"""


class SSBOCard:
    # this constructor takes either a name for the ssbo data
    def __init__(self, parent: NodePath, data_buffer: gltypes.Buffer, *args):
        if len(*args) < 2:
            # buffer contains values
            array_name = *args
            render_args = dict(
                ssbo=data_buffer.glsl(),
                array=array_name,
            )
        elif len(*args) == 2:
            array_name, key = *args
            render_args = dict(
                ssbo=data_buffer.full_glsl(),
                array=array_name,
                key=key,
            )
        else:
            # this should really only have two overloads right?
            # maybe in future this could handle multiple structs
            raise Exception("SSBOCard *args should contain a name for the " + 
                            "SSBO contents (str for values, iterable[str] for structs)")
        template = template(fragment_template)
        fragment_source = template.render(**render_args)
        vis_shader = shader.make(
            shader.sl_glsl,
            vertex=vertex_source,
            fragment=fragment_source,
        )
        cm = cardmaker('card')
        card = parent.attach_new_node(cm.generate())
        card.set_shader(vis_shader)
        card.set_shader_input(
            data_buffer.glsl_type_name,
            data_buffer.ssbo,
        )
        self.card = card

    def get_np(self):
        return self.card
