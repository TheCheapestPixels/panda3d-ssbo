import enum

from jinja2 import Template

from panda3d.core import NodePath
from panda3d.core import CardMaker
from panda3d.core import Shader
from panda3d.core import CullBinManager


class GraphStyle:
    def __init__(self, bars=True, low=(1, 1, 1), high=(1, 1, 1),
                 background=(0, 0, 0)):
        self.bars = bars
        self.low = low
        self.high = high
        self.bg = background


vertex_source = """
#version 430
uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 vertex;
in vec2 texcoord;
out vec2 v_texcoord;

void main() {
  gl_Position = p3d_ModelViewProjectionMatrix * vertex;
  v_texcoord = texcoord;
}
""".strip()


fragment_template = """
#version 430
uniform sampler2D p3d_Texture0;

in vec2 v_texcoord;

out vec4 p3d_FragColor;

{{ssbo}}

void main() {
  int idx = int(floor(v_texcoord.x * float({{array}}.length())));
  float value_data = {{array}}[idx].{{key}};
  float value_pos = v_texcoord.y;
  vec3 color_chart = mix(vec3({{low}}), vec3({{high}}), value_data);
  vec3 color_background = vec3({{background}});
  {{graph}}
}
""".strip()
bar_chart_template = """
  if (value_data >= value_pos) {
    p3d_FragColor = vec4(color_chart, 1);
  } else {
    p3d_FragColor = vec4(color_background, 1);
  }
""".strip()
line_chart_template = """
  float thickness = 0.01;
  float dist = abs(value_data - value_pos);
  float lineStrength = max((thickness-dist) / thickness, 0.0);
  vec3 color = mix(color_background, color_chart, lineStrength); {}
  p3d_FragColor = vec4(color, 1);
""".strip()


class SSBOCard:
    # pass value_buffer=True if the buffer does not contain structs
    def __init__(self, parent: NodePath, data_buffer, array_and_key,
                 fullscreencard=False, style=None, debug=False):
        array_name, key = array_and_key
        if style is None:
            style = GraphStyle()
        if style.bars:
            graph_style = bar_chart_template
        else:
            graph_style = line_chart_template
        render_args = dict(
            ssbo=data_buffer.full_glsl(),
            array=array_name,
            key=key,
            low=f"{style.low[0]}, {style.low[1]}, {style.low[2]}",
            high=f"{style.high[0]}, {style.high[1]}, {style.high[2]}",
            background=f"{style.bg[0]}, {style.bg[1]}, {style.bg[2]}",
            graph=graph_style,
        )
        template = Template(fragment_template)
        fragment_source = template.render(**render_args)
        if debug:
            print(self.__class__.__name__ + "::vertex")
            for line_nr, line_txt in enumerate(vertex_source.split('\n')):
                print(f"{line_nr:4d}  {line_txt}")
            print(self.__class__.__name__ + "::fragment")
            for line_nr, line_txt in enumerate(fragment_source.split('\n')):
                print(f"{line_nr:4d}  {line_txt}")

        vis_shader = Shader.make(
            Shader.SL_GLSL,
            vertex=vertex_source,
            fragment=fragment_source,
        )
        if vis_shader is None:
            print("Couldn't compile SSBOCard shaders!")
            raise Exception
        cm = CardMaker('card')
        if fullscreencard:
            cm.setFrameFullscreenQuad()
        card = parent.attach_new_node(cm.generate())
        # add a fixed bin between opaque and transparent
        CullBinManager.get_global_ptr().add_bin("SSBOCard", 
                                                CullBinManager.BT_fixed, 20)
        card.set_shader(vis_shader)
        card.set_bin("SSBOCard", 25)
        card.set_shader_input(
            data_buffer.glsl_type_name,
            data_buffer.ssbo,
        )
        self.card = card

    def get_np(self):
        return self.card
