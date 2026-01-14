from jinja2 import Template

from panda3d.core import NodePath
from panda3d.core import CardMaker
from panda3d.core import Shader
from panda3d.core import CullBinManager

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
  float value = {{array}}[idx].{{key}};
  if (value >= v_texcoord.y) {
    {{graph}}
  } else {
    p3d_FragColor = vec4(0, 0, 0, 1);
  }
}
""".strip()


class SSBOCard:
    # pass value_buffer=True if the buffer does not contain structs
    def __init__(self, parent: NodePath, data_buffer, *args, fullscreencard=False, barchart=False):
        if len(args) < 2:
            # buffer contains values
            array_name = args[0]
            render_args = dict(
                ssbo=data_buffer.glsl(),
                array=array_name,
                key='x',
            )
        elif len(args) >= 2:
            # buffer contains structs
            array_name, key = args
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
        if barchart:
            # show barchart-style data (hard edges)
            render_args['graph'] = "p3d_FragColor = vec4(1.);"
        else:
            # (default) show heatmap-style data (red gradient)
            render_args['graph'] = "p3d_FragColor = vec4(1.0 - value, value, 0., 1.);"
        template = Template(fragment_template)
        fragment_source = template.render(**render_args)
        vis_shader = Shader.make(
            Shader.SL_GLSL,
            vertex=vertex_source,
            fragment=fragment_source,
        )
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
