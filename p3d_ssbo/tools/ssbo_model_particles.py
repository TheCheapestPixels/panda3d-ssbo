from jinja2 import Template

vertex_template = """#version 430

{{ssbo}}

uniform mat4 p3d_ModelViewProjectionMatrix;

void main() {
  vec3 pos = {{array}}[gl_VertexID].{{key}};
  gl_Position = p3d_ModelViewProjectionMatrix * vec4(pos, 1);
}
"""

fragment_template = """
#version 430

out vec4 p3d_FragColor;

void main() {
  p3d_FragColor = vec4(1, 1, 1, 1);
}
"""


class SSBOModelParticles:
    def __init__(self, parent, model, data_buffer, array_and_key):
        array_name, key = array_and_key
        render_args = dict(
            ssbo=data_buffer.full_glsl(),
            array=array_name,
            key=key,
        )
        vert_template = Template(vertex_template)
        vert_source = vert_template.render(**render_args)
        frag_template = Template(fragment_template)
        frag_source = frag_template.render(**render_args)
        vis_shader = Shader.make(
            Shader.SL_GLSL,
            vertex=vert_source,
            fragment=frag_source,
        )
        num_particles = data_buffer.get_field(array_name).get_num_elements()[0]
        particles = self.set_up_particle_visualization(parent, num_particles)
        particles.set_shader(vis_shader)
        particles.set_shader_input(
            data_buffer.glsl_type_name,
            data_buffer.ssbo,
        )
        self.particles = particles
        
    def get_np(self):
        return self.particles
