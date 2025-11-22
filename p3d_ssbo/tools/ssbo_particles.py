from jinja2 import Template

from panda3d.core import GeomVertexFormat
from panda3d.core import GeomVertexData
from panda3d.core import GeomEnums
from panda3d.core import GeomPoints
from panda3d.core import Geom
from panda3d.core import GeomNode
from panda3d.core import BoundingVolume
from panda3d.core import BoundingBox
from panda3d.core import Shader


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


class SSBOParticles:
    def __init__(self, parent, data_buffer, array_and_key):
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

    def set_up_particle_visualization(self, parent, num_elements):
        # We do not store any vertices or primitives, we'll just
        # calculate the relevant data in the vertex shader. Therefore
        # this node has no implicit bounding volume, so we'll have to
        # set one explicitly, or the node would get culled.
        v_format = GeomVertexFormat.get_empty()
        v_data = GeomVertexData(
            'particles',
            v_format,
            GeomEnums.UH_static,
        )
        points = GeomPoints(GeomEnums.UH_static)
        points.add_next_vertices(num_elements)
        geom = Geom(v_data)
        geom.add_primitive(points)
        geom.set_bounds(BoundingBox((0, 0, 0), (1, 1, 1)))
        node = GeomNode("node")
        node.add_geom(geom)
        node.set_bounds_type(BoundingVolume.BT_box)
        path = parent.attach_new_node(node)
        path.show_bounds()
        return path
