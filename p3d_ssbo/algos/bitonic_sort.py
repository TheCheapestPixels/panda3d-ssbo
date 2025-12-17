from math import log2

from jinja2 import Template

from panda3d.core import BoundingVolume
from panda3d.core import NodePath
from panda3d.core import ComputeNode
from panda3d.core import Shader
from panda3d.core import ShaderAttrib


sorter_template = """#version 430
layout (local_size_x = 32, local_size_y = 1) in;

{{ssbo}}

uniform int span;
uniform int reverseSpan;

void compare(int low, int high) {
  {{type_name}} dataLow = {{array_name}}[low];
  {{type_name}} dataHigh = {{array_name}}[high];
  if (dataLow.{{key}} > dataHigh.{{key}}) {
    {{array_name}}[low] = dataHigh;
    {{array_name}}[high] = dataLow;
  }
}

void main() {
  // From where to shere is this span?
  int idx = int(gl_GlobalInvocationID.x);
  int idxOfSpan = int(floor(idx / span));
  int spanBoundLow = idxOfSpan * span * 2;
  int spanBoundHigh = (idxOfSpan + 1) * span * 2 - 1;

  // In what direction does this span go, and where is its start *really*?
  int reversed = int(round(mod(idxOfSpan / reverseSpan, 2))) * (-2) + 1;  // +1 = straight, -1 = reversed
  int spanStart = abs(min(spanBoundLow * reversed, spanBoundHigh * reversed));

  // Which pair of elements do we compare?
  int idxInSpan = int(round(mod(idx, span)));
  int idxLow = spanStart + idxInSpan * reversed;
  int idxHigh = idxLow + span * reversed;

  // Compare, and switch if necessary.
  compare(idxLow, idxHigh);
}
"""


class BitonicSort:
    def __init__(self, ssbo, array_and_key, debug=False):
        array_name, key = array_and_key
        dims = ssbo.get_field(array_name).get_num_elements()
        assert len(dims) == 1, "Only 1D arrays for now."
        num_elements = dims[0]
        render_args = dict(
            ssbo=ssbo.full_glsl(),
            type_name=ssbo.get_field(array_name).glsl_type_name,
            array_name=array_name,
            key=key,
        )
        template = Template(sorter_template)
        source = template.render(**render_args)
        if debug:
            for line_nr, line_txt in enumerate(source.split('\n')):
                print(f"{line_nr:4d}  {line_txt}")
        shader = Shader.make_compute(Shader.SL_GLSL, source)
        workgroups = (num_elements // 64, 1, 1)
        self.ssbo = ssbo
        self.shader = shader
        self.workgroups = workgroups
        sorter_arrays = []
        for e in range(int(log2(num_elements))):
            for s in range(e, -1, -1):
                sorter_arrays.append((2**s, 2**(e-s)))
        self.sorter_arrays = sorter_arrays

    def dispatch(self):
        np = NodePath("dummy")
        np.set_shader(self.shader)
        np.set_shader_input(
            self.ssbo.glsl_type_name,
            self.ssbo.ssbo,
        )
        for span, reverse_span in self.sorter_arrays:
            np.set_shader_input('span', span)
            np.set_shader_input('reverseSpan', reverse_span)
            sattr = np.get_attrib(ShaderAttrib)
            base.graphicsEngine.dispatch_compute(
                self.workgroups,
                sattr,
                base.win.get_gsg(),
            )

    def attach(self, np, bin_name):
        for idx, (span, reverse_span) in enumerate(self.sorter_arrays):
            cn = ComputeNode(f"BitonicSort-{idx}")
            cn.add_dispatch(self.workgroups)
            cnnp = np.attach_new_node(cn)
            cnnp.set_shader(self.shader)
            cnnp.set_shader_input(self.ssbo.glsl_type_name, self.ssbo.ssbo)
            cnnp.set_shader_input('span', span)
            cnnp.set_shader_input('reverseSpan', reverse_span)
            cnnp.set_bin(bin_name, idx)
            cn.set_bounds_type(BoundingVolume.BT_box)
            cn.set_bounds(np.get_bounds())
