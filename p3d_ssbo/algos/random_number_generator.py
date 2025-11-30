import random

from jinja2 import Template

from panda3d.core import Vec3
from panda3d.core import BoundingVolume
from panda3d.core import NodePath
from panda3d.core import ComputeNode
from panda3d.core import Shader
from panda3d.core import ShaderAttrib


pcg_rng_template = """#version 430
#extension GL_ARB_gpu_shader_int64 : require

layout (local_size_x = 32, local_size_y = 1) in;

uniform int rngSeed;

{{ssbo}}

uint64_t state = 0;
const uint64_t multiplier = 6364136223846793005ul;
const uint64_t increment = 1442695040888963407ul;

uint rotr32(uint x, uint r)
{
    return (x >> r) | (x << (32 - r));
}

uint pcg32()
{
    uint64_t x = state;
    uint count = uint(x >> 59);

    state = x * multiplier + increment;
    x ^= x >> 18;
    return rotr32(uint(x >> 27), count);
}

void pcg32_init(uint64_t seed)
{
    state = seed + increment;
    pcg32();
}

float pcgf()
{
    return (pcg32() / float((2<<62) - 1)) / 2.0;
}

void main() {
  int idx = int(gl_GlobalInvocationID.x);
  pcg32_init(uint64_t(idx + rngSeed));

  {% for array, key, field_type in targets %}// {{array}}.{{key}} = {{field_type}}
  {% if field_type=='float' %}{{array}}[idx].{{key}} = pcgf();
  {% elif field_type=='vec3' %}{{array}}[idx].{{key}} = vec3(pcgf(), pcgf(), pcgf());
  {% endif %}{% endfor %}
}
"""


mmh3_32_rng_template = """#version 430
layout (local_size_x = 32, local_size_y = 1) in;

uniform uint rngSeed;

{{ssbo}}

uint state = 0;

uint murmur_32_scramble(uint k) {
  k *= 0xcc9e2d51;
  k = (k << 15) | (k >> 17);
  k *= 0x1b873593;
  return k;
}

void mmh3_32_single_round(uint k) {
  state = state ^ murmur_32_scramble(k);
  state = (state << 13) | (state >> 19);
  state = state * 5 + 0xe6546b64;

  state = state ^ uint(1);
  state = state ^ state >> 16;
  state = state * 0x85ebca6b;
  state = state ^ state >> 13;
  state = state * 0xc2b2ae35;
  state = state ^ state >> 16;
}

float mmh3() {
  mmh3_32_single_round(state ^ uint(gl_GlobalInvocationID.x));
  return float(state) / 4294967295.0;
}

void main() {
  state = rngSeed;
  uint idx = uint(gl_GlobalInvocationID.x);

  {% for array, key, field_type in targets %}// {{array}}.{{key}} = {{field_type}}
  {% if field_type=='float' %}{{array}}[idx].{{key}} = mmh3();
  {% elif field_type=='vec3' %}{{array}}[idx].{{key}} = vec3(mmh3(), mmh3(), mmh3());
  {% endif %}{% endfor %}
}
"""


class RandomNumberGenerator:
    def __init__(self, ssbo, *targets, debug=False):
        dims = None
        rng_specs = []
        for target in targets:
            array_name, key = target
            struct = ssbo.get_field(array_name)
            field_type = struct.get_field(key).glsl_type_name
            if dims is None:
                dims = struct.get_num_elements()
                assert len(dims) == 1, "Just 1D arrays for now."
            else:
                assert dims == struct.get_num_elements(), "Working on differently-sized arrays."
            rng_specs.append(
                (array_name, key, field_type)
            )
        render_args = dict(
            ssbo=ssbo.full_glsl(),
            targets=rng_specs,
        )
        template = Template(self.rng_template)
        source = template.render(**render_args)
        if debug:
            for line_nr, line_txt in enumerate(source.split('\n')):
                print(f"{line_nr:4d}  {line_txt}")
        shader = Shader.make_compute(Shader.SL_GLSL, source)
        workgroups = (dims[0] // 32, 1, 1)
        self.ssbo = ssbo
        self.shader = shader
        self.workgroups = workgroups

    def dispatch(self, seed=0):
        np = NodePath("dummy")
        np.set_shader(self.shader)
        np.set_shader_input(self.ssbo.glsl_type_name, self.ssbo.ssbo)
        np.set_shader_input('rngSeed', seed)
        sattr = np.get_attrib(ShaderAttrib)
        base.graphicsEngine.dispatch_compute(
            self.workgroups,
            sattr,
            base.win.get_gsg(),
        )

    def attach(self, np, bin_sort=0, seed=None, task=None):
        cn = ComputeNode(f"PermutedCongruentialGenerator")
        cn.add_dispatch(self.workgroups)
        cnnp = np.attach_new_node(cn)

        cnnp.set_shader(self.shader)
        cnnp.set_shader_input(self.ssbo.glsl_type_name, self.ssbo.ssbo)
        if seed is None:
            seed = random.randint(0,2**31-1)
        cnnp.set_shader_input('rngSeed', seed)

        cnnp.set_bin("preliminary_compute_pass", bin_sort, 0)
        cn.set_bounds_type(BoundingVolume.BT_box)
        cn.set_bounds(np.get_bounds())
        self.cnnp = cnnp
        if task is not None:
            args, kwargs = task
            base.task_mgr.add(self.update, *args, **kwargs)

    def update(self, task):
        seed = random.randint(0, 2**31-1)
        self.cnnp.set_shader_input('rngSeed', seed)
        return task.cont


class PermutedCongruentialGenerator(RandomNumberGenerator):
    rng_template = pcg_rng_template


class MurmurHash(RandomNumberGenerator):
    rng_template = mmh3_32_rng_template
