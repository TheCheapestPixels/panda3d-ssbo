from p3d_ssbo import GlType
from p3d_ssbo import Struct
from p3d_ssbo import SSBO


def test_struct():
    struct = Struct(
        'MyType',
        ('position', GlType.vec3),
        ('oldIndex', GlType.uint),
    )


def test_getting_struct_dependencies_empty():
    struct_a = Struct('A', ('dud', GlType.uint))
    deps = struct_a.get_dependencies()
    assert set(deps.keys()) == set([struct_a])
    assert deps[struct_a] == []


def test_getting_struct_dependencies_simple():
    struct_a = Struct('A', ('dud', GlType.uint))
    struct_b = Struct('B', ('hit', struct_a))
    deps = struct_b.get_dependencies()
    assert set(deps.keys()) == set([struct_a, struct_b])
    assert deps[struct_a] == []
    assert deps[struct_b] == [struct_a]


def test_getting_struct_dependencies_complex():
    struct_a = Struct('A', ('dud', GlType.uint))
    struct_b = Struct('B', ('hit_a', struct_a))
    struct_c = Struct('C', ('hit_a', struct_a))
    struct_d = Struct('D', ('hit_b', struct_b), ('hit_c', struct_c))
    deps = struct_d.get_dependencies()
    assert set(deps.keys()) == set([struct_a, struct_b, struct_c, struct_d])
    assert deps[struct_a] == []
    assert deps[struct_b] == [struct_a]
    assert deps[struct_c] == [struct_a]
    assert deps[struct_d] == [struct_b, struct_c]


def test_getting_ordered_dependencies_simple():
    struct_a = Struct('A', ('dud', GlType.uint))
    struct_b = Struct('B', ('hit_a', struct_a))
    struct_c = Struct('C', ('hit_a', struct_a))
    struct_d = Struct('D', ('hit_b', struct_b), ('hit_c', struct_c))
    order = struct_d.get_ordered_dependencies()
    assert order == [struct_a, struct_b, struct_c, struct_d]


def test_getting_ordered_dependencies_complex():
    struct_a = Struct('A', ('dud', GlType.uint))
    struct_b = Struct('B', ('hit_a', struct_a))
    struct_c = Struct('C', ('hit_a', struct_a))
    struct_d = Struct('D', ('hit_b', struct_b), ('hit_c', struct_c))
    struct_e = Struct('E', ('dud', GlType.uint))
    struct_f = Struct('F', ('hit_e', struct_e))
    struct_g = Struct('G', ('hit_e', struct_e))
    struct_h = Struct('H', ('hit_f', struct_f), ('hit_g', struct_g))
    struct_i = Struct('I', ('hit_d', struct_d), ('hit_h', struct_h))    
    order = struct_i.get_ordered_dependencies()
    assert order == [
        # Structs without dependencies first
        struct_a, struct_e,
        # Then those depending on the already included
        struct_b, struct_c, struct_f, struct_g,
        # Then the same rule again...
        struct_d, struct_h,
        # ...and again.
        struct_i,
    ]


expected_local_glsl = """struct A {
  uint dud;
  vec3 dud2;
}"""


def test_local_glsl():
    struct_a = Struct('A', ('dud', GlType.uint), ('dud2', GlType.vec3))
    assert struct_a.get_glsl() == expected_local_glsl
    

expected_global_glsl = """struct A {
  uint dud;
}

struct B {
  A hit_a;
}
"""

def test_global_glsl():
    struct_a = Struct('A', ('dud', GlType.uint))
    struct_b = Struct('B', ('hit_a', struct_a))
    assert struct_b.get_full_glsl() == expected_global_glsl


expected_array_glsl = """struct A {
  uint dud[5];
}
"""


def test_glsl_with_array_one_dimensional():
    struct_a = Struct('A', ('dud', GlType.uint, 5))
    assert struct_a.get_full_glsl() == expected_array_glsl


expected_2d_array_glsl = """struct A {
  uint dud[2][3];
}
"""


def test_glsl_with_array_two_dimensional():
    struct_a = Struct('A', ('dud', GlType.uint, (2, 3)))
    assert struct_a.get_full_glsl() == expected_2d_array_glsl


expected_ssbo_glsl = """layout(std430) buffer A {
  uint dud[2][3];
}
"""


def test_ssbo():
    ssbo = SSBO('A', ('dud', GlType.uint, (2, 3)))
    assert ssbo.get_full_glsl() == expected_ssbo_glsl


def test_size_simple():
    ssbo = SSBO('A', ('dud', GlType.uint, (2, 3)))
    assert ssbo.get_byte_size() == 4*2*3


def test_size_complex():
    struct = Struct(
        'P',
        ('pos', GlType.vec3),  # 16 bytes (3*4 + padding)
        ('dir', GlType.vec3),
        ('idx', GlType.uint),  # 4 bytes
    )
    ssbo = SSBO('A', ('particle', struct, 65536))
    assert ssbo.get_byte_size() == 65536 * (16+16+4)


def test_to_bytes():
    ssbo = Struct(
        'P',
        ('pos', GlType.vec3),
        ('dir', GlType.vec3),
        ('idx', GlType.uint),
    )
    data = [(0,0,0),(0,0,0),0]
    assert ssbo.to_bytes(data) == b'\x00' * 36
    data = [(1,2,3),(4,5,6),7]
    assert ssbo.to_bytes(data) == b'\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x00\x00\x00\x00\x80@\x00\x00\xa0@\x00\x00\xc0@\x00\x00\x00\x00\x07\x00\x00\x00'
