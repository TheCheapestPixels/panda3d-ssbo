from array import array
import random

from p3d_ssbo.gltypes import GlFloat
from p3d_ssbo.gltypes import GlVec3

from p3d_ssbo.gltypes import Struct
from p3d_ssbo.gltypes import Buffer
from p3d_ssbo.gltypes import BufferSet


def test_glsl_struct_type():
    my_struct_type = Struct(
        'MyStruct',
        GlVec3('foo'),
        GlFloat('bar'),
    )
    glsl = my_struct_type.glsl()
    assert glsl == 'struct MyStruct {\n  vec3 foo;\n  float bar;\n}'


def test_glsl_struct_instance():
    my_struct_type = Struct(
        'MyStruct',
        GlVec3('foo'),
        GlFloat('bar'),
    )
    my_struct = my_struct_type('fnord')
    glsl = my_struct.glsl()
    assert glsl == 'MyStruct fnord;'


def test_glsl_struct_instance_array():
    my_struct_type = Struct(
        'MyStruct',
        GlFloat('bar'),
    )
    my_struct = my_struct_type('fnord', 2)
    glsl = my_struct.glsl()
    assert glsl == 'MyStruct fnord[2];'


def test_size_struct():
    my_struct_type = Struct(
        'MyStruct',
        GlVec3('foo'),
        GlVec3('bar'),
        GlFloat('baz'),
    )
    my_struct = my_struct_type('fnord')
    assert my_struct.size() == 8 * 4

    my_struct_type = Struct(
        'MyStruct',
        GlVec3('foo', 2),
        GlFloat('baz'),
    )
    my_struct = my_struct_type('fnord')
    assert my_struct.size() == 9 * 4


def test_size_struct_array():
    my_struct_type = Struct(
        'MyStruct',
        GlFloat('foo'),
        GlVec3('bar'),
    )
    my_struct = my_struct_type('fnord', 2)
    assert my_struct.size() == (4+4+4+3) * 4


def test_pack_struct():
    my_struct_type = Struct(
        'MyStruct',
        GlFloat('foo'),
        GlVec3('bar'),
    )
    my_struct = my_struct_type('fnord')
    data_buffer = my_struct.pack(
        (1.5, (0.2, 0.4, 0.6))
    )
    expected_data = array('f', [1.5]).tobytes()
    expected_data += b'\x00' * 12
    expected_data += array('f', [0.2, 0.4, 0.6]).tobytes()
    assert data_buffer == expected_data


def test_pack_struct_array():
    my_struct_type = Struct(
        'MyStruct',
        GlFloat('foo'),
        GlVec3('bar'),
    )
    my_struct = my_struct_type('fnord', 2)
    data_buffer = my_struct.pack(
        (
            (1.5, (0.2, 0.4, 0.6)),
            (0.5, (0.1, 0.3, 0.5)),
        )
    )
    expected_data = array('f', [1.5]).tobytes()
    expected_data += b'\x00' * 12
    expected_data += array('f', [0.2, 0.4, 0.6]).tobytes()
    expected_data += b'\x00' * 4  # Post-struct padding

    expected_data += array('f', [0.5]).tobytes()
    expected_data += b'\x00' * 12
    expected_data += array('f', [0.1, 0.3, 0.5]).tobytes()

    assert data_buffer == expected_data


def test_unpack_struct():
    my_struct_type = Struct(
        'MyStruct',
        GlFloat('foo'),
        GlVec3('bar'),
    )
    my_struct = my_struct_type('fnord')
    data_buffer = array('f', [1.5]).tobytes()
    data_buffer += b'\x00' * 12
    data_buffer += array('f', [0.25, 0.5, 0.75]).tobytes()
    decoded_data = my_struct.unpack(data_buffer)
    expected_data = (1.5, (0.25, 0.5, 0.75))
    assert decoded_data == expected_data


def test_unpack_struct_array():
    my_struct_type = Struct(
        'MyStruct',
        GlFloat('foo'),
        GlVec3('bar'),
    )
    my_struct = my_struct_type('fnord', 2)
    data_buffer = array('f', [1.5]).tobytes()
    data_buffer += b'\x00' * 12
    data_buffer += array('f', [0.25, 0.5, 0.75]).tobytes()
    data_buffer += b'\x00' * 4
    data_buffer += array('f', [3.0]).tobytes()
    data_buffer += b'\x00' * 12
    data_buffer += array('f', [0.125, 0.625, 0.875]).tobytes()
    decoded_data = my_struct.unpack(data_buffer)
    expected_data = (
        (1.5, (0.25, 0.5, 0.75)),
        (3.0, (0.125, 0.625, 0.875)),
    )
    assert decoded_data == expected_data


def test_get_struct_types_single():
    my_struct_type = Struct(
        'MyStruct',
        GlFloat('foo'),
    )
    types = my_struct_type._get_struct_types()
    assert types == [my_struct_type]

    my_struct = my_struct_type('fnord')
    types = my_struct._get_struct_types()
    assert types == [my_struct_type]


def test_get_struct_types_linear():
    type_a = Struct(
        'MyStructA',
        GlFloat('foo'),
    )
    type_b = Struct(
        'MyStructB',
        type_a('foo'),
    )

    types = type_b._get_struct_types()
    assert types == [type_a, type_b]

    my_struct = type_b('fnord')
    types = my_struct._get_struct_types()
    assert types == [type_a, type_b]


def test_get_struct_types_tree():
    type_a = Struct(
        'MyStructA',
        GlFloat('foo'),
    )
    type_b = Struct(
        'MyStructB',
        GlFloat('foo'),
    )
    type_c = Struct(
        'MyStructC',
        type_a('foo'),
        type_b('bar'),
    )

    types = type_c._get_struct_types()
    assert types == [type_a, type_b, type_c]

    my_struct = type_c('fnord')
    types = my_struct._get_struct_types()
    assert types == [type_a, type_b, type_c]


def test_get_struct_types_diamond():
    type_a = Struct(
        'MyStructA',
        GlFloat('foo'),
    )
    type_b = Struct(
        'MyStructB',
        type_a('foo'),
    )
    type_c = Struct(
        'MyStructC',
        type_a('foo'),
    )
    type_d = Struct(
        'MyStructD',
        type_b('foo'),
        type_c('bar'),
    )

    types = type_d._get_struct_types()
    assert types == [type_a, type_b, type_c, type_d]

    my_struct = type_d('fnord')
    types = my_struct._get_struct_types()
    assert types == [type_a, type_b, type_c, type_d]


def test_get_struct_types_buffer_single():
    my_struct_type = Struct(
        'MyStruct',
        GlFloat('foo'),
    )
    my_buffer = Buffer(
        'MyBuffer',
        my_struct_type('foo'),
    )
    
    types = my_buffer._get_struct_types()
    assert types == [my_struct_type]


def test_get_struct_types_buffer_set():
    my_struct_type = Struct(
        'MyStruct',
        GlFloat('foo'),
    )
    my_buffer_a = Buffer(
        'MyBufferA',
        my_struct_type('foo'),
    )
    my_buffer_b = Buffer(
        'MyBufferB',
        my_struct_type('foo'),
    )
    my_buffer_set = BufferSet(my_buffer_a, my_buffer_b)
    
    types = my_buffer_set._get_struct_types()
    assert types == [my_struct_type]



recursive_buffer_glsl = """
struct MyBasicStruct {
  float foo;
}

struct LeftDiamond {
  MyBasicStruct foo;
}

struct RightDiamond {
  MyBasicStruct foo;
}

struct Capstone {
  LeftDiamond foo;
  RightDiamond bar;
}

layout(std430) buffer MyBuffer {
  Capstone cap;
}"""[1:]


def test_glsl_buffer():
    basic_type = Struct(
        'MyBasicStruct',
        GlFloat('foo'),
    )
    left_diamond_type = Struct(
        'LeftDiamond',
        basic_type('foo'),
    )
    right_diamond_type = Struct(
        'RightDiamond',
        basic_type('foo'),
    )
    capstone_type = Struct(
        'Capstone',
        left_diamond_type('foo'),
        right_diamond_type('bar'),
    )
    my_buffer = Buffer(
        'MyBuffer',
        capstone_type('cap')
    )

    glsl = my_buffer.full_glsl()
    assert glsl == recursive_buffer_glsl

    my_buffer_set = BufferSet(my_buffer)
    glsl = my_buffer_set.full_glsl()
    assert glsl == recursive_buffer_glsl


double_buffer_glsl = """
struct BasicType {
  float foo;
}

layout(std430) buffer MyBufferA {
  BasicType foo;
}

layout(std430) buffer MyBufferB {
  BasicType foo;
}"""[1:]


def test_glsl_buffer_set():
    basic_type = Struct(
        'BasicType',
        GlFloat('foo'),
    )
    my_buffer_a = Buffer(
        'MyBufferA',
        basic_type('foo'),
    )
    my_buffer_b = Buffer(
        'MyBufferB',
        basic_type('foo'),
    )
    my_buffer_set = BufferSet(my_buffer_a, my_buffer_b)
    glsl = my_buffer_set.full_glsl()
    assert glsl == double_buffer_glsl


def test_buffer_ssbo():
    basic_type = Struct(
        'BasicType',
        GlFloat('foo'),
    )
    my_buffer = Buffer(
        'MyBuffer',
        basic_type('foo'),
    )
    initial_data = ((7,),)
    shader_buffer = my_buffer.ssbo(initial_data)


def test_buffer_ssbo_2():
    my_buffer = Buffer(
        'MyBuffer',
        GlFloat('foo', 200),
    )

    shader_buffer = my_buffer.ssbo()
    assert shader_buffer.data_size_bytes == 800

    initial_data = tuple(
        [
            tuple(
                [
                    random.random()
                    for _ in range(200)
                ]
            )
        ]
    )
    shader_buffer = my_buffer.ssbo(initial_data)
    assert shader_buffer.data_size_bytes == 800
