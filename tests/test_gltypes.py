from array import array

from p3d_ssbo.gltypes import GlFloat
from p3d_ssbo.gltypes import GlVec3
from p3d_ssbo.gltypes import Struct


def test_glsl_float():
    glsl = GlFloat('myFloat').glsl()
    assert glsl == 'float myFloat;'


def test_glsl_float_array():
    glsl = GlFloat('myFloat', 1, 2, 3).glsl()
    assert glsl == 'float myFloat[1][2][3];'


def test_size_float():
    assert GlFloat('myFloat').size() == 4
    

def test_size_float_array():
    assert GlFloat('myFloat', 2, 2, 2).size() == 32
    

def test_pack_float():
    data_buffer = GlFloat('myFloat').pack(17)
    assert data_buffer == array('f', [17]).tobytes()


def test_pack_float_array():
    data_buffer = GlFloat('myFloat', 5).pack([1,2,3,4,5])
    assert data_buffer == array('f', [1,2,3,4,5]).tobytes()


def test_unpack_float():
    byte_data = array('f', [17]).tobytes()
    py_data = GlFloat('myFloat').unpack(byte_data)
    assert py_data == 17


def test_unpack_float_array():
    byte_data = array('f', (1,2,3,4,5,6)).tobytes()
    py_data = GlFloat('myFloat', 2, 3).unpack(byte_data)
    assert py_data == ((1,2,3), (4,5,6))


###
###  Vec3
###


def test_glsl_vec3():
    glsl = GlVec3('myVec').glsl()
    assert glsl == 'vec3 myVec;'


def test_glsl_vec3_array():
    glsl = GlVec3('myVec', 1, 2, 3).glsl()
    assert glsl == 'vec3 myVec[1][2][3];'


def test_size_vec3():
    assert GlVec3('myVec').size() == 12
    

def test_size_vec3_array():
    expected_length = 8 * 16 - 4  # Last element has no trailing padding
    assert GlVec3('myVec', 2, 2, 2).size() == expected_length
    

def test_pack_vec3():
    data_buffer = GlVec3('myVec').pack((1,2,3))
    assert data_buffer == array('f', [1,2,3]).tobytes()


def test_pack_vec3_array():
    data_buffer = GlVec3('myVec', 2).pack(((1,2,3), (4,5,6)))
    assert data_buffer == array('f', [1,2,3,0,4,5,6]).tobytes()


def test_unpack_vec3():
    byte_data = array('f', [1,2,3]).tobytes()
    py_data = GlVec3('myVec').unpack(byte_data)
    assert py_data == (1,2,3)


def test_unpack_vec3_array():
    byte_data = array('f', (1,2,3,0,4,5,6)).tobytes()
    py_data = GlVec3('myVec', 2).unpack(byte_data)
    assert py_data == ((1,2,3), (4,5,6))


###
### Structs
###


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
