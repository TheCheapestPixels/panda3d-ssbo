from array import array

from p3d_ssbo.gltypes import GlVec3


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
