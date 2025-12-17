from array import array

from p3d_ssbo.gltypes import GlUInt


def test_glsl_uint():
    glsl = GlUInt('myUInt').glsl()
    assert glsl == 'uint myUInt;'


def test_glsl_uint_array():
    glsl = GlUInt('myUInt', 1, 2, 3).glsl()
    assert glsl == 'uint myUInt[1][2][3];'


def test_size_uint():
    assert GlUInt('myUInt').size() == 4
    

def test_size_uint_array():
    assert GlUInt('myUInt', 2, 2, 2).size() == 32
    

def test_pack_uint():
    data_buffer = GlUInt('myUInt').pack(17)
    assert data_buffer == array('I', [17]).tobytes()


def test_pack_uint_array():
    data_buffer = GlUInt('myUInt', 5).pack([1,2,3,4,5])
    assert data_buffer == array('I', [1,2,3,4,5]).tobytes()


def test_unpack_uint():
    byte_data = array('I', [17]).tobytes()
    py_data = GlUInt('myUInt').unpack(byte_data)
    assert py_data == 17


def test_unpack_uint_array():
    byte_data = array('I', (1,2,3,4,5,6)).tobytes()
    py_data = GlUInt('myUInt', 2, 3).unpack(byte_data)
    assert py_data == ((1,2,3), (4,5,6))
