from array import array

from p3d_ssbo.gltypes import GlFloat


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
