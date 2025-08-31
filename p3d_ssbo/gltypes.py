# SSBO buffer content is laid out according to the rules in:
#   https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf
#   7.6.2.2 Standard Uniform Block Layout
#
# There are two versions, std140 and std430. The latter relaxes rules of
# the former in two places. Therefore this summary will assume std430 as
# the default, and will have notes on std140 where those rules need to
# be constricted.
#
# These rules are for the most part about aligning data that is to be
# written. This means that the data has to start at a certain buffer
# position, usually a multiple of the data's length, so padding may have
# to be added to the buffer to reach the appropriate length.
#
#   1. Scalars align to their length.
#   2. nvec2 and nvec4 align to 2/4 times their scalar length.
#   3. nvec3 aligns to 4 times its scalar length.
#   4. Arrays of scalars or vectors
#      * align to their elements' length
#        * std140: round up to vec4's alignment
#      * have an array stride of the same
#      * have, if another element follows after the array, padding to
#        the element's length
# 5-8. Matrices are stored as arrays.
#      * std140 applies accordingly.
#   9. Structures
#      * use the maximum of its members' alignment as its own alignment
#        * std140: Round up to vec4's alignment
#      * Individual members align to their own alignment / trailing
#        offset
#      * Padding after array elements still applies
#      * may be padded at the end like arrays
#  10. Arrays of structures use the struct's alignment.
#
# The workhorse of this library is the class `GlType`, on which
# everything else is built.
#
# * `size`
#   * `_size`
# * `glsl`
# * `pack`: User-facing API that basically just calls...
#   * `_pack`: Return updated buffer and trailing padding by...
#     * `_pad_to_alignment`: Align to beginning of this/these elemet(s).
#     * `_add_element`: Return buffer with added data from...
#       * pack_element`: Turn Python data to byte data







from array import array
import math

from panda3d.core import LVecBase3f


class GlType:
    def __init__(self, field_name, *dims):
        self.field_name = field_name
        self.dims = dims

    def size(self):
        size, trailing_offset = self._size()
        return size * 4

    def _size(self, size_so_far=0, trailing_offset=0):
        # Align
        pad_length = self._calculate_offset(size_so_far, trailing_offset)
        size_so_far += pad_length
        # Add element(s)
        stride = self.element_size + self._calculate_offset(self.element_size)
        array_factor = math.prod(self.dims)
        size_so_far += stride * (array_factor - 1) + self.element_size
        # If this is an array or struct, the next element may require
        # padding.
        trailing_offset = 0
        if self.dims != () or isinstance(self, StructInstance):
            trailing_offset = self.alignment
        return size_so_far, trailing_offset

    def _calculate_offset(self, size_so_far=0, trailing_offset=0):
        alignment = max(self.alignment, trailing_offset)
        pad_length = 0
        if size_so_far % alignment != 0:
            pad_length = alignment - size_so_far % alignment
        return pad_length
        
    def glsl(self):
        dim_string = ''.join(f'[{d}]' for d in self.dims)
        return self.glsl_type_name  + ' ' + self.field_name + dim_string + ';'

    def pack(self, py_data, buffer_so_far=None, rest_dims=None):
        buffer_data, trailing_offset = self._pack(
            py_data,
            b''
        )
        return buffer_data

    def _pack(self, py_data, buffer_so_far, rest_dims=None, trailing_offset=0):
        if rest_dims is None:
            rest_dims = self.dims
            buffer_so_far = self._pad_to_alignment(buffer_so_far, trailing_offset)
        # Add data
        if rest_dims == ():  # Individual element
            buffer_so_far = self._pad_to_alignment(buffer_so_far)
            buffer_so_far, next_trailing_offset = self._add_element(buffer_so_far, py_data)
        else:  # Array
            assert len(py_data) == self.dims[0], f"{len(py_data)} elements given for a size {self.dims[0]} array."
            for idx, py_data_piece in enumerate(py_data):
                buffer_so_far, trailing_offset = self._pack(
                    py_data_piece,
                    buffer_so_far,
                    rest_dims[1:],
                )
            next_trailing_offset = self.alignment
        return buffer_so_far, next_trailing_offset

    def _pad_to_alignment(self, buffer_so_far, trailing_offset=0):
        assert len(buffer_so_far) % 4 == 0
        pad_length = self._calculate_offset(len(buffer_so_far) // 4, trailing_offset)
        buffer_so_far += b'\x00' * pad_length * 4
        return buffer_so_far

    def _add_element(self, buffer_so_far, py_data):
        buffer_so_far += self.pack_element(py_data)
        trailing_offset = 0
        return buffer_so_far, trailing_offset

    def unpack(self, byte_data, read_at=0, rest_dims=None):
        # FIXME: Trailing offset
        if rest_dims is None:
            rest_dims = self.dims
        # Align
        if read_at % self.alignment != 0:
            offset = self.alignment - read_at % self.alignment
            read_at += offset
        # Unpack_data
        if rest_dims == ():
            py_data = self.unpack_element(byte_data, read_at)
        else:
            stride = self.element_size * math.prod(rest_dims[1:])
            py_data = tuple(
                self.unpack(
                    byte_data,
                    read_at + index * stride,
                    rest_dims[1:],
                )
                for index in range(rest_dims[0])
            )
        return py_data


class GlFloat(GlType):
    glsl_type_name = 'float'
    alignment = 1
    element_size = 1

    def pack_element(self, py_data):
        assert isinstance(py_data, (int, float))
        return array('f', [py_data]).tobytes()

    def unpack_element(self, byte_data, read_at):
        start = read_at * 4
        end = (read_at + self.element_size) * 4
        element_byte_data = byte_data[start:end]
        a = array('f')
        a.frombytes(element_byte_data)
        py_data = a.tolist()[0]
        return py_data


class GlVec3(GlType):
    glsl_type_name = 'vec3'
    alignment = 4
    element_size = 3

    def pack_element(self, py_data):
        if isinstance(py_data, LVecBase3f):
            x,y,z = py_data.x, py_data.y, py_data.z
        else:
            assert len(py_data) == 3
            assert all(isinstance(e, (int, float)) for e in py_data)
            x,y,z = py_data
        byte_data = array('f', [x,y,z]).tobytes()
        return byte_data

    def unpack_element(self, byte_data, read_at):
        start = read_at * 4
        end = (read_at + 3) * 4
        element_byte_data = byte_data[start:end]
        a = array('f')
        a.frombytes(element_byte_data)
        py_data = tuple(a.tolist())
        return py_data


class Struct:
    def __init__(self, type_name, *fields):
        self.type_name = type_name
        self.fields = fields
        self.glsl_type_name = type_name
        self.alignment = max(f.alignment for f in fields)

    def __call__(self, field_name, *dims):
        instance = StructInstance(self, field_name, dims)
        return instance

    def glsl(self):
        text = f"struct {self.type_name} {{\n"
        for field in self.fields:
            text += f"  {field.glsl()}\n"
        text += "}"
        return text


class StructInstance(GlType):
    def __init__(self, type_obj, field_name, dims):
        self.type_obj = type_obj
        self.field_name = field_name
        self.alignment = self.type_obj.alignment
        self.dims = dims
        size = 0
        trailing = 0
        for field in self.type_obj.fields:
            size, trailing = field._size(size, trailing)
        self.element_size = size

    def glsl(self):
        dim_string = ''.join(f'[{d}]' for d in self.dims)
        text = f"{self.type_obj.type_name} {self.field_name}{dim_string};"
        return text

    def _add_element(self, buffer_so_far, py_data):
        for field, data in zip(self.type_obj.fields, py_data):
            buffer_so_far, trailing_offset = field._pack(data, buffer_so_far)
        return buffer_so_far, self.alignment

    def unpack_element(self, byte_data, read_at):
        pass
