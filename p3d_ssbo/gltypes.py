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
# * `glsl`: Returns the element's GLSL representation. Called at the top
#   level element, this returns the whole definition that can be dropped
#   into shaders.
# * `size`: Wrapper for `_size`, returns length in bytes, to be called
#   at the top level. Syntactic candy, like `pack` and `unpack`.
#   * `_size`: (buffer size, trailing) -> (new size, new trailing)
# * `pack`: py_data -> byte_data.
#   * `_pack`: (py_data, byte_data, rest_dims, trailing) ->
#              (new byte_data, new trailing)
#     There are two possible cases:
#     1) This element is *not* an array. Pad to trailing alignment,
#        `_add_element`, return the updated buffer and the new trailing.
#     2) It *is* an array. Pad to trailing alignment, then call `_pack`
#        recursively (sidestepping the padding to trailing) until
#        individual elements are reached again.
#     * `_add_element`: (byte_data, py_data)->(new byte_data, trailing)
#       * pack_element`: Turn Python data to byte data
# * `unpack`: byte_data -> py_data
#   * `_unpack`: (byte_data, read_at, rest_dims, trailing) ->
#                (py_data, read_at, trailing)
#     The converse to `_pack`. Instead of removing data from the buffer,
#     though, it uses an index into the data (read_at). Otherwise, it
#     works exactly like `_pack`.
#     * `_read_element`: (byte_data, read_at)->(py_data, trailing)
#       * `unpack_element`: Return Python data at read_at
#
# Structs are a bit special, FIXME:
# * There's Type and there's Instance
# * They overwrite `_add_element` and `_read_element`.
#
# Often used variable names
# * buffer_so_far: Binary buffer content.
# * size_so_far: Length of the binary buffer.
# * trailing: Alignment that the following element has (at the least) to
#   align to.


from array import array
import math

from panda3d.core import LVecBase2f
from panda3d.core import LVecBase3f
from panda3d.core import ShaderBuffer
from panda3d.core import GeomEnums


class GlType:
    def __init__(self, field_name, *dims, unbounded=False):
        self.field_name = field_name
        self.dims = dims
        self.unbounded = unbounded

    def glsl(self):
        """
        Return the GLSL string defining the element.
        """
        if self.unbounded:
            dim_string = ''.join(f'[{d}]' for d in self.dims[:-1]) + '[]'
        else:
            dim_string = ''.join(f'[{d}]' for d in self.dims)
        return self.glsl_type_name  + ' ' + self.field_name + dim_string + ';'

    def _calculate_offset(self, size_so_far=0, trailing=0):
        """
        Given the size of the buffer so far, and the offset that the
        previous element requires, how much padding does the buffer
        require to be properly aligned for this element?
        """
        alignment = max(self.alignment, trailing)
        pad_length = 0
        if size_so_far % alignment != 0:
            pad_length = alignment - size_so_far % alignment
        return pad_length
        
    def size(self):
        "Returns this element's data size, without leading of trailing offsets."
        size, trailing = self._size()
        size += self._calculate_offset(size, trailing)

        return size * 4

    def _size(self, size_so_far=0, trailing=0):
        # Align
        pad_length = self._calculate_offset(size_so_far, trailing)
        size_so_far += pad_length
        # Add element(s)
        stride = self.element_size + self._calculate_offset(self.element_size)
        array_factor = math.prod(self.dims)
        size_so_far += stride * (array_factor - 1) + self.element_size
        # FIXME: If this is an array or struct, the next element may
        # require padding.
        trailing = 0
        if self.dims != () or isinstance(self, StructInstance):
            trailing = self.alignment
        return size_so_far, trailing

    def pack(self, py_data):
        "Turn this Python data structure into buffer byte data."
        byte_data, trailing = self._pack(
            py_data,
            b''
        )
        byte_data += b'\x00' * self._calculate_offset(len(byte_data), trailing)
        return byte_data

    def _pack(self, py_data, byte_data, rest_dims=None, trailing=0):
        if rest_dims is None:
            rest_dims = self.dims
            byte_data = self._pad_to_alignment(byte_data, trailing)
        # Add data
        if rest_dims == ():  # Individual element
            byte_data = self._pad_to_alignment(byte_data)
            byte_data, next_trailing = self._add_element(byte_data, py_data)
        else:  # Array
            assert len(py_data) == self.dims[0], f"{len(py_data)} elements given for a size {self.dims[0]} array."
            for idx, py_data_piece in enumerate(py_data):
                byte_data, trailing = self._pack(
                    py_data_piece,
                    byte_data,
                    rest_dims[1:],
                )
            next_trailing = self.alignment
        return byte_data, next_trailing

    def _pad_to_alignment(self, byte_data, trailing=0):
        assert len(byte_data) % 4 == 0
        pad_length = self._calculate_offset(len(byte_data) // 4, trailing)
        byte_data += b'\x00' * pad_length * 4
        return byte_data

    def _add_element(self, byte_data, py_data):
        byte_data += self.pack_element(py_data)
        trailing = 0
        return byte_data, trailing

    def unpack(self, byte_data):
        py_data, read_at, trailing = self._unpack(byte_data)
        return py_data

    def _unpack(self, byte_data, read_at=0, rest_dims=None, trailing=0):
        # Align to trailing offset
        if rest_dims is None:
            rest_dims = self.dims
            read_at += self._calculate_offset(read_at, trailing)
        # Unpack_data
        if rest_dims == ():
            read_at += self._calculate_offset(read_at)
            py_data, trailing = self._read_element(byte_data, read_at)
            read_at += self.element_size
        else:
            stride = self.element_size * math.prod(rest_dims[1:])
            py_data = tuple(
                self._unpack(
                    byte_data,
                    read_at + index * stride,
                    rest_dims[1:],
                )[0]
                for index in range(rest_dims[0])
            )
            read_at += stride * rest_dims[0]
            trailing = self.alignment
        return py_data, read_at, trailing

    def _read_element(self, byte_data, read_at):
        py_data = self.unpack_element(byte_data, read_at)
        trailing = 0  # FIXME?
        return py_data, trailing


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


class GlUInt(GlType):
    glsl_type_name = 'uint'
    alignment = 1
    element_size = 1

    def pack_element(self, py_data):
        assert isinstance(py_data, int)
        return array('I', [py_data]).tobytes()

    def unpack_element(self, byte_data, read_at):
        start = read_at * 4
        end = (read_at + self.element_size) * 4
        element_byte_data = byte_data[start:end]
        a = array('I')
        a.frombytes(element_byte_data)
        py_data = a.tolist()[0]
        return py_data


class GlVec2(GlType):
    glsl_type_name = 'vec2'
    alignment = 4
    element_size = 2

    def pack_element(self, py_data):
        if isinstance(py_data, LVecBase2f):
            x, y = py_data.x, py_data.y
        else:
            assert len(py_data) == 2
            assert all(isinstance(e, (int, float)) for e in py_data)
            x, y = py_data
        byte_data = array('f', [x,y]).tobytes()
        return byte_data

    def unpack_element(self, byte_data, read_at):
        start = read_at * 4
        end = (read_at + 2) * 4
        element_byte_data = byte_data[start:end]
        a = array('f')
        a.frombytes(element_byte_data)
        py_data = tuple(a.tolist())
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
        self.fields = fields
        self.field_by_name = {f.field_name: f for f in fields}
        self.glsl_type_name = type_name
        self.alignment = max(f.alignment for f in fields)

    def __call__(self, field_name, *dims, unbounded=False):
        instance = StructInstance(self, field_name, dims, unbounded=unbounded)
        return instance

    def glsl(self):
        text = f"struct {self.glsl_type_name} {{\n"
        for field in self.fields:
            text += f"  {field.glsl()}\n"
        text += "};"
        return text

    def _get_struct_types(self, types=None):
        if types is None:
            types = []
        if self not in types:
            structs = [
                f for f in self.fields
                if isinstance(f, StructInstance)
            ]
            for field in structs:
                types = field._get_struct_types(types)
            types.append(self)
        return types


class StructInstance(GlType):
    def __init__(self, type_obj, field_name, dims, unbounded=False):
        self.type_obj = type_obj
        self.glsl_type_name = type_obj.glsl_type_name
        self.field_name = field_name
        self.alignment = self.type_obj.alignment
        self.dims = dims
        size = 0
        trailing = 0
        for field in self.type_obj.fields:
            size, trailing = field._size(size, trailing)
        self.element_size = size
        self.unbounded = unbounded

    def glsl(self):
        if self.unbounded:
            dim_string = ''.join(f'[{d}]' for d in self.dims[:-1]) + '[]'
        else:
            dim_string = ''.join(f'[{d}]' for d in self.dims)
        text = f"{self.type_obj.glsl_type_name} {self.field_name}{dim_string};"
        return text

    def _add_element(self, byte_data, py_data):
        for field, data in zip(self.type_obj.fields, py_data):
            byte_data, trailing = field._pack(data, byte_data)
        return byte_data, self.alignment

    def _read_element(self, byte_data, read_at):
        struct_py_data = []
        for field in self.type_obj.fields:          
            py_data, read_at, trailing = field._unpack(byte_data, read_at)
            struct_py_data.append(py_data)
        return tuple(struct_py_data), self.alignment

    def _get_struct_types(self, types=None):
        if types is None:
            types = []
        if self.type_obj not in types:
            types = self.type_obj._get_struct_types(types)
        return types

    def get_num_elements(self):
        return self.dims

    def get_field(self, field_name):
        field = self.type_obj.field_by_name[field_name]
        return field


class Buffer(GlType):
    dims = ()

    def __init__(self, type_name: str, *fields: GlType, initial_data=None, bind_buffer=None, num_elements=0):
        self.fields = fields
        self.field_by_name = {f.field_name: f for f in fields}
        self.glsl_type_name = type_name
        self.alignment = max(f.alignment for f in fields)
        size = 0
        trailing = 0
        for field in self.fields:
            size, trailing = field._size(size, trailing)
        self.element_size = size
        if bind_buffer is not None:
            assert type(bind_buffer) is ShaderBuffer, f'Only ShaderBuffers can be bound to p3d_ssbo.gltypes.Buffer!'
            size = bind_buffer.data_size_bytes
            assert size % self.element_size == 0, f'buffer bound to p3d_ssbo.gltypes.Buffer is not a multiple of {self.element_size} long!'
            self.ssbo = bind_buffer
        else:
            if initial_data is None:
                size_or_data = size
            else:
                size_or_data = self.pack(initial_data)
            self.ssbo = ShaderBuffer(
                self.glsl_type_name,
                size_or_data,
                GeomEnums.UH_static
            )

    def glsl(self):
        # generate glsl for declaration of ssbo
        field = self.fields[0]
        text = (f"layout(std430, binding = 0) buffer {self.glsl_type_name} " "{ ") # f"{field.glsl()}"[:-1] "[] " "}")
        text += f"{field.glsl_type_name} {field.field_name}[];"
        text += " };"
        return text

    def _add_element(self, byte_data, py_data):
        for field, data in zip(self.fields, py_data):
            byte_data, trailing = field._pack(data, byte_data)
        return byte_data, self.alignment

    def _read_element(self, byte_data, read_at):
        struct_py_data = []
        for field in self.fields:
            raa = read_at
            py_data, read_at, trailing = field._unpack(byte_data, read_at)
            print(raa, read_at, trailing)
            struct_py_data.append(py_data)
        return tuple(struct_py_data), self.alignment

    def _get_struct_types(self, types=None):
        if types is None:
            types = []
        structs = [
            f
            for f in self.fields
            if isinstance(f, StructInstance)
        ]
        for field in structs:
            types = field._get_struct_types(types)
        return types

    def _get_buffers(self):
        return [self]

    def full_glsl(self):
        # generate glsl for declaration of ssbo and structs contained in it
        structs = self._get_struct_types()
        struct_glsl = '\n\n'.join([s.glsl() for s in structs])
        buffer_glsl = self.glsl()
        glsl = '\n\n'.join([struct_glsl, buffer_glsl])
        return glsl

    def get_field(self, field_name):
        field = self.field_by_name[field_name]
        return field


class BufferSet:
    def __init__(self, *buffers):
        self.buffers = buffers

    def glsl(self):
        glsl_types = ''.join(t for typ in self._get_types())
        glsl_buffers = ''.join(buf.glsl() for buf in self.buffers)
        glsl = ''.join([glsl_types, glsl_buffers])
        return glsl

    def _get_struct_types(self, types=None):
        if types is None:
            types = []
        for buf in self.buffers:
            types = buf._get_struct_types(types)
        return types

    def _get_buffers(self):
        return self.buffers

    def full_glsl(self):
        structs = self._get_struct_types()
        buffers = self._get_buffers()
        struct_glsl = '\n\n'.join([s.glsl() for s in structs])
        buffer_glsl = '\n\n'.join([b.glsl() for b in buffers])
        glsl = '\n\n'.join([struct_glsl, buffer_glsl])
        return glsl
