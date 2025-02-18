import enum
from collections import defaultdict
from functools import reduce
from array import array

from jinja2 import Template

from panda3d.core import GeomEnums
from panda3d.core import ShaderBuffer


class GlType(enum.Enum):
    uint = 1
    vec2 = 2
    vec3 = 3
    vec4 = 4


glsl_types = {
    GlType.uint: 'uint',
    GlType.vec2: 'vec2',
    GlType.vec3: 'vec3',
    GlType.vec4: 'vec4',
}


byte_sizes = {
    GlType.uint: 4,
    GlType.vec2: 8,
    GlType.vec3: 16,
    GlType.vec4: 16,
}


def python_to_bytes_uint(value):
    v = array('I', [value]).tobytes()
    assert len(v) == 4
    return v


def python_to_bytes_vec2(value):
    v = array('f', value).tobytes()
    assert len(v) == 8
    return v


def python_to_bytes_vec3(value):
    v = array('f', value).tobytes()
    assert len(v) == 12
    return v + b'\x00\x00\x00\x00'


def python_to_bytes_vec4(value):
    v = array('f', value).tobytes()
    assert len(v) == 16
    return v


python_to_bytes = {
    GlType.uint: python_to_bytes_uint,
    GlType.vec3: python_to_bytes_vec3,
}


local_struct_glsl = """struct {{type_name}} {
{% for name, type, dimensions in fields %}  {{type}} {{name}}{% for d in dimensions %}[{{d}}]{% endfor %};
{% endfor %}}"""

local_ssbo_glsl = """layout(std430) buffer {{type_name}} {
{% for name, type, dimensions in fields %}  {{type}} {{name}}{% for d in dimensions %}[{{d}}]{% endfor %};
{% endfor %}}"""


global_glsl = """{% for struct in order %}{{struct.get_glsl()}}
{% if not loop.last %}
{% endif %}{% endfor %}"""


class Struct:
    local_glsl = local_struct_glsl

    def __init__(self, type_name, *fields):
        self.type_name = type_name
        self.fields = []
        for field in fields:
            if len(field) == 2:  # No array specified
                field = (field[0], field[1], [])
            elif isinstance(field[2], int):
                field = (field[0], field[1], [field[2]])
            self.fields.append(field)

    def get_dependencies(self, dependencies=None):
        if dependencies is None:
            dependencies = {}
        if self in dependencies:  # self has been processed already
            return
        else:
            dependencies[self] = []
        for field_name, field_type, _ in self.fields:
            if isinstance(field_type, Struct):
                if field_type not in dependencies[self]:
                    dependencies[self].append(field_type)
                    field_type.get_dependencies(dependencies)
        return dependencies

    def get_ordered_dependencies(self):
        to_process = self.get_dependencies()
        linearization = []
        while to_process:
            additions = []
            unadded = {}
            for field_type, deps in to_process.items():
                if all(dep in linearization for dep in deps):
                    additions.append(field_type)
                else:
                    unadded[field_type] = deps
            linearization += additions
            to_process = unadded
        return linearization

    def get_glsl(self):
        template = Template(self.local_glsl)
        glsl_fields = []
        for field_name, field_type, dimensions in self.fields:
            if field_type in glsl_types:  # Basic types
                glsl_type = glsl_types[field_type]
            else:  # Struct
                glsl_type = field_type.type_name
            glsl_fields.append((field_name, glsl_type, dimensions))
        source = template.render(
            type_name=self.type_name,
            fields=glsl_fields,
        )
        return source
    
    def get_full_glsl(self):
        order = self.get_ordered_dependencies()
        template = Template(global_glsl)
        source = template.render(order=order)
        return source

    def get_byte_size(self):
        size = 0
        for _, field_type, dimensions in self.fields:
            if field_type in byte_sizes:
                base_size = byte_sizes[field_type]
            else:
                base_size = field_type.get_byte_size()
            if dimensions:
                base_size *= reduce(lambda x, y: x*y, dimensions)
            size += base_size
        return size

    def to_bytes(self, data):
        assert len(data) == len(self.fields)
        data_bytes = b''
        for field_data, (_, field_type, dimensions) in zip(data, self.fields):
            if not dimensions:
                data_bytes += python_to_bytes[field_type](field_data)
        return data_bytes


class SSBO(Struct):
    local_glsl = local_ssbo_glsl

    def __init__(self, type_name, *fields, initial_data=None):
        Struct.__init__(self, type_name, *fields)
        if initial_data is None:
            size_or_data = self.get_byte_size()
        else:
            raise Exception  # Implement conversion of data to bytes
        self.ssbo = ShaderBuffer(
            type_name,
            size_or_data,
            GeomEnums.UH_static,
        )

    def get_buffer(self):
        return self.ssbo
