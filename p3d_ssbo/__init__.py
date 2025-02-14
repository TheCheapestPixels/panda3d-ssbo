import enum
from collections import defaultdict

from jinja2 import Template

class GlType(enum.Enum):
    uint = 1
    vec3 = 2


glsl_types = {
    GlType.uint: 'uint',
    GlType.vec3: 'vec3',
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
            glsl_type=glsl_type,
            fields=glsl_fields,
        )
        return source
    
    def get_full_glsl(self):
        order = self.get_ordered_dependencies()
        template = Template(global_glsl)
        source = template.render(order=order)
        return source


class SSBO(Struct):
    local_glsl = local_ssbo_glsl
