# panda3d-ssbo
A little module to make working with SSBOs in Panda3D a bit easier.

```python
from p3d_ssbo import GlFloat, GlVec3, Struct


struct = Struct(
    "Test",  # Type name of the struct
    ("foo", GlFloat()),
    ("bar", GlVec3()),
)
glsl = struct.glsl()
size = struct.size()


values_in = 0.0, (0,0,0)
data_buffer_content = struct.pack(values_in)
values_out = struct.unpack(data_buffer_content)



```