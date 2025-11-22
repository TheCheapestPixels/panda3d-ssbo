# Let's plug together a simple application. We will...
# * create a bunch of numbers between 0 and 1
# * stuff them into an SSBO
# * replace the data with new random values
# * sort the SSBO
# * display the values on a quad
#
# Let's look past the part where doing all of that makes little sense
# for a second; Part of the point of `panda3d-ssbo` is to keep your code
# compact and easy to work with, so commenting out bits of it is easy,
# too.


from math import floor
from math import sin
from random import random

from panda3d.core import PStatClient
from panda3d.core import Vec3

from direct.showbase.ShowBase import ShowBase

from p3d_ssbo.gltypes import GlFloat
from p3d_ssbo.gltypes import Struct
from p3d_ssbo.gltypes import Buffer
from p3d_ssbo.algos.random_number_generator import PermutedCongruentialGenerator
from p3d_ssbo.algos.bitonic_sort import BitonicSort
from p3d_ssbo.tools.ssbo_card import SSBOCard


# Let's set up a basic Panda3D application first.
ShowBase()
base.cam.set_pos(0.5, -2.0, 0.5)
base.accept('escape', base.task_mgr.stop)
PStatClient.connect()
base.set_frame_rate_meter(True)


# How many numbers do we want to deal with right now?
# I've managed to use 2**20 (a million) numbers and beyond, but there is
# a weird pause after caling `base.run()` when using those higher
# numbers. Debugging is needed.
# The minima are 2**5 for the rng, and 2**6 for the sorter.
num_elements = 2**6
print(f"Configured for {num_elements} elements.")


# We *can* stuff initial data into the SSBO. We don't have to. If we do
# want to, well, we need some numbers. So how about a nice sine wave?
# If you have pushed num_elements high enough, this may take a while
# though.
def make_data():
    def sine_data(i):
        ratio = float(i) / float(num_elements - 1)
        value = (sin(ratio * 2 * 3.141592) + 1) / 2
        return value
    print("Creating initial data...")
    struct_data = [(sine_data(i), ) for i in range(num_elements)]
    buffer_data = (struct_data, )
    print("Data created.")
    return buffer_data


# Let's define an SSBO!
struct = Struct(
    'Data',
    GlFloat('value'),
)
data_buffer = Buffer(
    'dataBuffer',
    struct('data', num_elements),
    # Remove out the next line, and the buffer will contain no initial
    # data. How surprising.
    #initial_data=make_data(),
)
print("SSBO created.")


# Visualization
card = SSBOCard(base.render, data_buffer, ('data', 'value'))


# The compute shaders. The `debug` argument makes them print their GLSL
# code.
rng = PermutedCongruentialGenerator(
    data_buffer,
    ('data', 'value'),
    # debug=True,
)
sorter = BitonicSort(
    data_buffer,
    ('data', 'value'),
    # debug=True,
)


# Now comes the fun bit.
# We can dispatch or attach our shaders. If we dispatch them, they will
# run as soon as possible, and only once.
#rng.dispatch()
#sorter.dispatch()
# Alternatively we can attach them, specifically our card's NodePath,
# and they will run whenever the card is not immediately culled away.
# In the case of the random number generator we will probably have it
# use a different RNG seed each frame, so we can let it create a task,
# for which we pass the arguments (other than the function that the task
# will call) to indicate that we want it created.
rng.attach(card.get_np(), task=((), {}))
sorter.attach(card.get_np())


# Data extraction
#data = base.win.gsg.get_engine().extract_shader_buffer_data(
#    data_buffer.ssbo,
#    base.win.gsg,
#)
#py_data = data_buffer.unpack(data)
#for element in py_data[0]:
#    print(element[0])


print("Starting application...")
base.run()
