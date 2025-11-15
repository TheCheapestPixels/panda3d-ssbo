from math import floor
from random import random

from panda3d.core import PStatClient
from panda3d.core import Vec3

from direct.showbase.ShowBase import ShowBase

from p3d_ssbo.gltypes import GlFloat
from p3d_ssbo.gltypes import Struct
from p3d_ssbo.gltypes import Buffer

from p3d_ssbo.tools.random_number_generator import PermutedCongruentialGenerator
from p3d_ssbo.tools.bitonic_sort import BitonicSort
from ssbo_card import SSBOCard


ShowBase()
base.cam.set_pos(0.5, -2.0, 0.5)
base.accept('escape', base.task_mgr.stop)
PStatClient.connect()


# The data
num_elements = 2**8
lines = 7.0
numbers = [floor((i / (num_elements - 1)) * lines) / lines for i in range(num_elements)]
# numbers = [float(i) / (num_elements - 1) for i in range(num_elements)]
initial_data = ([(n, ) for n in numbers], )
struct = Struct(
    'Data',
    GlFloat('value'),
)
data_buffer = Buffer(
    'dataBuffer',
    struct('data', num_elements),
    #initial_data=initial_data,
)


# Visualization
card = SSBOCard(base.render, data_buffer, ('data', 'value'))


# The compute shaders
rng = PermutedCongruentialGenerator(
    data_buffer,
    ('data', 'value'),
    # debug=True,
)
sorter = BitonicSort(
    data_buffer,
    ('data', 'value'),
    debug=True,
)


rng.attach(card.get_np())
base.task_mgr.add(rng.update)
sorter.attach(card.get_np())
#rng.dispatch()
#sorter.dispatch()


# Data extraction
#data = base.win.gsg.get_engine().extract_shader_buffer_data(
#    data_buffer.ssbo,
#    base.win.gsg,
#)
#py_data = data_buffer.unpack(data)
#for element in py_data[0]:
#    print(element[0])

base.run()
