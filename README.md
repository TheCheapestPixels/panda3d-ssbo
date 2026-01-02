# panda3d-ssbo

A little module to make working with SSBOs in Panda3D a bit easier.
Specifically, it offers:

* An SSBO mapper
* An algorithm library
* Tools

However, it is also in an early stage of development, though what is
there already does seem to work.


## SSBO Mapper

SSBOs are structured binary data on the GPU, consisting of the basic
numeric types of OpenGL, organized in arrays and structs; On the Panda3D
side, they are represented by `panda3d.core.ShaderBuffer`, into which
binary data (or a buffer size for an uninitialized buffer) can be fed,
and from which it can be extracted again.

One little problem with that is that the format of the data is spelled
out in section 7.6.2.2 in the OpenGL 4.60 standard, and they are... not
exactly trivial to deal with. Additionally you need to write out their
structure in GLSL, while juggling the data on the Python side. This
module deals with that by providing a mapper that lets you define (and
introspect) the SSBO's structure in Python, automatically calculate its
size, feed Python data structures into it to get binary data, and also
turn binary data back into Python data.

```python
from p3d_ssbo.gltypes import GlFloat, Buffer


values = [i for i in range(65536)]
buffer_data = [struct_data]

data_buffer = Buffer(
    'dataBuffer',
    GlFloat('value', 65536),
    initial_data=buffer_data,
)
shader_buffer_object = data_buffer.ssbo
glsl_code = data_buffer.full_glsl()
value_array = data_buffer.get_field('value')
num_elements = value_array.get_num_elements()
```

CAVEATS
* Right now `uint`, `float` and `vec3` are supported; That's it.
* I do not truly trust the code yet, despite all the green tests...


## Algorithms

With the hardships of managing your data and its structure squared away,
it is time to work on your actual algorithms. The thing about
algorithms, though, is that many are mostly cobbled together from more
fundamental algorithms. So, where is the standard library?

Scattered all over the internet as snippets of code that people
copypaste around. Engines like Unreal, Unity, and Godot of course have
their own collection of ready-to-use shader snippets, and ways to
compose them. Until now, Panda3D does not. So in this module we intend
to fill this gap in functionality. For example, this is how you fill
fields on structs in an array with random numbers, then sort the array
elements by them:

```python
from p3d_ssbo.gltypes import GlFloat, Struct, Buffer
from p3d_ssbo.algos.random_number_generator import PermutedCongruentialGenerator
from p3d_ssbo.algos.bitonic_sort import BitonicSort


struct = Struct(
    'Data',
    GlFloat('value'),
)
data_buffer = Buffer(
    'dataBuffer',
    struct('data', num_elements),
)


rng = PermutedCongruentialGenerator(
    data_buffer,
    ('data', 'value'),
)
sorter = BitonicSort(
    data_buffer,
    ('data', 'value'),
)


rng.dispatch()
sorter.dispatch()
```

The `.dispatch()` calls cause the compute shaders to be run immediately;
They can also instead be attached to level geometry, in which case they
will run in every frame in which the geometry's bounding volume is in
the camera's view.

CAVEAT
* These algorithms make many unstated assumptions about the data.
  * Most work on 1D arrays (stored top-level in the buffer)
    * ...the size of which is assumed to be a multiple of 32
    * ...except BitonicSort, which expects a power of 2 greater than 64.
* The API needs a complete overhaul.
* All classes are a big copy-and-paste job currently, and need a common
  base class.


## Tools

* `ssbo_card.SSBOCard`: Generate a quad with CardMaker. Its fragment
  shader will read the element that roughly corresponds with the card's
  UV's `x` elements, and uses it as the red channel. Used in
  `examples/main_rng_and_sort_on_a_card.py` to display sorted random
  numbers as a flickering gradient.
* `ssbo_particles`: Generate a mesh consisting of points. FIXME

One short-term goal in development is adding the same for particles.
After that... Who knows?


## Example programs

* `examples/main_rng_and_sort_on_a_card.py`: Creates quad on the screen
  that displays (as a black-red gradient) the contents of a buffer,
  which each frame gets filled with random data, and then sorted.


## TODO

* `main_boids.py`
  * Move GLSL code into its own module
  * Add weighing and exponent sliders for influences
  * Influence ranges
  * More influences
    * Waypoint
    * Obstacle
    * Birds of prey
* `p3d_ssbo.algos.spatial_hash`
  * Deal with 2D
  * `PairwiseAction`
    * Per-axis-settable boundary condition of "closed" or "looping"
    * Grid cell selection should select all cells intersecting a sphere,
      not a box. Up to 50% performance improvement.
* `gltypes`
  * Support more types.
  * API can be prettier.
  * Data should be passable to the buffers after initing the Python
    objects; `ShaderBuffer` creation should be deferrable. Of course,
    passing data after setting up everything then means dropping the old
    `ShaderBuffer` object, which may have been picked up by e.g.
    algorithms already.
* `algos`
  * Common base class, please.
  * The API has to be a lot more flexible.
  * `random_number_generator`
    * Add types as they are implemented as gltypes.
    * Make ranges settable per element component.
    * Make distributions settable.
  * `bitonic_sort`
    * Currently deals only with arrays sized `2**n`; At the least `32*n` should be supported.
* Implement foundational algorithms
  * [Radix Sort](https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf)
  * Up-/Downsampling
  * Kernel filters
  * Spatial hashing
  * Octree
  * Barnes-Hut
  * FFT
  * Kawase downsampling
  * https://en.wikipedia.org/wiki/Jump_flooding_algorithm
* Demo application
  * Fireworks
    * Basic particle system logic
  * Convolutional Bloom
    * uses FFT
    * https://www.youtube.com/watch?v=ml-5OGZC7vE
