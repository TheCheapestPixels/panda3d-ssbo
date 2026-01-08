import numpy as np
import math
import time
from panda3d.core import (
    NodePath, Shader, ShaderAttrib, # ShaderBuffer, 
    GeomEnums, ComputeNode, load_prc_file_data,
    GraphicsPipeSelection, FrameBufferProperties,
    WindowProperties, GraphicsPipe, CullBinManager
)
from direct.showbase.ShowBase import ShowBase
from p3d_ssbo.gltypes import GLType, Buffer, GLVec2

DIGIT_REVERSE_SHADER = """
#version 430
layout (local_size_x = 64) in;
layout(std430, binding = 0) buffer DA { vec2 data_in[]; };
layout(std430, binding = 1) buffer DR { vec2 data_out[]; };

uniform uint nItems;
uniform uint log16N;

uint reverse_digits_base16(uint x, uint n) {
    uint res = 0;
    for (uint i = 0; i < n; i++) {
        res = (res << 4) | (x & 0xF);
        x >>= 4;
    }
    return res;
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= nItems) return;
    uint target = reverse_digits_base16(gid, log16N);
    data_out[target] = data_in[gid];
}
"""

RADIX16_BUTTERFLY_SHADER = """
#version 430
layout (local_size_x = 64) in;
layout(std430, binding = 0) buffer DA { vec2 data_in[]; };
layout(std430, binding = 1) buffer DR { vec2 data_out[]; };

uniform uint nItems;
uniform uint stage;
uniform int inverse;

const float PI = 3.14159265358979323846;

vec2 complex_mul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

dvec2 lie_twiddle(float theta) {
    // define the complex vector corresponding to the angle of rotation, without using trig
    // use Baker-Campbell-Haussdorff (BCH)? exponential map? need Lie brackets. placeholder is trig lol
    // i*theta -> e^i*theta. tangent line maos to circle
    double real = cos(theta);
    double imaginary = sin(theta);
    return dvec2(real, imaginary);
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint elements_per_group = stage * 16;
    uint num_groups = nItems / elements_per_group;
    
    if (gid >= num_groups * stage) return;
    
    uint group_id = gid / stage;
    uint elem_in_stage = gid % stage;
    uint base_idx = group_id * elements_per_group + elem_in_stage;
    
    // Load 16 elements
    vec2 temp[16];
    for (uint k = 0; k < 16; k++) {
        temp[k] = data_in[base_idx + k * stage];
    }
    
    // Compute 16-point DFT using twiddle factors
    vec2 result[16];
    for (uint k = 0; k < 16; k++) {
        result[k] = vec2(0.0, 0.0);
        for (uint n = 0; n < 16; n++) {
            float angle = -1.0 * inverse * 2.0 * PI * float(k * n) / 16.0;
            float angle_stage = -1.0 * inverse * 2.0 * PI * float(elem_in_stage * n) / float(elements_per_group);
            float total_angle = angle + angle_stage;
            vec2 twiddle = vec2(cos(total_angle), sin(total_angle));
            result[k] += complex_mul(temp[n], twiddle);
        }
    }
    
    // Write back
    for (uint k = 0; k < 16; k++) {
        data_out[base_idx + k * stage] = result[k];
    }
}
"""

class Radix16FFT:
    def __init__(self, app):
        self.app = app
        self._setup_context()
        
        # Compile Radix-16 FFT Shaders
        self.dr_node = self._compile(DIGIT_REVERSE_SHADER)
        self.fft16_node = self._compile(RADIX16_BUTTERFLY_SHADER)

        self.render_output = False

        self.buffA: Buffer = None
        self.buffB: Buffer = None

    def _setup_context(self):
        pipe = GraphicsPipeSelection.get_global_ptr().make_default_pipe()
        fb_prop = FrameBufferProperties()
        win_prop = WindowProperties.size(1, 1)
        self.app.win = self.app.graphics_engine.make_output(
            pipe, "fft16_headless", 0, fb_prop, win_prop, GraphicsPipe.BF_refuse_window
        )

    def _compile(self, code):
        CullBinManager.get_global_ptr().add_bin("fft_bin",
                CullBinManager.BT_fixed, 0)
        shader = Shader.make_compute(Shader.SL_GLSL, code)
        node = NodePath(ComputeNode("fft16_op"))
        shader.attach(node, "fft_bin")
        # node.set_shader(shader)
        
        # Digit-Reversal Pass (base 16)
        self.buffB = Buffer("DR_Out", self.n * 8, GeomEnums.UH_stream)
        self.dr_node.set_shader_input("DA", self.buffA.buffer)
        self.dr_node.set_shader_input("DR", self.buffB)
        self.dr_node.set_shader_input("nItems", int(self.n))
        self.dr_node.set_shader_input("log16N", int(self.log16n))
        
        # Radix-16 Butterfly Stages
        for s in range(self.log16n):
            stage_size = 16 ** s
            # out_sbuf = ShaderBuffer(f"Stage_{s}", n * 8, GeomEnums.UH_stream)
            # out_sbuf = Buffer(f"Stage_{s}", GLVec2)

            self.fft16_node.set_shader_input("DA", self.buffB)
            self.fft16_node.set_shader_input("DR", self.buffA)
            self.fft16_node.set_shader_input("nItems", int(n))
            self.fft16_node.set_shader_input("stage", int(stage_size))
            self.fft16_node.set_shader_input("inverse", inv_flag)

        return node

    def push(self, data):
        """Upload complex data to GPU as vec2."""
        data = np.ascontiguousarray(data, dtype=np.complex64)
        # sbuf = ShaderBuffer("Data", data.tobytes(), GeomEnums.UH_stream)
        return Buffer("Data", GLVec2, data.tobytes())

    def fetch(self, gpu_handle):
        """Download buffer back to NumPy."""
        gsg = self.app.win.get_gsg()
        raw = self.app.graphics_engine.extract_shader_buffer_data(gpu_handle.buffer, gsg)
        return np.frombuffer(raw, dtype=gpu_handle.cast)

    def fft(self, data, inverse=False):
        """Radix-16 Cooley-Tukey GPU implementation."""
        buf = data if isinstance(data, Buffer) else self.push(data)
        n = buf.n_items
        
        # Ensure n is a power of 16
        log16n = int(math.log(n) / math.log(16))
        if 16 ** log16n != n:
            raise ValueError(f"Size must be power of 16, got {n}")
        
        inv_flag = -1 if inverse else 1

        # Digit-Reversal Pass (base 16)
        self.app.graphics_engine.dispatch_compute(
            ((n + 63) // 64, 1, 1), self.dr_node.get_attrib(ShaderAttrib), self.app.win.get_gsg()
        )
        
        # Radix-16 Butterfly Stages
        for s in range(log16n):
            num_work_items = (self.n // 16) 
            self.app.graphics_engine.dispatch_compute(
                ((num_work_items + 63) // 64, 1, 1), 
                self.fft16_node.get_attrib(ShaderAttrib), 
                self.app.win.get_gsg()
            )

        res = Buffer("output", GLVec2, bind_buffer=self.buffA)
        
        if inverse:
            return self.fetch(res) / self.n
        return res

    def rolling_fft(self, signal, frame_count, window_size=256):
        # generate window (dirichlet)
        t = np.linspace(0, 1, window_size) + (frame_count * 0.01)
        # setup compute fft
        self.buffA = data if isinstance(data, Buffer) else self.push(data)
        self.n = self.buffA.n_items
        
        # Ensure n is a power of 16
        self.log16n = int(math.log(self.n) / math.log(16))
        if 16 ** self.log16n != self.n:
            raise ValueError(f"Size must be power of 16, got {self.n}")
        
        inv_flag = -1 if inverse else 1

    def fft_tick(self, signal):
        self.push(signal.asType(np.complex64)
        complex_result = self.fetch(fft_handle)

        self.app.graphics_engine.dispatch_compute(
            ((self.n + 63) // 64, 1, 1), self.dr_node.get_attrib(ShaderAttrib), self.app.win.get_gsg()
        )

        # this will now be picked up by the card's frag shader due to the compute shader's early bin timing

class FFT16Demo(ShowBase):
    def __init__(self):
        load_prc_file_data("", "window-type none\naudio-library-name null")
        ShowBase.__init__(self)
        self.hmath = Radix16FFT(self)
        self.frame_counter = 0.

    def run_test(self, N=65536):
        print(f"Testing Radix-16 {N}-point FFT...")
        
        # Generate test signal
        t = np.linspace(0, 1, N)
        x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        x = x + 0.2 * np.random.randn(N)
        x = x.astype(np.complex64)

        # GPU FFT
        t0 = time.perf_counter()
        g_res_handle = self.hmath.fft(x)
        final_gpu = self.hmath.fetch(g_res_handle)
        t_gpu = time.perf_counter() - t0

        # CPU FFT (reference)
        t1 = time.perf_counter()
        final_cpu = np.fft.fft(x)
        t_cpu = time.perf_counter() - t1

        # Display results
        print("\n" + "="*90)
        print(f"{'Index':<8} | {'GPU Value':<30} | {'CPU Value':<30} | {'Abs Diff':<12}")
        print("-" * 90)
        for i in range(100, 120): 
            g = final_gpu[i]
            c = final_cpu[i]
            diff = abs(g - c)
            print(f"{i:<8} | {str(g):<30} | {str(c):<30} | {diff:.3e}")
        print("="*90 + "\n")

        max_diff = np.max(np.abs(final_gpu - final_cpu))
        mean_diff = np.mean(np.abs(final_gpu - final_cpu))
        
        print(f"GPU Time:  {t_gpu:.5f}s")
        print(f"CPU Time:  {t_cpu:.5f}s")
        print(f"Speedup:   {t_cpu/t_gpu:.2f}x")
        print(f"Max Diff:  {max_diff:.3e}")
        print(f"Mean Diff: {mean_diff:.3e}")
        print(f"Valid:     {max_diff < 1e-1}")
        
        # Test inverse FFT
        print("\nTesting Inverse FFT...")
        t2 = time.perf_counter()
        final_inv = self.hmath.fft(g_res_handle, inverse=True)
        t_inv = time.perf_counter() - t2
        
        inv_diff = np.max(np.abs(final_inv - x))
        print(f"IFFT Time:     {t_inv:.5f}s")
        print(f"Roundtrip Diff: {inv_diff:.3e}")
        print(f"IFFT Valid:    {inv_diff < 1e-1}")
    
    def rolling_test(self, task):
        # Create two frequencies: 50Hz and a drifting high frequency
        freq2 = 80 + 20 * np.sin(self.frame_counter * 0.05)
        signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
        signal += 0.2 * np.random.randn(self.N) # Add noise
        
        # update fft stack
        # self.hmath.set_out(card)
        self.hmath.rolling_fft(signal)

        # --- Visualization ---
        # Calculate Magnitude Spectrum
        # We only need the first half (0 to Nyquist frequency)
        magnitudes = np.abs(complex_result[:self.N // 2])

        self.frame_counter += 1
        return task.cont

if __name__ == "__main__":
    # Test with different sizes (all powers of 16)
    sizes = [16**2, 16**3, 16**4, 16**5]  # 256, 4096, 65536
    
    for size in sizes:
        print(f"\n{'='*90}")
        print(f"Testing N = {size}")
        print('='*90)
        app = FFT16Demo()
        app.run_test(N=size)
        app.destroy()
        print()
