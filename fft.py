import p3d_ssbo
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    NodePath, loadPrcFileData, Shader, ShaderBuffer, ComputeNode, ShaderInput, 
    Vec2, GeomEnums, ShaderAttrib
)
import numpy as np
from math import log
# from jinja2 import Template

config_vars = """
gl-debug true
"""
loadPrcFileData("", config_vars)

def verify_fft_np(compare: np.array) -> bool:
    np_fft = np.fft.fft(signal)
    
    if np.allclose(compare, np_fft, atol=1e-1):
        print("GPU output matches NumPy!")
        return True
    else:
        print("Discrepancy detected! 0:10:")
        print(compare[0:10])
        print(np_fft[0:10])
        print("400:410:")
        print(compare[400:410])
        print(np_fft[400:410])
    return False


class Stage:
    @staticmethod
    def build_radix(n_x: int, n_y: int, mode, vector_size: int, shared_banked: bool, 
        radix: int, size, pow2_stride: bool) -> tuple:
        wg_x: int = 0 
        wg_y: int = 0

        assert not ((n_y > 1) and n_x == 1), f'WorkGroupSize.y must be 1, when Ny == 1.'

        divisor = size.z;
        reduce(size.y, divisor)
        reduce(size.x, divisor)

        match mode:
            case "Horizontal":
                wg_x = (2 * Nx) / (vector_size * radix * size.x)
                wg_y = Ny / size.y
            case _:
                print("Mode not implemented")

        return (size, wg_x, wg_y, radix, vector_size, shared_banked);


    @staticmethod
    def is_radix_valid(n_x: int, n_y: int, mode, vector_size: int, radix: int,
        size, pow2_stride: bool):
        res = build_radix(Nx, Ny,
            mode, vector_size, false, radix,
            size,
            pow2_stride)

        return ((res.num_workgroups_x > 0) and (res.num_workgroups_y > 0))

    def __init__(self, stage_id: int, p_val: int, size: int):
        self.is_p1 = True if stage_id < 1 else False
        self.p = p_val
        self.size = int(size)
        self.nodepath = NodePath(f"Stage_{stage_id}")

        shader_file = "fft_p1_template.glsl" if self.is_p1 else "fft_template.glsl"

        with open(shader_file, "r") as f:
            shader_src = f.read()

        self.shader = Shader.make_compute(Shader.SL_GLSL, shader_src)
        self.nodepath.set_shader(self.shader)
        
        # shader_src = shader_src.replace("{{ P1 }}", p1_link) if self.is_p1 else shader_src.replace("{{ P1 }}", general_link)

class FFT4:
    def __init__(self, size: int = 1024):
        self.size = size
        self.num_stages = int(log(size, 4))
        
        # Pre-create stages
        self.stages = []
        for i in range(self.num_stages):
            p = 4**i
            self.stages.append(Stage(i, p, size/self.num_stages))

    def process(self, signal):
        zeros = np.zeros(len(signal), dtype=np.complex64)
        buf_a = ShaderBuffer("BufferA", signal.tobytes(), GeomEnums.UH_dynamic)
        buf_b = ShaderBuffer("BufferB", zeros.tobytes(), GeomEnums.UH_dynamic)

        current_in, current_out = buf_a, buf_b
        
        num_groups = max(1, (self.size // 4) // 64)

        for stage in self.stages:
            stage.nodepath.set_shader_input("Block", current_in)
            stage.nodepath.set_shader_input("BlockOut", current_out)
            stage.nodepath.set_shader_input(ShaderInput('psizevec', Vec2(p_val, self.size)))
            sattr = stage.nodepath.get_attrib(ShaderAttrib)
            gsg = base.win.get_gsg()
            base.graphicsEngine.dispatch_compute(
                (num_groups, 1, 1), 
                sattr, 
                gsg
            )
            
            current_in, current_out = current_out, current_in

        raw_output = base.graphicsEngine.extract_shader_buffer_data(current_in, gsg)
        raw_output = np.frombuffer(raw_output, dtype=np.float32)
        complex_gpu = raw_output[0::2] + 1j * raw_output[1::2]
        
        return complex_gpu

if __name__ == "__main__":
    base = ShowBase()
    
    size = 1024
    test_signal = np.sin(np.arange(size) * 2 * np.pi * 440.).astype(np.complex64)
    # print(test_sig[0:100])
    fft_engine = FFT4(size)
    output = fft_engine.process(test_signal)
    verify_fft_np(output)
