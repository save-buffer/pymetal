import numpy as np
import objc
from Foundation import NSBundle, NSData, NSError
from Metal import *
import ctypes
import os
from dataclasses import dataclass

def init_metal():
    device = MTLCreateSystemDefaultDevice()
    if not device:
        raise RuntimeError("Metal is not supported on this device")
    command_queue = device.newCommandQueue()
    return device, command_queue

g_device, g_command_queue = init_metal()

def compile_kernel(device, kernel_path):
    with open(kernel_path, 'r') as f:
        kernel_source = f.read()
    
    library, error = device.newLibraryWithSource_options_error_(
        kernel_source, None, None
    )
    if error:
        raise RuntimeError(f"Failed to compile kernel: {error.localizedDescription()}")
    
    return library

def execute_kernel(
    device,
    command_queue,
    library,
    kernel_name,
    inputs,
    output_shape,
    grid_size,
    threadgroup_size,
    output_dtype = np.float32,
):
    kernel_function = library.newFunctionWithName_(kernel_name)
    if not kernel_function:
        raise RuntimeError(f"Kernel '{kernel_name}' not found")
    
    pipeline_state, error = device.newComputePipelineStateWithFunction_error_(
        kernel_function, None
    )
    if error:
        raise RuntimeError(f"Pipeline creation failed: {error.localizedDescription()}")
    
    buffers = []
    output_bytes = np.prod(output_shape) * np.dtype(output_dtype).itemsize
    output_buffer = device.newBufferWithLength_options_(
        output_bytes, MTLResourceStorageModeShared
    )
    buffers.append(output_buffer)
    
    for input in inputs:
        buffer_size = input.nbytes
        buffer = device.newBufferWithBytes_length_options_(
            input, buffer_size, MTLResourceStorageModeShared
        )
        buffers.append(buffer)
    
    command_buffer = command_queue.commandBuffer()
    compute_encoder = command_buffer.computeCommandEncoder()
    
    compute_encoder.setComputePipelineState_(pipeline_state)
    for i, buffer in enumerate(buffers):
        compute_encoder.setBuffer_offset_atIndex_(buffer, 0, i)
    
    mtl_grid_size = MTLSizeMake(*grid_size)
    mtl_threadgroup_size = MTLSizeMake(*threadgroup_size)

    compute_encoder.dispatchThreads_threadsPerThreadgroup_(mtl_grid_size, mtl_threadgroup_size)
    
    compute_encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    
    output_ptr = bytes(output_buffer.contents().as_buffer(output_bytes))
    output = np.frombuffer(output_ptr, dtype=np.float32, count=np.prod(output_shape))
    return output.reshape(output_shape)

def clean_inputs(inputs : list[np.ndarray | int]):
    input_arrays = []
    for input in inputs:
        if isinstance(input, int):
            input_arrays.append(np.array(input).astype(np.uint32))
        else:
            input_arrays.append(input)
    return input_arrays


def run_metal_kernel(
    kernel_name : str,
    inputs : list[np.ndarray | int],
    output_shape : tuple[int, ...],
    grid_size : tuple[int, int, int],
    threadgroup_size : tuple[int, int, int],
) -> np.ndarray:
    inputs_clean = clean_inputs(inputs)

    library = compile_kernel(g_device, f"kernels/{kernel_name}.metal")
    result = execute_kernel(
        g_device,
        g_command_queue,
        library,
        kernel_name,
        inputs_clean,
        output_shape,
        grid_size,
        threadgroup_size,
    )
    return result

def main(width=1024, height=1024):
    A = np.random.rand(height, width).astype(np.float32)
    B = np.random.rand(height, width).astype(np.float32)
    
    metal_result = run_metal_kernel(
        "matrix_add",
        [A, B],
        (height, width),
        (height * width, 1, 1),
        (16, 1, 1),
    )
    
    expected_result = A + B

    if np.allclose(metal_result, expected_result, atol=1e-5):
        print("Results match!")
    else:
        print("Results differ!")
        print(f"Metal output (first 10): {metal_result.flatten()[:10]}")
        print(f"NumPy output (first 10): {expected_result.flatten()[:10]}")

if __name__ == "__main__":
    main()
