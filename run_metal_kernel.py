import ctypes
from dataclasses import dataclass
from tempfile import mkdtemp
import os
from typing import Sequence
import subprocess

import numpy as np

import objc
from Foundation import NSBundle, NSData, NSError, NSURL
from Metal import *

def init_metal():
    device = MTLCreateSystemDefaultDevice()
    if not device:
        raise RuntimeError("Metal is not supported on this device")
    return device

g_device = init_metal()

def compile_kernel(device, kernel_path, enable_logging=False):
    with open(kernel_path, 'r') as f:
        kernel_source = f.read()

    options = MTLCompileOptions()
    options.setEnableLogging_(enable_logging)
    library, error = device.newLibraryWithSource_options_error_(
        kernel_source, options, None
    )
    if error:
        raise RuntimeError(f"Failed to compile kernel: {error.localizedDescription()}")
    
    return library

def execute_kernel(
    device,
    library,
    kernel_name,
    inputs,
    outputs,
    grid_size,
    threadgroup_size,
    output_dtype,
    profile=False,
):
    kernel_function = library.newFunctionWithName_(kernel_name)
    if not kernel_function:
        raise RuntimeError(f"Kernel '{kernel_name}' not found")
    
    pipeline_state, error = device.newComputePipelineStateWithFunction_error_(
        kernel_function,
        None,
    )
    if error:
        raise RuntimeError(f"Pipeline creation failed: {error.localizedDescription()}")
    
    buffers = []
    for i, out in enumerate(outputs):
        output_bytes = np.prod(out) * np.dtype(output_dtype[i]).itemsize
        output_buffer = device.newBufferWithLength_options_(
            output_bytes,
            MTLResourceStorageModeShared,
        )
        if not output_buffer:
            raise RuntimeError(f"Failed to allocate output {i} of size {output_bytes} bytes")
        buffers.append(output_buffer)
        
    for i, input in enumerate(inputs):
        input_buffer_size = input.nbytes
        # TODO: Probably possible to make a new buffer that wraps the numpy array bytes
        # TODO: If an input is an int, make it use setBytes
        input_buffer = device.newBufferWithBytes_length_options_(
            input,
            input_buffer_size,
            MTLResourceStorageModeShared,
        )
        if not input_buffer:
            raise RuntimeError(f"Failed to allocate input {i} of size {input_buffer_size} bytes")
        buffers.append(input_buffer)

    command_queue = g_device.newCommandQueue()
    if profile:
        capture_manager = MTLCaptureManager.sharedCaptureManager()
        if not capture_manager.supportsDestination_(MTLCaptureDestinationGPUTraceDocument):
            raise RuntimeError("Capturing to GPU trace file is not supported")

        trace_url = NSURL.fileURLWithPath_(mkdtemp() + "/trace.gputrace")
        capture_descriptor = MTLCaptureDescriptor()
        capture_descriptor.setCaptureObject_(command_queue)
        capture_descriptor.setDestination_(MTLCaptureDestinationGPUTraceDocument)
        capture_descriptor.setOutputURL_(trace_url)

        _, error = capture_manager.startCaptureWithDescriptor_error_(
            capture_descriptor,
            None,
        )
        if error:
            raise RuntimeError(f"Failed to start capture: {error.localizedDescription()}")

    command_buffer = command_queue.commandBuffer()
    compute_encoder = command_buffer.computeCommandEncoder()
    
    compute_encoder.setComputePipelineState_(pipeline_state)
    for i, buffer in enumerate(buffers):
        compute_encoder.setBuffer_offset_atIndex_(buffer, 0, i)
        
    mtl_grid_size = MTLSizeMake(*grid_size)
    mtl_threadgroup_size = MTLSizeMake(*threadgroup_size)

    compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(mtl_grid_size, mtl_threadgroup_size)
    
    compute_encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    if command_buffer.status() == MTLCommandBufferStatusError:
        raise RuntimeError(f"Kernel command returned error: {command_buffer.error()}")

    if profile:
        capture_manager.stopCapture()
        subprocess.run(["open", trace_url])

    results = []
    for i, output_shape in enumerate(outputs):
        output_bytes = np.prod(output_shape) * np.dtype(output_dtype[i]).itemsize
        output_ptr = bytes(buffers[i].contents().as_buffer(output_bytes))
        output = np.frombuffer(output_ptr, dtype=output_dtype[i], count=np.prod(output_shape))
        results.append(output.reshape(output_shape))
    return results[0] if len(results) == 1 else tuple(results)


def clean_inputs(inputs : list[np.ndarray | int]):
    input_arrays = []
    for input in inputs:
        if isinstance(input, int):
            input_arrays.append(np.array(input).astype(np.uint32))
        else:
            input_arrays.append(input)
    return input_arrays


def clean_outputs(output_shape):
    if not isinstance(output_shape, (tuple, list)):
        raise ValueError("Argument output_shape is not a sequence")
    if len(output_shape) == 0:
        return ()
    if isinstance(output_shape[0], tuple):
        return output_shape
    return (output_shape,)


def clean_output_dtype(outputs_clean, output_dtype):
    if output_dtype is None:
        result = [np.float32] * len(outputs_clean)
    elif isinstance(output_dtype, np.dtype):
        result = [output_dtype]
    else:
        result = output_dtype

    if output_dtype is not None and len(outputs_clean) != len(out_dtype_clean):
        raise ValueError("Not enough output datatypes specified!")
    return result

def run_metal_kernel(
    kernel_name : str,
    output_shape : tuple[int, ...] | Sequence[tuple[int, ...]],
    inputs : list[np.ndarray | int],
    grid_size : tuple[int, int, int],
    threadgroup_size : tuple[int, int, int],
    output_dtype : np.dtype | list[np.dtype] | None = None,
    enable_logging : bool = False,
    profile : bool = False,
) -> np.ndarray | tuple[np.ndarray, ...]:
    if np.prod(threadgroup_size) > 1024:
        raise ValueError("Metal only supports up to 1024 threads per threadgroup")

    inputs_clean = clean_inputs(inputs)
    outputs_clean = clean_outputs(output_shape)
    output_dtype_clean = clean_output_dtype(outputs_clean, output_dtype)

    library = compile_kernel(g_device, f"kernels/{kernel_name}.metal", enable_logging=enable_logging)
    result = execute_kernel(
        g_device,
        library,
        kernel_name,
        inputs_clean,
        outputs_clean,
        grid_size,
        threadgroup_size,
        output_dtype=output_dtype_clean,
        profile=profile,
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
