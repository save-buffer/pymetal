# PyMetal is a Python-Driven Kernel Library for Apple GPUs
It's designed as a tool for rapid iteration of Metal compute kernels for ML applications. An 
example of how to use it is in `test_kernels.py`:
```python
def test_matmul():
    M, N, K = 512, 512, 512
    A = np.random.randn(M, N).astype(np.float32)
    B = np.random.randn(N, K).astype(np.float32)

    expected = A @ B
    actual = run_metal_kernel(
        "matmul",
        [A, B, N],
        (M, K),
        (M, K, 1),
        (16, 16, 1),
    )
    
    np.testing.assert_allclose(
        actual,
        expected,
    )
```

The main function, `run_metal_kernel` looks for a kernel called `matmul` in the `kernels/` 
subdirectory. Specifically, it opens the file `kernels/matmul.metal` and invokes the function
called `matmul` in `matmul.metal`. The name of the kernel must match the name of its source file!

The next argument is a list of inputs. An input can be either a `numpy` array or an integer, and
the arguments must appear in the same order they are declared in the kernel. There is no
checking for this at this time, so make sure they match! 

Next is the output shape, so that the function knows how big of an output buffer to allocate.

Lastly we provide the grid size and threadgroup size as the last two arguments.
Let's see what the corresponding Metal shader's signature looks like:
```metal
kernel void matmul(
    device float *out [[buffer(0)]],
    device const float *A [[buffer(1)]],
    device const float *B [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]])
{
    ...
}
```

Notice that the `out` buffer is the first argument. `buffer(0)` is always the output. At this time
we only support one output. The next two arguments are `A` and `B`, and correspond to `numpy` arrays
in the Python invocation. Next we have `N`, which is a `uint`. This is passed as a Python `int` to
`run_metal_kernel`, and gets correctly allocated and copies to the device as you'd expect (internally
it gets converted into a numpy array of size 1). The last two arguments are arguments that the
kernel receives from the runtime, so we can omit them from the Python side. 
