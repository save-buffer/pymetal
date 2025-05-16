#include <metal_stdlib>
using namespace metal;

kernel void matrix_add(
    device float *out [[buffer(0)]],
    device const float *A [[buffer(1)]],
    device const float *B [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    out[index] = A[index] + B[index];
}
