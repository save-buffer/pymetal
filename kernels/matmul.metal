#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device float *out [[buffer(0)]],
    device const float *A [[buffer(1)]],
    device const float *B [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]])
{
    uint M = grid_size.x;
    uint K = grid_size.y;

    uint im = gid.y;
    uint ik = gid.x;

    float result = 0.0;
    
    for(uint in = 0; in < N; in++)
    {
        uint ia = im * M + in;
        uint ib = in * N + ik;
        result += A[ia] * B[ib];
    }

    uint iout = im * M + ik;
    out[iout] = result;
}
