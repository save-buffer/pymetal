#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using float8x8 = metal::simdgroup_float8x8;

kernel void matmul(
    device float *out [[buffer(0)]],
    device const float *A [[buffer(1)]],
    device const float *B [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint2 tile_id [[threadgroup_position_in_grid]],
    uint2 ntiles [[threadgroups_per_grid]])
{
    uint M = ntiles.x * 8;
    uint K = ntiles.y * 8;

    uint im = tile_id.x * 8;
    uint ik = tile_id.y * 8;

    float8x8 tile_a;
    float8x8 tile_b;
    float8x8 result = {};

    for(uint in = 0; in < N; in += 8)
    {
        uint ia = im * M + in;
        uint ib = in * N + ik;

        simdgroup_load(tile_a, A + ia, N);
        simdgroup_load(tile_b, B + ib, K);
        simdgroup_multiply_accumulate(result, tile_a, tile_b, result);
    }

    uint iout = im * M + ik;
    simdgroup_store(result, out + iout, K);
}
