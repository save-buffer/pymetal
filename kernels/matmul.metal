#include <metal_compute>
#include <metal_stdlib>
#include <metal_simdgroup_matrix>

#define BlockSize_M 64
#define BlockSize_N 32
#define BlockSize_K 64
#define TileSize 8

using float8x8 = metal::simdgroup_float8x8;

kernel void matmul(
    device float *out [[buffer(0)]],
    device const float *A [[buffer(1)]],
    device const float *B [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint2 block_id [[threadgroup_position_in_grid]],
    uint thread_id_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint nsimdgroups [[simdgroups_per_threadgroup]],
    uint simdgroup_size [[threads_per_simdgroup]],
    uint2 nblocks [[threadgroups_per_grid]])
{
    uint M = 4096;//nblocks.x * BlockSize_M;
    uint K = 4096; // nblocks.y * BlockSize_K;

    uint thread_id = simdgroup_id * simdgroup_size + thread_id_in_simdgroup;
    uint nthreads = 128; //nsimdgroups * simdgroup_size;

    constexpr uint TileDim_M = BlockSize_M / TileSize;
    constexpr uint TileDim_N = BlockSize_N / TileSize;
    constexpr uint TileDim_K = BlockSize_K / TileSize;
    constexpr uint TilesPerBlock = (BlockSize_M * BlockSize_K) / (TileSize * TileSize);

    uint im_base = block_id.x * BlockSize_M;
    uint ik_base = block_id.y * BlockSize_K;

    threadgroup float tg_block_a[BlockSize_M * BlockSize_N];
    threadgroup float tg_block_b[BlockSize_N * BlockSize_K];
    threadgroup float tg_block_out[BlockSize_M * BlockSize_K];

    float8x8 tile_a;
    float8x8 tile_b;
    float8x8 tile_out;

    for(uint in_base = 0; in_base < N; in_base += BlockSize_N)
    {
        #pragma clang loop unroll(full)
        for(uint iload = thread_id; iload < BlockSize_M * BlockSize_N; iload += nthreads)
        {
            uint offset_m = iload / BlockSize_N;
            uint offset_n = iload % BlockSize_N;
            uint im = im_base + offset_m;
            uint in = in_base + offset_n;
            uint istore_a = offset_m * BlockSize_N + offset_n;
            uint iload_a = im * N + in;
            tg_block_a[istore_a] = A[iload_a];
        }
        #pragma clang loop unroll(full)
        for(uint iload = thread_id; iload < BlockSize_N * BlockSize_K; iload += nthreads)
        {
            uint offset_n = iload / BlockSize_K;
            uint offset_k = iload % BlockSize_K;
            uint in = in_base + offset_n;
            uint ik = ik_base + offset_k;
            uint istore_b = offset_n * BlockSize_K + offset_k;
            uint iload_b = in * K + ik;
            tg_block_b[istore_b] = B[iload_b];
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        for(uint itile = simdgroup_id; itile < TilesPerBlock; itile += nsimdgroups)
        {
            uint itile_m = itile / TileDim_K;
            uint itile_k = itile % TileDim_K;
            uint iout = itile_m * TileSize * BlockSize_K + itile_k * TileSize;
            simdgroup_load(tile_out, tg_block_out + iout, BlockSize_K);

            #pragma clang loop unroll(full)
            for(uint itile_n = 0; itile_n < TileDim_N; itile_n += 1)
            {
                uint ia = itile_m * TileSize * BlockSize_N + itile_n * TileSize;
                uint ib = itile_n * TileSize * BlockSize_K + itile_k * TileSize;
                simdgroup_load(tile_a, tg_block_a + ia, BlockSize_N);
                simdgroup_load(tile_b, tg_block_b + ib, BlockSize_K);
                simdgroup_multiply_accumulate(tile_out, tile_a, tile_b, tile_out);
            }
            simdgroup_store(tile_out, tg_block_out + iout, BlockSize_K);
        }
        threadgroup_barrier(metal::mem_flags::mem_none);
    }

#if 0
    uint im_out = im_base + (simdgroup_id / TileDim) * 8;
    uint ik_out = ik_base + (simdgroup_id % TileDim) * 8;
    uint iout = im_out * K + ik_out;
    simdgroup_store(tile_out, out + iout, M);
#endif

    for(uint istore = thread_id; istore < BlockSize_M * BlockSize_K; istore += nthreads)
    {
        uint offset_m = istore / BlockSize_K;
        uint offset_k = istore % BlockSize_K;
        uint im = im_base + offset_m;
        uint ik = ik_base + offset_k;
        uint istore_out = im * K + ik;
        out[istore_out] = tg_block_out[istore];
    }
}
