#include <metal_compute>
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void matmul(
    device bfloat *out [[buffer(0)]],
    device bfloat *A [[buffer(1)]],
    device bfloat *B [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    constexpr int M = 8192;
    constexpr int N = 8192;
    constexpr int K = 8192;

    constexpr int TileSize_M = 64;
    constexpr int TileSize_N = 64;
    constexpr int TileSize_K = 128;

    extents<int, K, M> extA;
    extents<int, N, K> extB;
    extents<int, N, M> extC;

    tensor tensorA(A, extA);
    tensor tensorB(B, extB);
    tensor tensorC(out, extC);

    uint tile_id = tgid.x;

    uint ix = tile_id & 0x55555555;
    ix = (ix | (ix >> 1)) & 0x33333333;
    ix = (ix | (ix >> 2)) & 0x0F0F0F0F;
    ix = (ix | (ix >> 4)) & 0x00FF00FF;
    ix = (ix | (ix >> 8)) & 0x0000FFFF;

    uint iy = (tile_id >> 1) & 0x55555555;
    iy = (iy | (iy >> 1)) & 0x33333333;
    iy = (iy | (iy >> 2)) & 0x0F0F0F0F;
    iy = (iy | (iy >> 4)) & 0x00FF00FF;
    iy = (iy | (iy >> 8)) & 0x0000FFFF;

    int im = ix * TileSize_M;
    int in = iy * TileSize_N;

    constexpr auto desc = matmul2d_descriptor(
        TileSize_M,
        TileSize_N,
        TileSize_K,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply_accumulate);

    matmul2d<desc, execution_simdgroups<4>> op;
    auto coop_c = op.get_destination_cooperative_tensor<decltype(tensorA), decltype(tensorB), bfloat>();

    constexpr int NumKTiles = K / TileSize_K;
    for(int i = 0; i < NumKTiles; i++)
    {
        threadgroup_barrier(mem_flags::mem_none);
        int ik = i * TileSize_K;
        auto tileA = tensorA.slice<TileSize_K, TileSize_M>(ik, im);
        auto tileB = tensorB.slice<TileSize_N, TileSize_K>(in, ik);
        op.run(tileA, tileB, coop_c);
    }

    auto tileC = tensorC.slice<TileSize_N, TileSize_M>(in, im);
    coop_c.store(tileC);
}
