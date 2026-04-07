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
    uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;

    constexpr int TileSize_M = 64;
    constexpr int TileSize_N = 64;

    extents<int, K, M> extA;
    extents<int, N, K> extB;
    extents<int, N, M> extC;

    tensor tensorA(A, extA);
    tensor tensorB(B, extB);
    tensor tensorC(out, extC);

    int im = tgid.x * TileSize_M;
    int in = tgid.y * TileSize_N;

    auto tileA = tensorA.slice(0, im);
    auto tileB = tensorB.slice(in, 0);
    auto tileC = tensorC.slice(in, im);

    constexpr auto desc = matmul2d_descriptor(
        TileSize_M,
        TileSize_N,
        K);

    matmul2d<desc, execution_simdgroups<4>> op;
    op.run(tileA, tileB, tileC);
}
