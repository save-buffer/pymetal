#include <metal_stdlib>
#include <metal_math>
#include <metal_compute>
#include <metal_logging>

using namespace metal;

constant uint nq = 32;
constant uint nkv = 8;
constant uint qctx = 128;
constant uint dhead = 128;

constexpr uint IQK(uint iqhead, uint iqctx, uint ictx, uint nctx)
{
    // [nq, qctx, nctx]
    return iqhead * qctx * nctx + iqctx * nctx + ictx;
}

constexpr uint IQ(uint iqhead, uint iqctx)
{
    // [nq, qctx, dhead]
    return iqhead * qctx * dhead + iqctx * dhead;
}

constexpr uint IKV(uint ikvhead, uint ictx, uint nctx)
{
    // [nkv, nctx, dhead]
    return ikvhead * nctx * dhead + ictx * dhead;
}

constexpr uint IO(uint iqhead, uint iqctx)
{
    // [nq, qctx, dhead]
    return iqhead * qctx * dhead + iqctx * dhead;
}

kernel void gqa(
    device float *out,
    device float *out_qk,
    device const float *Q,
    device const float *K,
    device const float *V,
    constant uint &nctx,
    uint2 threadgroup_idx [[threadgroup_position_in_grid]],
    uint2 thread_index [[thread_position_in_threadgroup]],
    uint warp_size [[threads_per_simdgroup]])
{
    uint iqhead = threadgroup_idx.x;
    uint iqctx = threadgroup_idx.y;
    uint ikvhead = iqhead % nkv;

    float running_max = -INFINITY;
    float running_l = 0.0f;

    threadgroup float logits[32] = {};
    threadgroup float o[dhead] = {};

    float prev_d = 0.0;
    for(uint ictx = thread_index.x; ictx < nctx; ictx += warp_size)
    {
        float qk_score = 0.0;
        for(uint i = 0; i < dhead; i++)
        {
            int iq = IQ(iqhead, iqctx) + i;
            int ik = IKV(ikvhead, ictx, nctx) + i;
            qk_score += Q[iq] * K[ik];
        }

        qk_score /= sqrt(static_cast<float>(dhead));

        uint iqk = IQK(iqhead, iqctx, ictx, nctx);
        out_qk[iqk] = qk_score;

        // Now we have 1x32 qk-scores
        float tile_max = simd_max(qk_score);
        float logit = exp(qk_score - tile_max);

        float tile_l = simd_sum(logit);
        float new_max = fmax(tile_max, running_max);
        float new_l = exp(running_max - new_max) * running_l + exp(tile_max - new_max) * tile_l;

        // now we compute [1x32] x [32x128] => 1x128
        logits[thread_index.x] = logit;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        uint warp_base_ctx = ictx - thread_index.x;
        for(uint id = thread_index.x; id < dhead; id += warp_size)
        {
            float v_proj = 0.0;
            for(uint iqk = 0; iqk < warp_size; iqk++)
            {
                uint iv = IKV(ikvhead, warp_base_ctx + iqk, nctx) + id;
                v_proj += logits[iqk] * V[iv];
            }
            float rescaled_old_o = running_l * exp(running_max - new_max) * o[id];
            float rescaled_v_proj = exp(tile_max - new_max) * v_proj;
            float new_o = (rescaled_old_o + rescaled_v_proj) / new_l;
            o[id] = new_o;
        }
        running_l = new_l;
        running_max = new_max;
    }
    for(uint i = thread_index.x; i < dhead; i += warp_size)
    {
        int io = IO(iqhead, iqctx) + i;
        out[io] = o[i];
    }
}
