#!/usr/bin/env python

import numpy as np
from einops import rearrange, einsum
from run_metal_kernel import run_metal_kernel

def test_matrix_add():
    shape = (1024, 1024)
    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(*shape).astype(np.float32)

    expected = A + B
    actual = run_metal_kernel(
        "matrix_add",
        shape,
        [A, B],
        (np.prod(shape) // 128, 1, 1),
        (128, 1, 1),
    )

    np.testing.assert_allclose(
        actual,
        expected,
    )


def test_matmul_simple():
    M, N, K = 4096, 4096, 4096
    A = np.random.randn(M, N).astype(np.float32)
    B = np.random.randn(N, K).astype(np.float32)

    expected = A @ B
    actual = run_metal_kernel(
        "matmul_simple",
        (M, K),
        [A, B, N],
        (M // 32, K // 32, 1),
        (32, 32, 1),
        enable_logging=True,
    )
    np.testing.assert_allclose(
        actual,
        expected,
    )


def softmax(x):
    m = np.max(x, axis=-1, keepdims=True)
    z = np.exp(x - m)
    s = np.sum(z, axis=-1, keepdims=True)
    return z / s

def gqa_reference(Q, K, V):
    nq, qctx, dhead = Q.shape
    nkv, nctx, _ = K.shape
 
    Q = rearrange(Q, '(groups nkv) qctx dhead -> groups nkv qctx dhead', nkv=nkv)
    QK = einsum(Q, K, 'groups nkv qctx dhead, nkv nctx dhead -> groups nkv qctx nctx')
    QK /= np.sqrt(dhead)
    logits = softmax(QK)
    O = einsum(logits, V, 'groups nkv qctx nctx, nkv nctx dhead -> groups nkv qctx dhead')
    result = rearrange(O, 'groups nkv qctx dhead -> (groups nkv) qctx dhead')

    return result, QK


def test_gqa():
    # Llama 3 8B config
    nq = 32
    nkv = 8
    dhead = 128
    qctx = 128
    nctx = 4096

    np.random.seed(42)
    Q = np.random.randn(nq, qctx, dhead).astype(np.float32)
    K = np.random.randn(nkv, nctx, dhead).astype(np.float32)
    V = np.random.randn(nkv, nctx, dhead).astype(np.float32)

    expected, exp_qk = gqa_reference(Q, K, V)
    actual, act_qk = run_metal_kernel(
        "gqa",
        [
            (nq, qctx, dhead),
            (nq, qctx, nctx),
        ],
        [Q, K, V, nctx],
        (nq, qctx, 1),
        (32, 1, 1),
    )
    np.testing.assert_allclose(
        act_qk,
        exp_qk.reshape(act_qk.shape),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-5,
        rtol=1e-5,
    )

if __name__ == '__main__':
    test_matrix_add()
    test_matmul_simple()
    test_gqa()
