#!/usr/bin/env python

import numpy as np
from run_metal_kernel import run_metal_kernel

def test_matrix_add():
    shape = (1024, 1024)
    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(*shape).astype(np.float32)

    expected = A + B
    actual = run_metal_kernel(
        "matrix_add",
        [A, B],
        shape,
        (np.prod(shape), 1, 1),
        (16, 1, 1),
    )

    np.testing.assert_allclose(
        actual,
        expected,
    )


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

if __name__ == '__main__':
    test_matmul()
