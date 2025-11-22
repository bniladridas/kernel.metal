// SPDX-License-Identifier: LicenseRef-KERNELMETAL-NC
// Copyright (c) 2025 KERNEL.METAL (harpertoken)

#include <metal_stdlib>
using namespace metal;

// This is the compute kernel function that performs vector addition on the GPU.
// It takes three buffers: A and B as input vectors, C as output vector.
// Each thread processes one element of the vectors.
// The thread position in the grid determines which element this thread handles.
kernel void vector_add(
    // Buffer 0: input vector A, read-only
    device const float* A [[buffer(0)]],
    // Buffer 1: input vector B, read-only  
    device const float* B [[buffer(1)]],
    // Buffer 2: output vector C, write-only
    device float* C [[buffer(2)]],
    // The global thread ID, used to index into the arrays
    uint id [[thread_position_in_grid]]
) {
    // Perform the addition: C[id] = A[id] + B[id]
    // This is executed in parallel across all threads
    C[id] = A[id] + B[id];
}