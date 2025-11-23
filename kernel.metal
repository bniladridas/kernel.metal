// SPDX-License-Identifier: MIT
// Copyright (c) 2025 KERNEL.METAL (harpertoken)

#include <metal_stdlib>
using namespace metal;

// This is the compute kernel function that performs vector addition on the GPU.
// It takes three buffers: A and B as input vectors, C as output vector, and count as the number of elements.
// Each thread processes one element of the vectors.
// The thread position in the grid determines which element this thread handles.
// Includes bounds check to prevent out-of-bounds access.
kernel void vector_add(
    // Buffer 0: input vector A, read-only
    device const float* A [[buffer(0)]],
    // Buffer 1: input vector B, read-only
    device const float* B [[buffer(1)]],
    // Buffer 2: output vector C, write-only
    device float* C [[buffer(2)]],
    // Buffer 3: element count, constant
    constant uint &count [[buffer(3)]],
    // The global thread ID, used to index into the arrays
    uint id [[thread_position_in_grid]]
) {
    // Bounds check to avoid out-of-bounds access
    if (id >= count) return;
    // Perform the addition: C[id] = A[id] + B[id]
    // This is executed in parallel across all threads
    C[id] = A[id] + B[id];
}