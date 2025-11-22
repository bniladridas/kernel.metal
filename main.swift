// SPDX-License-Identifier: LicenseRef-KERNELMETAL-NC
// Copyright (c) 2025 KERNEL.METAL (harpertoken)

import Metal
import Foundation

// This is the main Swift file for the host code that sets up and runs the Metal compute kernel.
// It initializes Metal, creates buffers, dispatches the kernel, and reads back results.

// Get the default Metal device (the GPU on Apple Silicon)
guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this device")
}

// Define the Metal shader source code as a string
// This includes the kernel function for vector addition
let source = """
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    C[id] = A[id] + B[id];
}
"""

// Create a Metal library from the source code
let library: MTLLibrary
do {
    library = try device.makeLibrary(source: source, options: nil)
} catch {
    fatalError("Failed to create Metal library: \(error)")
}

// Get the compute function named "vector_add" from the library
guard let function = library.makeFunction(name: "vector_add") else {
    fatalError("Failed to find function")
}

// Create the compute pipeline state using the function
let pipelineState: MTLComputePipelineState
do {
    pipelineState = try device.makeComputePipelineState(function: function)
} catch {
    fatalError("Failed to create pipeline state: \(error)")
}

// Define the problem size: 1,000,000 floats
let size = 1_000_000

// Calculate buffer size in bytes
let bufferSize = size * MemoryLayout<Float>.size

// Create three buffers: A, B, C
// Using .storageModeShared so CPU and GPU can access them
guard let bufferA = device.makeBuffer(length: bufferSize, options: .storageModeShared),
      let bufferB = device.makeBuffer(length: bufferSize, options: .storageModeShared),
      let bufferC = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
    fatalError("Failed to create buffers")
}

// Fill input buffers A and B with data
// A[i] = i, B[i] = i * 2, so C[i] should be i + 2*i = 3*i
let pointerA = bufferA.contents().bindMemory(to: Float.self, capacity: size)
let pointerB = bufferB.contents().bindMemory(to: Float.self, capacity: size)
for i in 0..<size {
    pointerA[i] = Float(i)
    pointerB[i] = Float(i * 2)
}

// Create a command queue for submitting commands to the GPU
guard let commandQueue = device.makeCommandQueue() else {
    fatalError("Failed to create command queue")
}

// Create a command buffer to hold the commands
guard let commandBuffer = commandQueue.makeCommandBuffer() else {
    fatalError("Failed to create command buffer")
}

// Create a compute command encoder to encode compute commands
guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
    fatalError("Failed to create compute encoder")
}

// Set the compute pipeline state on the encoder
computeEncoder.setComputePipelineState(pipelineState)

// Set the buffers on the encoder at their respective indices
computeEncoder.setBuffer(bufferA, offset: 0, index: 0)
computeEncoder.setBuffer(bufferB, offset: 0, index: 1)
computeEncoder.setBuffer(bufferC, offset: 0, index: 2)

// Define the thread execution configuration
// threadsPerGrid: total number of threads to execute (1,000,000 in 1D)
let threadsPerGrid = MTLSize(width: size, height: 1, depth: 1)
// threadsPerThreadgroup: number of threads per threadgroup (256 threads)
let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)

// Dispatch the threads to execute the kernel
computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

// End the compute encoding
computeEncoder.endEncoding()

// Commit the command buffer to execute on the GPU
commandBuffer.commit()
// Wait for the GPU to finish execution
commandBuffer.waitUntilCompleted()

// Read back the results from buffer C
let pointerC = bufferC.contents().bindMemory(to: Float.self, capacity: size)

// Print sample output to console
print("Sample results:")
// Print first 10 elements
for i in 0..<10 {
    print("C[\(i)] = \(pointerC[i])")
}
print("...")
// Print last 10 elements
for i in (size-10)..<size {
    print("C[\(i)] = \(pointerC[i])")
}