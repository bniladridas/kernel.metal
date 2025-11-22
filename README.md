Metal Vector Addition Compute Kernel
========================================

Performing large-scale vector operations on the CPU can be slow and inefficient for data-intensive tasks. Utilizing the GPU via Metal compute shaders on Apple Silicon enables parallel processing, significantly accelerating computations like vector addition.

Recommendation
==============

Use Metal compute kernels for parallelizable operations to leverage GPU performance. Ensure proper thread mapping and buffer management for optimal results.

Example
=======

This project adds two vectors of 1,000,000 floats (A[i] = i, B[i] = 2*i) to produce C[i] = 3*i.

Kernel Code (kernel.metal)
--------------------------
```metal
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
```

Host Code (main.swift)
----------------------
```swift
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
```

Sample Output
-------------
```
Sample results:
C[0] = 0.0
C[1] = 3.0
...
C[999999] = 2999997.0
```

Local Build and Run
===================

To build and run locally on Apple Silicon (M1/M2):

1. Install dependencies: `./install.sh`
2. Compile: `swiftc main.swift -framework Metal -o vector_add`
3. Run: `./vector_add`

This performs vector addition on 1,000,000 floats using GPU compute.

CI/CD
=====

- **GitHub Actions**: Runs linting on ubuntu-latest and syncs to GitLab.
- **GitLab CI**: Runs full build and test on macOS for Apple Silicon compatibility.
- **CircleCI**: Runs validation on Ubuntu (file checks, actionlint).
- **Local CI**: Use `act` for GitHub Actions simulation, `gitlab-ci-local` for GitLab CI, or CircleCI local CLI.

Note: GitHub's hosted macOS runners are Intel-based and don't support Metal. Build/test are local or on GitLab. Lint and sync work on hosted runners.

References
==========

- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [Metal Shading Language Guide](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [GPU Compute Best Practices](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu)

## 📄 License & Usage FAQ

**Can I use this software in my open source project?**  
✔ Yes, as long as it is **non-commercial** and credit is maintained.

**Can I modify the code for personal or academic use?**  
✔ Yes — document any changes.

**Can I publish a modified version on GitHub?**  
✔ Yes, but still **non-commercial only** and must keep the license.

**I want to use this in a product or service that earns money.**  
➡ You must request commercial permission.  
Create a **Commercial License Request** issue.

**Does the author provide warranties?**  
❌ No, the software is “as-is”.
