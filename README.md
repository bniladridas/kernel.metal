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
    constant uint &count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    C[id] = A[id] + B[id];
}
```

Host Code (main.swift)
----------------------
```swift
// ... (full code as provided)
```

Sample Output
-------------
```
Elapsed (s): 0.001
Sample results:
C[0] = 0.0
C[1] = 3.0
C[2] = 6.0
C[3] = 9.0
C[4] = 12.0
...
C[999995] = 2999985.0
C[999996] = 2999988.0
C[999997] = 2999991.0
C[999998] = 2999994.0
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

> **⚠️ Important:** GitHub's hosted macOS runners are Intel-based and do not support Metal compute shaders. Full build and test must be performed locally on Apple Silicon hardware or using dedicated macOS runners (e.g., via GitLab CI). Only linting and syncing run on GitHub's hosted runners.

References
==========

- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [Metal Shading Language Guide](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [GPU Compute Best Practices](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu)


