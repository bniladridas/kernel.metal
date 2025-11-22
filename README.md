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
// ... (full code as provided)
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

References
==========

- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [Metal Shading Language Guide](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [GPU Compute Best Practices](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu)
