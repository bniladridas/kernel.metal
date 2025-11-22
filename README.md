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
❌ No, the software is “as-is”. Contributor & Public FAQ (OPTIONAL — add to README)
