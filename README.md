# msml
MSML is a minimalistic machine learning library written in C99, designed for speed, flexibility, and ease of integration.<br>
With no runtime allocations and lightweight architecture, MSML is perfect for efficient model training and inference.<br>
Python bindings make it accessible for both C and Python developers.<br>

### Key Features
* Zero Runtime Allocations: Memory pools handle everything, optimizing performance during training and inference.
* Efficient Tensor Operations, optimized with SIMD intrinsics.
* Runtime CPU Detection: Detects your CPU and uses the fastest instructions available
* Portable: Optimized for different platforms.
* Built to be simple, fast, and easily embeddable in any project where performance matters.
* PRNG Algorithms: Supports Mersenne Twister and PCG for fast random number generation.
* Single-file Design: No third-party dependencies, just copy the header and source file into your project.
* Python Bindings: Seamless Python integration with native C performance.
