# CUDA `<memory_resource>` Extension Proposal

## Motivation

Performance sensitive applications that make frequent dynamic memory allocations often find that allocating memory to be a significant overhead. 
CUDA developers are even more acutely aware of the costs of dynamic allocation due to the relatively higher cost of `cudaMalloc/cudaFree` compared to standard `malloc/free`.
As a result, developers devise custom, high-performance memory allocators as optimized as the application the allocator serves. 
However, what works well for one application will not always satisfy another, which leads to a proliferation of custom allocator implementations. 
Interoperation among these applications is difficult without an interface to enable sharing a common allocator.

In Standard C++, [`Allocator`s](https://en.cppreference.com/w/cpp/named_req/Allocator) have traditionally provided this common interface.
C++17 introduced [`<memory_resource>`](https://en.cppreference.com/w/cpp/header/memory_resource) and the [`std::pmr::memory_resource`](https://en.cppreference.com/w/cpp/memory/memory_resource) abstract class that defines a minimal interface for (de)allocating raw bytes and sits below `Allocator`s. 
This polymorphic interface provides the lingua franca for those who trade in custom memory allocators. 


<!--- In addition, `<memory_resource>` provides a handful of standard `memory_resource` implementations akin to custom allocators that seek to perform better than standard allocation, e.g., [`unsynchronized_pool_resource`](https://en.cppreference.com/w/cpp/memory/unsynchronized_pool_resource). --->

However, the `std::pmr::memory_resource` interface is insufficient to capture the unique features of the CUDA C++ programming model.
For example, Standard C++ only recognizes a single, universally accessible memory space; whereas CUDA C++ applications trade in at least four different kinds of dynamically allocated memory.
Furthermore, CUDA's "stream"-based asynchronous execution model was extended in CUDA 11.2 with the addition of [`cudaMallocAsync`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html)<sup>[1](#link-footnote)</sup> to include memory allocation as stream-ordered events.
Therefore, there is a need for a common allocator interface similar to `std::pmr::memory_resource` that accounts for the unique features of CUDA C++.

<!--- standard device memory, standard pageable host memory, unified memory pageable between host and device, and pinned host memory directly accessible from device --->

<!--- in response to the proliferation of custom CUDA device memory allocators to provide users with an allocator faster than `cudaMalloc/cudaFree` and reduce the burden of many developers to maintain their own allocator. --->


<a name="link-footnote">[1]</a>: Note that `cudaMallocAsync` does not obviate the need for custom, CUDA-aware allocators nor a common allocation interface.
There will never be one allocator that satisfies all users. 
Furthermore, a common interface allows composing and layering utilities like logging, leak checking, tracking, etc. 

## Description

We propose extending `<memory_resource>` to provide a common memory allocation interface that meets the needs of CUDA C++ programmers.

We chose `<memory_resource>` as the basis for a CUDA-specific allocator interface for several reasons:

- `<memory_resource>` is the direction taken by Standard C++ for custom, stateful allocators. It will ease working between Standard and CUDA C++ for there to be an allocator interface with a common look and feel. 

- The [RAPIDS Memory Management](https://github.com/rapidsai/rmm) library has had three years of [success](https://developer.nvidia.com/blog/fast-flexible-allocation-for-cuda-with-rapids-memory-manager/) using its `rmm::device_memory_resource` interface based on `std::pmr::memory_resource`. 

- Likewise, [Thrust](https://github.com/NVIDIA/thrust) has had similar success with its `thrust::mr::memory_resource` interface. 

Given the direction of Standard C++ and the success of two widely used CUDA libraries with a similar interface, `<memory_resource>` is the logical choice. 



