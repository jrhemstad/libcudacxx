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

- `<memory_resource>` is the direction taken by Standard C++ for custom, stateful allocators. It will ease working between Standard and CUDA C++ for there to be an allocator interface with a common look and feel. For more information on `<memory_resource>` see [here](https://www.youtube.com/watch?v=l14Zkx5OXr4) and [here](https://www.youtube.com/watch?v=l14Zkx5OXr4).

- The [RAPIDS Memory Management](https://github.com/rapidsai/rmm) library has had three years of [success](https://developer.nvidia.com/blog/fast-flexible-allocation-for-cuda-with-rapids-memory-manager/) using its `rmm::device_memory_resource` interface based on `std::pmr::memory_resource`. 

- Likewise, [Thrust](https://github.com/NVIDIA/thrust) has had similar success with its `thrust::mr::memory_resource` interface. 

Given the direction of Standard C++ and the success of two widely used CUDA libraries with a similar interface, `<memory_resource>` is the logical choice. 

This proposal includes the addition of the following to libcu++:

### `memory_kind` 

A scoped enumerator demarcating the different kinds of dynamically allocated CUDA memory. 
This is intended to be similar to the existing `thread_scope` enum.

```c++
enum class memory_kind {
  device,  ///< Device memory accessible only from device
  unified, ///< Unified memory accessible from both host and device
  pinned,  ///< Page-locked system memory accessible from both host and device
  host     ///< System memory only accessible from host code
};
```

### `stream_view`

A strongly typed, non-owning, view-type for `cudaStream_t`. 
This type provides a more typesafe C++ wrapper around `cudaStream_t` and will serve as the input argument type for any libcu++ API that takes a CUDA stream.

### `cuda::memory_resource`

The `cuda::memory_resource` class template is the abstract base class interface akin to `std::pmr::memory_resource` with two main differences:

1. The `Kind` template parameter determines the `memory_kind` allocated by the resource.

2. The `Context` template parameter determines the "execution context" in which memory allocated by the resource can be accessed without synchronization.
By default, the `Context` is the `any_context` tag type that indicates storage may be accessed immediately on any thread or CUDA stream without synchronization.

```c++
/**
 * @brief Tag type for the default context of `memory_resource`.
 *
 * Default context in which storage may be used immediately on any thread or any
 * CUDA stream without synchronization.
 */
struct any_context{};

template <memory_kind Kind, typename Context = any_context>
class memory_resource{
public:
   void* allocate(size_t n, size_t alignment){ return do_allocate(n, alignment); }
   void deallocate(void * p, size_t n, size_t alignment){ return do_deallocate(p, n, alignment); }
   Context get_context(){ return do_get_context(); }
private:
   virtual void* do_allocate(size_t n, size_t alignment) = 0;
   virtual void do_deallocate(void* p, size_t n, size_t alignment) = 0;
   virtual void do_get_context() = 0;
};
```

The purpose of the `Context` template parameter is to allow for more generic allocation semantics. 
For example, consider a "stream-bound" memory resource where allocated memory may only be accessed without synchronization on a particular stream bound at construction:

```c++
struct stream_context{
    cuda::stream_view s;
};

template <memory_kind Kind>
class stream_bound_memory_resource : public cuda::memory_resource<Kind, stream_context>{
public:
   stream_bound_memory_resource(cuda::stream_view s) : s_{s} {}
private:
   void* do_allocate(size_t n, size_t alignment)  override  { // always allocate on `s` }
   void do_deallocate(void* p, size_t n, size_t alignment) override { // always deallocate on `s` }
   stream_context do_get_context(){ return s_; }
   stream_context s_;
};
```

### `cuda::pmr_adaptor`

`cuda::memory_resource` is similar to `std::pmr::memory_resource`, but they do not share a common inheritance hierarchy, therefore an object that derives from `cuda::memory_resource` cannot be used polymorphically as a `std::pmr::memory_resource`, i.e., a `cuda::memory_resource` derived type cannot be passed to a function that expects a `std::pmr::memory_resource` pointer or reference. 
However, there may be situations where one wishes to use a `cuda::memory_resource` derived type as if it were a `std::pmr::memory_resource` derived type.
The `cuda::pmr_adaptor` class is intended to provide this functionality by inheriting from `std::pmr::memory_resource` and adapting an appropriate `cuda::memory_resource`. 




### `cuda::stream_ordered_memory_resource`

The `cuda::stream_ordered_memory_resource` class template is the abstract base class interface for _stream-ordered_ memory allocation.
This is similar to `cuda::memory_resource` but `allocate` and `deallocate` both take a stream argument and follow stream-ordered memory allocation semantics as defined by [`cudaMallocAsync`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html). 

```c++
template <memory_kind Kind>
class stream_ordered_memory_resource : public memory_resource<_Kind /* default context */>
{
public:
    static constexpr size_t default_alignment = alignof(max_align_t);
    // Two overloads exist so that callers can still implicitly use the `default_alignment` when passing a stream
    void* allocate(size_t n, cuda::stream_view s){ return do_allocate(n, default_alignment, s); }
    void* allocate(size_t n, size_t alignment, cuda::stream_view s){ return do_allocate(n, alignment, s); }
    void deallocate(void* p, size_t n, cuda::stream_view s){ return do_deallocate(p, n, default_alignment, s); }
    void deallocate(void* p, size_t n, size_t alignment, cuda::stream_view s){ return do_deallocate(p, n, alignment, s); }
 private:
    virtual void* do_allocate(size_t n, size_t alignment, cuda::stream_view s) = 0;
    virtual void do_deallocate(void* p, size_t n, size_t alignment, cuda::stream_view s) = 0;
};
```

### Concrete Resource Implementations:

Just as `<memory_resource>` provides concrete, derived implementations of `std::pmr::memory_resource`, libcu++ will provide the following:

- `cuda::new_delete_resource : public cuda::memory_resource<memory_kind::host>`
   - Uses `::operator new()`/`::operator delete()` for allocating host memory
- `cuda::cuda_resource : public cuda::memory_resource<memory_kind::device>`
   - Uses `cudaMalloc/cudaFree` for allocating device memory
- `cuda::unified_resource : public cuda::memory_resource<memory_kind::unified>`
   - Uses `cudaMallocManaged/cudaFree` for unified memory
- `cuda::pinned_resource : public cuda::memory_resource<memory_kind::pinned>`
   - Uses `cudaMallocHost/cudaFreeHost` for page-locked host memory
- `cuda::cuda_async_resource : public cuda::stream_oredered_memory_resource<memory_kind::device>`
   - Uses `cudaMallocAsync/cudaFreeAsync` for device memory

Other resource implementations may be added as deemed appropriate.

### `cuda::polymorphic_allocator`

TBD

### `cuda::stream_ordered_allocator`

TBD










