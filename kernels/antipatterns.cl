// Kernel-agnostic OpenCL antipattern microbenchmarks.
// Each pair isolates ONE factor. Host builds with -D variants where noted.

// ---------------------------------------------------------------------------
// 1) fma / mad24 vs plain arithmetic (same FLOPs / same traffic)
// ---------------------------------------------------------------------------
__kernel void saxpy_plain(__global float* y, __global const float* x, const float a, const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    y[i] = a * x[i] + y[i];
}

__kernel void saxpy_fma(__global float* y, __global const float* x, const float a, const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    y[i] = fma(a, x[i], y[i]);
}

// Index gather with mad24 vs int mul+add (stress address codegen)
__kernel void gather_plain(__global float* dst, __global const float* src, const int cols, const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    const int row = i / cols;
    const int col = i - row * cols;
    dst[i] = src[row * cols + col];
}

__kernel void gather_mad24(__global float* dst, __global const float* src, const int cols, const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    const int row = i / cols;
    const int col = i - row * cols;
    dst[i] = src[mad24(row, cols, col)];
}

// ---------------------------------------------------------------------------
// 2) pointer-cast float4 vs vload4/vstore4
// ---------------------------------------------------------------------------
__kernel void copy_ptrcast(__global float* dst, __global const float* src, const int n4) {
    const int i = get_global_id(0);
    if (i >= n4) return;
    *(__global float4*)(dst + (i << 2)) = *(__global const float4*)(src + (i << 2));
}

__kernel void copy_vload(__global float* dst, __global const float* src, const int n4) {
    const int i = get_global_id(0);
    if (i >= n4) return;
    vstore4(vload4(i, src), i, dst);
}

// ---------------------------------------------------------------------------
// 3) prefetch on streaming copy
// ---------------------------------------------------------------------------
__kernel void copy_noprefetch(__global float* dst, __global const float* src, const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    dst[i] = src[i];
}

__kernel void copy_prefetch(__global float* dst, __global const float* src, const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    // Prefetch ahead of the stream; on some mobile GPUs this evicts useful lines.
    prefetch(src + i + 64, 1);
    dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// 4) LDS tiling with zero reuse (+ barrier) vs direct global
// ---------------------------------------------------------------------------
__kernel void stream_direct(__global float* dst, __global const float* src, const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    dst[i] = src[i] * 1.0001f + 0.0001f;
}

__kernel void stream_lds_noreuse(__global float* dst,
                                 __global const float* src,
                                 __local float* tile,
                                 const int n) {
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);
    const int lsz = get_local_size(0);

    if (gid < n) {
        tile[lid] = src[gid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Each WI only reads its own slot — zero cross-WI reuse; barrier still paid.
    if (gid < n) {
        dst[gid] = tile[lid] * 1.0001f + 0.0001f;
    }
    (void)lsz;
}

// ---------------------------------------------------------------------------
// 5) elements per work-item (build with -DELEMS=1|4|8|16)
// ---------------------------------------------------------------------------
#ifndef ELEMS
#define ELEMS 8
#endif

__kernel void stream_elems(__global float* dst, __global const float* src, const int n_elems_total) {
    const int base = get_global_id(0) * ELEMS;
#pragma unroll
    for (int k = 0; k < ELEMS; ++k) {
        const int i = base + k;
        if (i < n_elems_total) {
            dst[i] = src[i] * 1.0001f + 0.0001f;
        }
    }
}

// ---------------------------------------------------------------------------
// 6) bandwidth ceiling — triad (2 reads + 1 write)
// ---------------------------------------------------------------------------
__kernel void triad(__global float* a, __global const float* b, __global const float* c, const float s, const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    a[i] = b[i] + s * c[i];
}
