// OpenCL antipattern microbenchmarks for HarmonyOS / desktop.
// Isolates kernel-agnostic A/B pairs with CL event profiling.

#include <CL/cl.h>
#ifdef OCR_OPENCL_DLOPEN
#include "opencl_dynload.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

#define OCL_CHECK(call, msg)                                                                       \
    do {                                                                                           \
        cl_int _err = (call);                                                                      \
        if (_err != CL_SUCCESS) {                                                                  \
            std::fprintf(stderr, "OpenCL error %d at %s:%d: %s\n", _err, __FILE__, __LINE__, msg); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

struct Args {
    std::string kernel_path = "kernels/antipatterns.cl";
    int runs = 20;
    int warmup = 3;
    size_t nbytes = 64ull << 20;  // 64 MiB default
    int local_size = 256;
    std::string only;  // empty = all suites
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--kernel")) {
            a.kernel_path = need("--kernel");
        } else if (!std::strcmp(argv[i], "--runs")) {
            a.runs = std::atoi(need("--runs"));
        } else if (!std::strcmp(argv[i], "--warmup")) {
            a.warmup = std::atoi(need("--warmup"));
        } else if (!std::strcmp(argv[i], "--mb")) {
            a.nbytes = static_cast<size_t>(std::atoi(need("--mb"))) << 20;
        } else if (!std::strcmp(argv[i], "--local")) {
            a.local_size = std::atoi(need("--local"));
        } else if (!std::strcmp(argv[i], "--only")) {
            a.only = need("--only");
        } else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            std::printf(
                "Usage: %s [--kernel PATH] [--runs N] [--warmup N] [--mb MiB] [--local N]\n"
                "          [--only fma|mad24|ptrcast|prefetch|lds|elems|bandwidth]\n",
                argv[0]);
            std::exit(0);
        }
    }
    return a;
}

std::string readText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "Cannot open %s\n", path.c_str());
        std::exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

cl_device_id pickDevice(cl_platform_id platform) {
    cl_device_id dev = nullptr;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr) != CL_SUCCESS) {
        OCL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, nullptr), "clGetDeviceIDs");
    }
    return dev;
}

std::string deviceName(cl_device_id dev) {
    char buf[256] = {};
    OCL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(buf), buf, nullptr), "CL_DEVICE_NAME");
    return buf;
}

double profileUs(cl_event ev) {
    cl_ulong t0 = 0, t1 = 0;
    OCL_CHECK(clWaitForEvents(1, &ev), "clWaitForEvents");
    OCL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr), "START");
    OCL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr), "END");
    return static_cast<double>(t1 - t0) / 1e3;  // us
}

cl_program buildProgram(cl_context ctx, cl_device_id dev, const std::string& src, const char* opts) {
    cl_int err = CL_SUCCESS;
    const char* p = src.c_str();
    const size_t len = src.size();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &p, &len, &err);
    OCL_CHECK(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        std::vector<char> log(log_sz + 1, 0);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
        std::fprintf(stderr, "Build failed (%d) opts='%s':\n%s\n", err, opts ? opts : "", log.data());
        std::exit(1);
    }
    return prog;
}

struct Timed {
    double us = 0.0;
    double gbs = 0.0;  // effective traffic estimate
};

Timed timeKernel(cl_command_queue queue, cl_kernel kernel, size_t global, size_t local, int warmup,
                 int runs, double bytes_moved) {
    const size_t* local_ptr = local ? &local : nullptr;
    for (int i = 0; i < warmup; ++i) {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, local_ptr, 0, nullptr, &ev),
                  "warmup");
        OCL_CHECK(clWaitForEvents(1, &ev), "warmup wait");
        clReleaseEvent(ev);
    }
    double sum = 0.0;
    for (int i = 0; i < runs; ++i) {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, local_ptr, 0, nullptr, &ev),
                  "kernel");
        sum += profileUs(ev);
        clReleaseEvent(ev);
    }
    Timed t;
    t.us = sum / runs;
    t.gbs = (bytes_moved / 1e9) / (t.us * 1e-6);
    return t;
}

void printRow(const char* name, const Timed& t, const Timed* baseline) {
    if (baseline) {
        const double ratio = t.us / baseline->us;
        std::printf("  %-28s %10.1f us  %7.1f GB/s  ratio=%5.2fx %s\n", name, t.us, t.gbs, ratio,
                    ratio > 1.15 ? "SLOWER" : (ratio < 0.87 ? "FASTER" : "~same"));
    } else {
        std::printf("  %-28s %10.1f us  %7.1f GB/s\n", name, t.us, t.gbs);
    }
}

bool want(const Args& a, const char* suite) {
    return a.only.empty() || a.only == suite;
}

cl_kernel makeKernel(cl_program prog, const char* name) {
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(prog, name, &err);
    OCL_CHECK(err, name);
    return k;
}

void fillHost(std::vector<float>& v, unsigned seed) {
    for (size_t i = 0; i < v.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        v[i] = static_cast<float>((seed >> 9) & 0xffff) / 65536.f;
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);

#ifdef OCR_OPENCL_DLOPEN
    if (!opencl_load()) {
        std::fprintf(stderr, "Failed to dlopen OpenCL\n");
        return 1;
    }
#endif

    cl_uint np = 0;
    OCL_CHECK(clGetPlatformIDs(0, nullptr, &np), "platforms");
    if (np == 0) {
        std::fprintf(stderr, "No OpenCL platforms\n");
        return 1;
    }
    std::vector<cl_platform_id> plats(np);
    OCL_CHECK(clGetPlatformIDs(np, plats.data(), nullptr), "platforms");
    cl_device_id dev = pickDevice(plats[0]);
    std::printf("device: %s\n", deviceName(dev).c_str());
    std::printf("buffer: %.1f MiB  local=%d  warmup=%d runs=%d\n", args.nbytes / (1024.0 * 1024.0),
                args.local_size, args.warmup, args.runs);

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    OCL_CHECK(err, "context");
    cl_command_queue queue =
        clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    OCL_CHECK(err, "queue");

    const std::string src = readText(args.kernel_path);
    cl_program prog = buildProgram(ctx, dev, src, "");

    const size_t n = args.nbytes / sizeof(float);
    const size_t n4 = n / 4;
    std::vector<float> ha(n), hb(n), hc(n);
    fillHost(ha, 1);
    fillHost(hb, 2);
    fillHost(hc, 3);

    cl_mem buf_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, args.nbytes, ha.data(), &err);
    OCL_CHECK(err, "buf_a");
    cl_mem buf_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, args.nbytes, hb.data(), &err);
    OCL_CHECK(err, "buf_b");
    cl_mem buf_c = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, args.nbytes, hc.data(), &err);
    OCL_CHECK(err, "buf_c");

    const size_t local = static_cast<size_t>(args.local_size);
    auto align_global = [&](size_t g) {
        return ((g + local - 1) / local) * local;
    };

    std::printf("\n=== Antipattern microbench (higher us / lower GB/s = worse) ===\n");

    // ----- fma -----
    if (want(args, "fma")) {
        std::printf("\n[1] fma() vs plain mul+add  (bytes~2R+1W)\n");
        cl_kernel k_plain = makeKernel(prog, "saxpy_plain");
        cl_kernel k_fma = makeKernel(prog, "saxpy_fma");
        const float a = 1.0001f;
        const int ni = static_cast<int>(n);
        OCL_CHECK(clSetKernelArg(k_plain, 0, sizeof(cl_mem), &buf_a), "p0");
        OCL_CHECK(clSetKernelArg(k_plain, 1, sizeof(cl_mem), &buf_b), "p1");
        OCL_CHECK(clSetKernelArg(k_plain, 2, sizeof(float), &a), "p2");
        OCL_CHECK(clSetKernelArg(k_plain, 3, sizeof(int), &ni), "p3");
        OCL_CHECK(clSetKernelArg(k_fma, 0, sizeof(cl_mem), &buf_a), "f0");
        OCL_CHECK(clSetKernelArg(k_fma, 1, sizeof(cl_mem), &buf_b), "f1");
        OCL_CHECK(clSetKernelArg(k_fma, 2, sizeof(float), &a), "f2");
        OCL_CHECK(clSetKernelArg(k_fma, 3, sizeof(int), &ni), "f3");
        const size_t g = align_global(n);
        const double bytes = 3.0 * args.nbytes;
        Timed t0 = timeKernel(queue, k_plain, g, local, args.warmup, args.runs, bytes);
        Timed t1 = timeKernel(queue, k_fma, g, local, args.warmup, args.runs, bytes);
        printRow("saxpy_plain (GOOD)", t0, nullptr);
        printRow("saxpy_fma (BAD?)", t1, &t0);
        clReleaseKernel(k_plain);
        clReleaseKernel(k_fma);
    }

    // ----- mad24 -----
    if (want(args, "mad24")) {
        std::printf("\n[2] mad24() vs plain index  (gather, cols=1024)\n");
        cl_kernel k_plain = makeKernel(prog, "gather_plain");
        cl_kernel k_mad = makeKernel(prog, "gather_mad24");
        const int cols = 1024;
        const int ni = static_cast<int>(n);
        OCL_CHECK(clSetKernelArg(k_plain, 0, sizeof(cl_mem), &buf_a), "g0");
        OCL_CHECK(clSetKernelArg(k_plain, 1, sizeof(cl_mem), &buf_b), "g1");
        OCL_CHECK(clSetKernelArg(k_plain, 2, sizeof(int), &cols), "g2");
        OCL_CHECK(clSetKernelArg(k_plain, 3, sizeof(int), &ni), "g3");
        OCL_CHECK(clSetKernelArg(k_mad, 0, sizeof(cl_mem), &buf_a), "m0");
        OCL_CHECK(clSetKernelArg(k_mad, 1, sizeof(cl_mem), &buf_b), "m1");
        OCL_CHECK(clSetKernelArg(k_mad, 2, sizeof(int), &cols), "m2");
        OCL_CHECK(clSetKernelArg(k_mad, 3, sizeof(int), &ni), "m3");
        const size_t g = align_global(n);
        const double bytes = 2.0 * args.nbytes;
        Timed t0 = timeKernel(queue, k_plain, g, local, args.warmup, args.runs, bytes);
        Timed t1 = timeKernel(queue, k_mad, g, local, args.warmup, args.runs, bytes);
        printRow("gather_plain (GOOD)", t0, nullptr);
        printRow("gather_mad24 (BAD?)", t1, &t0);
        clReleaseKernel(k_plain);
        clReleaseKernel(k_mad);
    }

    // ----- ptrcast -----
    if (want(args, "ptrcast")) {
        std::printf("\n[3] pointer-cast float4 vs vload4/vstore4\n");
        cl_kernel k_cast = makeKernel(prog, "copy_ptrcast");
        cl_kernel k_vload = makeKernel(prog, "copy_vload");
        const int n4i = static_cast<int>(n4);
        OCL_CHECK(clSetKernelArg(k_cast, 0, sizeof(cl_mem), &buf_a), "c0");
        OCL_CHECK(clSetKernelArg(k_cast, 1, sizeof(cl_mem), &buf_b), "c1");
        OCL_CHECK(clSetKernelArg(k_cast, 2, sizeof(int), &n4i), "c2");
        OCL_CHECK(clSetKernelArg(k_vload, 0, sizeof(cl_mem), &buf_a), "v0");
        OCL_CHECK(clSetKernelArg(k_vload, 1, sizeof(cl_mem), &buf_b), "v1");
        OCL_CHECK(clSetKernelArg(k_vload, 2, sizeof(int), &n4i), "v2");
        const size_t g = align_global(n4);
        const double bytes = 2.0 * (n4 * 4 * sizeof(float));
        Timed t_bad = timeKernel(queue, k_cast, g, local, args.warmup, args.runs, bytes);
        Timed t_good = timeKernel(queue, k_vload, g, local, args.warmup, args.runs, bytes);
        printRow("copy_vload (GOOD)", t_good, nullptr);
        printRow("copy_ptrcast (BAD?)", t_bad, &t_good);
        clReleaseKernel(k_cast);
        clReleaseKernel(k_vload);
    }

    // ----- prefetch -----
    if (want(args, "prefetch")) {
        std::printf("\n[4] prefetch() on streaming copy\n");
        cl_kernel k0 = makeKernel(prog, "copy_noprefetch");
        cl_kernel k1 = makeKernel(prog, "copy_prefetch");
        const int ni = static_cast<int>(n);
        OCL_CHECK(clSetKernelArg(k0, 0, sizeof(cl_mem), &buf_a), "0");
        OCL_CHECK(clSetKernelArg(k0, 1, sizeof(cl_mem), &buf_b), "1");
        OCL_CHECK(clSetKernelArg(k0, 2, sizeof(int), &ni), "2");
        OCL_CHECK(clSetKernelArg(k1, 0, sizeof(cl_mem), &buf_a), "0");
        OCL_CHECK(clSetKernelArg(k1, 1, sizeof(cl_mem), &buf_b), "1");
        OCL_CHECK(clSetKernelArg(k1, 2, sizeof(int), &ni), "2");
        const size_t g = align_global(n);
        const double bytes = 2.0 * args.nbytes;
        Timed t0 = timeKernel(queue, k0, g, local, args.warmup, args.runs, bytes);
        Timed t1 = timeKernel(queue, k1, g, local, args.warmup, args.runs, bytes);
        printRow("copy_noprefetch (GOOD)", t0, nullptr);
        printRow("copy_prefetch (BAD?)", t1, &t0);
        clReleaseKernel(k0);
        clReleaseKernel(k1);
    }

    // ----- lds zero reuse -----
    if (want(args, "lds")) {
        std::printf("\n[5] LDS tile + barrier with zero reuse vs direct\n");
        cl_kernel k0 = makeKernel(prog, "stream_direct");
        cl_kernel k1 = makeKernel(prog, "stream_lds_noreuse");
        const int ni = static_cast<int>(n);
        OCL_CHECK(clSetKernelArg(k0, 0, sizeof(cl_mem), &buf_a), "0");
        OCL_CHECK(clSetKernelArg(k0, 1, sizeof(cl_mem), &buf_b), "1");
        OCL_CHECK(clSetKernelArg(k0, 2, sizeof(int), &ni), "2");
        OCL_CHECK(clSetKernelArg(k1, 0, sizeof(cl_mem), &buf_a), "0");
        OCL_CHECK(clSetKernelArg(k1, 1, sizeof(cl_mem), &buf_b), "1");
        OCL_CHECK(clSetKernelArg(k1, 2, local * sizeof(float), nullptr), "local");
        OCL_CHECK(clSetKernelArg(k1, 3, sizeof(int), &ni), "3");
        const size_t g = align_global(n);
        const double bytes = 2.0 * args.nbytes;
        Timed t0 = timeKernel(queue, k0, g, local, args.warmup, args.runs, bytes);
        Timed t1 = timeKernel(queue, k1, g, local, args.warmup, args.runs, bytes);
        printRow("stream_direct (GOOD)", t0, nullptr);
        printRow("stream_lds_noreuse (BAD?)", t1, &t0);
        clReleaseKernel(k0);
        clReleaseKernel(k1);
    }

    // ----- elems/WI -----
    if (want(args, "elems")) {
        std::printf("\n[6] elements per work-item (1/4/8/16)\n");
        const int widths[] = {1, 4, 8, 16};
        Timed best{};
        best.us = 1e300;
        int best_w = 0;
        for (int w : widths) {
            char opts[64];
            std::snprintf(opts, sizeof(opts), "-DELEMS=%d", w);
            cl_program p = buildProgram(ctx, dev, src, opts);
            cl_kernel k = makeKernel(p, "stream_elems");
            const int ni = static_cast<int>(n);
            OCL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &buf_a), "0");
            OCL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &buf_b), "1");
            OCL_CHECK(clSetKernelArg(k, 2, sizeof(int), &ni), "2");
            const size_t g = align_global((n + static_cast<size_t>(w) - 1) / static_cast<size_t>(w));
            const double bytes = 2.0 * args.nbytes;
            Timed t = timeKernel(queue, k, g, local, args.warmup, args.runs, bytes);
            char name[32];
            std::snprintf(name, sizeof(name), "ELEMS=%d", w);
            if (best_w == 0) {
                printRow(name, t, nullptr);
                best = t;
                best_w = w;
            } else {
                printRow(name, t, &best);
                if (t.us < best.us) {
                    best = t;
                    best_w = w;
                }
            }
            clReleaseKernel(k);
            clReleaseProgram(p);
        }
        std::printf("  -> fastest on this device: ELEMS=%d\n", best_w);
    }

    // ----- bandwidth -----
    if (want(args, "bandwidth")) {
        std::printf("\n[7] bandwidth ceiling (triad 2R+1W)\n");
        cl_kernel k = makeKernel(prog, "triad");
        const float s = 1.0001f;
        const int ni = static_cast<int>(n);
        OCL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &buf_a), "0");
        OCL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &buf_b), "1");
        OCL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &buf_c), "2");
        OCL_CHECK(clSetKernelArg(k, 3, sizeof(float), &s), "3");
        OCL_CHECK(clSetKernelArg(k, 4, sizeof(int), &ni), "4");
        const size_t g = align_global(n);
        const double bytes = 3.0 * args.nbytes;
        Timed t = timeKernel(queue, k, g, local, args.warmup, args.runs, bytes);
        printRow("triad", t, nullptr);
        std::printf("  -> treat this GB/s as memory-system reference plateau\n");
        clReleaseKernel(k);
    }

    std::printf("\nDone. ratio>1.15 on BAD? row => antipattern likely holds on this GPU.\n");

    clReleaseMemObject(buf_a);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_c);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
