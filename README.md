# opencl_antipattern_bench

Kernel-agnostic OpenCL A/B microbenchmarks for HarmonyOS phones (same push/run flow as `ocr_softmax_bench`).

Use this to verify whether an antipattern **actually hurts on your GPU**, not just in one production kernel.

## Which findings are kernel-agnostic?

| Finding | Agnostic? | Suite (`--only`) | How we test |
|---------|-----------|------------------|-------------|
| Never `fma()` / `mad24()` (breaks SIMD codegen) | Yes | `fma`, `mad24` | Same traffic/FLOPs: plain vs intrinsic |
| Never pointer-cast `float4*` (bypass L2) | Yes | `ptrcast` | `*(float4*)` vs `vload4`/`vstore4` |
| No `prefetch()` on streaming | Yes | `prefetch` | copy ± prefetch |
| No LDS tiling for zero-reuse data | Yes | `lds` | direct stream vs local+barrier, no reuse |
| 8 elem/WI sweet spot | Mostly yes | `elems` | `-DELEMS=1/4/8/16` on same stream kernel |
| Bandwidth plateau (~GB/s) | Yes | `bandwidth` | triad 2R+1W ceiling |

Pixel-pipeline specifics (OCR layout, phase splits) stay in the app kernel; this repo only covers the portable patterns above.

## HarmonyOS: build + run

Needs: DevEco OHOS native SDK, USB phone, `hdc`.

```powershell
cd D:\WorkStation\OptimizationSkill\opencl_antipattern_bench

# 1) Cross-compile arm64 (dlopen libOpenCL.so on device — no NDK OpenCL link required)
.\scripts\build_ohos.ps1

# Optional: pull device libOpenCL.so if you want to link explicitly
# .\scripts\pull_opencl_from_device.ps1
# .\scripts\build_ohos.ps1 -OpenCLLibrary third_party\ohos\libOpenCL.so

# 2) Push binary + .cl and run on phone
.\scripts\run_device.ps1

# Subset / larger buffer:
.\scripts\run_device.ps1 -ExtraArgs "--mb 128 --runs 30 --only ptrcast"
```

Manual:

```powershell
hdc file send build\ohos-ocl-test\ocl_antipattern_bench /data/vendor/camera/
hdc file send kernels\antipatterns.cl /data/vendor/camera/kernels/antipatterns.cl
hdc shell "cd /data/vendor/camera && chmod +x ocl_antipattern_bench && ./ocl_antipattern_bench --kernel kernels/antipatterns.cl --mb 64 --runs 20"
```

## How to read output

```text
copy_vload (GOOD)           3000.0 us   42.0 GB/s
copy_ptrcast (BAD?)         4500.0 us   28.0 GB/s  ratio=1.50x SLOWER
```

- **ratio > ~1.15 on BAD?** → antipattern likely holds on this device  
- **~same** → compiler/driver may have neutralized it; don’t treat as universal rule  
- **bandwidth** GB/s ≈ memory ceiling to compare other suites against  

## Layout

```text
kernels/antipatterns.cl          A/B OpenCL kernels
device_test/ocl_antipattern_bench.cpp
device_test/opencl_dynload.*     OHOS dlopen (from ocr_softmax_bench pattern)
scripts/build_ohos.ps1
scripts/run_device.ps1
third_party/OpenCL-Headers/
```
