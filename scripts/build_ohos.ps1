param(
    [string]$OhosNative = "",
    [string]$OpenCLLibrary = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build\ohos-ocl-test"

if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}

$Headers = Join-Path $Root "third_party\OpenCL-Headers\CL\cl.h"
if (-not (Test-Path $Headers)) {
    Write-Host "Missing: $Headers"
    exit 1
}

$Toolchain = Join-Path $OhosNative "build\cmake\ohos.toolchain.cmake"
$Cmake = Join-Path $OhosNative "build-tools\cmake\bin\cmake.exe"
$Ninja = Join-Path $OhosNative "build-tools\cmake\bin\ninja.exe"

if (-not (Test-Path $Cmake)) {
    Write-Host "DevEco OHOS native SDK not found at: $OhosNative"
    Write-Host "Pass -OhosNative <path>"
    exit 1
}

if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}

$cmakeArgs = @(
    "-G", "Ninja",
    "-S", $Root,
    "-B", $BuildDir,
    "-DCMAKE_MAKE_PROGRAM=$Ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain",
    "-DOHOS_ARCH=arm64-v8a",
    "-DOHOS_PLATFORM=OHOS",
    "-DOHOS_STL=c++_shared",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DOPENCL_HEADERS_DIR=$($Root -replace '\\','/')/third_party/OpenCL-Headers"
)

if (-not [string]::IsNullOrWhiteSpace($OpenCLLibrary)) {
    $cmakeArgs += "-DOPENCL_LIBRARY=$OpenCLLibrary"
}

& $Cmake @cmakeArgs
& $Cmake --build $BuildDir --target ocl_antipattern_bench
Write-Host "Built: $BuildDir\ocl_antipattern_bench"
