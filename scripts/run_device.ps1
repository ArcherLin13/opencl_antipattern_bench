param(
    [string]$OhosNative = "",
    [string]$RemoteDir = "/data/vendor/camera",
    [string]$ExtraArgs = "--mb 64 --runs 20"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "build\ohos-ocl-test\ocl_antipattern_bench"
$Kernel = Join-Path $Root "kernels\antipatterns.cl"

if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"

foreach ($f in @($Exe, $Kernel)) {
    if (-not (Test-Path $f)) {
        Write-Host "Missing: $f"
        Write-Host "Build first: .\scripts\build_ohos.ps1"
        exit 1
    }
}

if (-not (Test-Path $Hdc)) {
    Write-Host "hdc not found near SDK. Tried: $Hdc"
    exit 1
}

$targets = & $Hdc list targets 2>&1 | Out-String
if ($targets -match "Empty") {
    Write-Host "No hdc device connected."
    exit 1
}

& $Hdc shell "mkdir -p $RemoteDir/kernels"
& $Hdc file send $Exe "$RemoteDir/ocl_antipattern_bench"
& $Hdc file send $Kernel "$RemoteDir/kernels/antipatterns.cl"

$cmd = "cd $RemoteDir && chmod +x ocl_antipattern_bench && ./ocl_antipattern_bench --kernel kernels/antipatterns.cl $ExtraArgs"
Write-Host $cmd
& $Hdc shell $cmd
