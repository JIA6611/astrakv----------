param(
    [ValidateSet("cpu", "disk")]
    [string]$Backend = "cpu"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "../..")
Set-Location $Root

$Config = if ($Backend -eq "disk") { "configs/lmcache_disk_example.yaml" } else { "configs/lmcache_cpu_example.yaml" }
$env:LMCACHE_CONFIG_FILE = $Config
$env:ASTRAKV_MODEL = if ($env:ASTRAKV_MODEL) { $env:ASTRAKV_MODEL } else { "Qwen/Qwen2.5-7B-Instruct" }
if (-not $env:ASTRAKV_KV_TRANSFER_CONFIG) {
    $env:ASTRAKV_KV_TRANSFER_CONFIG = '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
}
if ($Backend -eq "cpu" -and -not $env:ASTRAKV_GPU_MEMORY_UTILIZATION) {
    $env:ASTRAKV_GPU_MEMORY_UTILIZATION = "0.72"
}
if ($Backend -eq "disk") {
    if (-not $env:ASTRAKV_GPU_MEMORY_UTILIZATION) {
        $env:ASTRAKV_GPU_MEMORY_UTILIZATION = "0.72"
    }
    $DiskPath = if ($env:LMCACHE_DISK_PATH) { $env:LMCACHE_DISK_PATH } else { "results/lmcache_disk_store" }
    New-Item -ItemType Directory -Force -Path $DiskPath | Out-Null
} else {
    $DiskPath = ""
}

Write-Host "Launching vLLM with LMCache backend intent: $Backend"
Write-Host "LMCACHE_CONFIG_FILE=$env:LMCACHE_CONFIG_FILE"
if ($DiskPath) {
    Write-Host "LMCache disk path=$DiskPath"
}
Write-Host "ASTRAKV_KV_TRANSFER_CONFIG=$env:ASTRAKV_KV_TRANSFER_CONFIG"
Write-Host "Verify the installed LMCache/vLLM versions accept this connector before official runs."

$LauncherConfig = if ($Backend -eq "disk") { "configs/dgx_spark_lmcache_disk.yaml" } else { "configs/dgx_spark_lmcache_cpu.yaml" }
& (Join-Path $PSScriptRoot "launch_vllm_server.ps1") -Config $LauncherConfig
