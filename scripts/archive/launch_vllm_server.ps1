param(
    [string]$Config = "configs/dgx_spark_vllm_qwen7b.yaml"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "../..")
Set-Location $Root

$Python = if ($env:ASTRAKV_PYTHON) { $env:ASTRAKV_PYTHON } else { "python" }
$Model = if ($env:ASTRAKV_MODEL) { $env:ASTRAKV_MODEL } else { "Qwen/Qwen2.5-7B-Instruct" }
$HostName = if ($env:ASTRAKV_HOST) { $env:ASTRAKV_HOST } else { "127.0.0.1" }
$Port = if ($env:ASTRAKV_PORT) { $env:ASTRAKV_PORT } else { "8000" }
$GpuMemory = if ($env:ASTRAKV_GPU_MEMORY_UTILIZATION) { $env:ASTRAKV_GPU_MEMORY_UTILIZATION } else { "0.78" }
$MaxModelLen = if ($env:ASTRAKV_MAX_MODEL_LEN) { $env:ASTRAKV_MAX_MODEL_LEN } else { "8192" }
$TensorParallel = if ($env:ASTRAKV_TENSOR_PARALLEL_SIZE) { $env:ASTRAKV_TENSOR_PARALLEL_SIZE } else { "1" }
$KvTransferConfig = if ($env:ASTRAKV_KV_TRANSFER_CONFIG) { $env:ASTRAKV_KV_TRANSFER_CONFIG } else { "" }

Write-Host "Launching vLLM server from $Config"
Write-Host "Model=$Model Host=$HostName Port=$Port MaxModelLen=$MaxModelLen"

$ArgsList = @(
    "-m", "vllm.entrypoints.openai.api_server",
    "--model", $Model,
    "--host", $HostName,
    "--port", $Port,
    "--dtype", "auto",
    "--gpu-memory-utilization", $GpuMemory,
    "--max-model-len", $MaxModelLen,
    "--tensor-parallel-size", $TensorParallel,
    "--trust-remote-code"
)

if ($KvTransferConfig) {
    Write-Host "KV transfer config=$KvTransferConfig"
    $ArgsList += @("--kv-transfer-config", $KvTransferConfig)
}

& $Python @ArgsList
