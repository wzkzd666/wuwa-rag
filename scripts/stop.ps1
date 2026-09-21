<#
.SYNOPSIS
    潮声智库 · 一键停止（关闭 start.ps1 拉起的 Celery / FastAPI / 前端，并停 Docker 容器）
.DESCRIPTION
    读取 .runtime/dev.pids.json 里记录的后台窗口 PID，用 Stop-Process / taskkill 精准结束其进程树
    （PowerShell 宿主 -> uv/npm -> 实际服务）。不碰记录之外的任何进程。
    默认 docker compose stop（保留数据卷），加 -Down 则 docker compose down（同样不删卷）。
.PARAMETER KeepDocker
    只关后端/前端窗口，不动 Docker 容器
.PARAMETER Down
    用 docker compose down 代替 stop（容器删除，数据卷仍保留）
.EXAMPLE
    .\scripts\stop.ps1
    .\scripts\stop.ps1 -KeepDocker
    .\scripts\stop.ps1 -Down
#>
[CmdletBinding()]
param(
    [switch]$KeepDocker,
    [switch]$Down
)

$ErrorActionPreference = 'Continue'

$Root    = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $Root '.runtime\dev.pids.json'

function Write-Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Info([string]$msg) { Write-Host "    $msg" -ForegroundColor Gray }
function Write-Warn2([string]$msg){ Write-Host "    [!] $msg" -ForegroundColor Yellow }

Write-Host "`n  潮声智库 · 一键停止" -ForegroundColor Magenta

# ---------- 1. 关闭后台服务窗口（按 PID 记录）----------
Write-Step '关闭后端 / 前端窗口'
$script:failed = @()   # 停止失败的记录，回写 PID 文件供下次重试
if (Test-Path $PidFile) {
    try {
        $records = Get-Content -Raw -Encoding UTF8 $PidFile | ConvertFrom-Json
    } catch {
        Write-Warn2 "PID 文件解析失败：$($_.Exception.Message)"
        $records = @()
    }

    # 结束进程：优先 PowerShell 原生 Stop-Process（部分环境 taskkill 会 Access denied），
    # 失败再用 taskkill /T /F 兜底连带子进程树。
    function Stop-ServiceProcess([int]$TargetPid) {
        try {
            Stop-Process -Id $TargetPid -Force -ErrorAction Stop
            Start-Sleep -Milliseconds 400
            if (-not (Get-Process -Id $TargetPid -ErrorAction SilentlyContinue)) { return $true }
        } catch { }
        # 兜底：taskkill 连子进程树（uv/npm 派生的实际服务进程）
        taskkill /PID $TargetPid /T /F 2>&1 | Out-Null
        Start-Sleep -Milliseconds 400
        return (-not (Get-Process -Id $TargetPid -ErrorAction SilentlyContinue))
    }

    if (-not $records) {
        Write-Info '没有记录到运行中的服务窗口'
    } else {
        foreach ($r in $records) {
            $procId = $r.pid
            $name = $r.name
            if (-not $procId) { continue }
            $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
            if ($null -eq $proc) {
                Write-Info "$name（PID $procId）已不在运行"
                continue
            }
            if (Stop-ServiceProcess -TargetPid $procId) { Write-Ok "$name 已停止（PID $procId）" }
            else {
                Write-Warn2 "$name 停止失败（PID $procId），可手动关闭窗口「$($r.title)」"
                $script:failed += $r   # 保留失败项，下次 stop 可重试
            }
        }
    }

    # 只有全部成功才删 PID 文件；有失败项则回写剩余记录，避免丢失线索
    if ($script:failed.Count -gt 0) {
        ConvertTo-Json -InputObject @($script:failed) -Depth 4 |
            Set-Content -Path $PidFile -Encoding UTF8
        Write-Warn2 "有 $($script:failed.Count) 个服务未停止，记录已保留在 PID 文件中"
    } else {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        Write-Info '已清理 PID 记录文件'
    }
} else {
    Write-Warn2 "未找到 $PidFile —— 服务可能不是由 start.ps1 启动的，请手动关闭对应窗口"
}

# 兜底：结束可能残留的独立服务端口占用（仅本项目已知端口，按 PID 精准 kill）
foreach ($port in 8000, 5173) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        $ownerPid = $c.OwningProcess
        if ($ownerPid -and $ownerPid -ne 0) {
            $p = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
            if ($p) {
                Write-Warn2 "端口 $port 仍被 PID $ownerPid（$($p.ProcessName)）占用，结束它"
                Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

# ---------- 2. Docker 容器 ----------
if ($KeepDocker) {
    Write-Step '保留 Docker 容器（-KeepDocker）'
} else {
    Write-Step '停止 Docker 容器（数据卷保留）'
    Push-Location $Root
    try {
        if ($Down) {
            docker compose down
            if ($LASTEXITCODE -eq 0) { Write-Ok 'docker compose down 完成（卷已保留）' }
            else { Write-Warn2 'docker compose down 返回非零' }
        } else {
            docker compose stop
            if ($LASTEXITCODE -eq 0) { Write-Ok 'docker compose stop 完成' }
            else { Write-Warn2 'docker compose stop 返回非零' }
        }
    } finally { Pop-Location }
}

Write-Host "`n  已全部停止。重新启动： .\dev.bat`n" -ForegroundColor Green
