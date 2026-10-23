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

# 端口从根 .env 现取，脚本不写死数字（理由见 scripts/ports.ps1 头注释）
. (Join-Path $PSScriptRoot 'ports.ps1')
$Ports = Get-WuwaPorts $Root

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

    # 结束进程：**必须先杀整棵树**（taskkill /T）。
    # 曾经的写法是「先 Stop-Process 单杀，杀掉就 return」—— 而杀窗口总是成功，于是
    # 后面那句 taskkill /T 永远不执行：窗口死了，它派生的 uv → celery.exe → python
    # 全部变成孤儿活下来（2026-09-22 实测：两套 Celery 共 8 个进程残留，抢同一个
    # Redis 队列）。现在改为 taskkill /T 优先；仅当它不可用（个别环境 Access denied）
    # 时退回 Stop-Process。
    function Stop-ServiceProcess([int]$TargetPid) {
        taskkill /PID $TargetPid /T /F 2>&1 | Out-Null
        Start-Sleep -Milliseconds 400
        if (-not (Get-Process -Id $TargetPid -ErrorAction SilentlyContinue)) { return $true }
        Stop-Process -Id $TargetPid -Force -ErrorAction SilentlyContinue
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

# 兜底：结束可能残留的独立服务端口占用（端口来自根 .env，按 PID 精准 kill）。
# ⚠️ 「按端口 kill 占用者」本身是有杀伤力的动作：谁占了就算谁，不看是不是本项目。
# 这正是本项目改用 2xxxx 不常用端口的第二个理由（第一个是不与别的项目撞车）——
# 端口越冷门，这里撞上无关进程的概率越低。
# 仍然硬性跳过两类：PID 0/4（System，杀了蓝屏）、com.docker.*（那是 Docker Desktop
# 宿主，它替**所有**容器转发端口，杀掉等于把本机全部容器一起带走，不只是本项目）。
foreach ($port in $Ports.Api, $Ports.Web) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        $ownerPid = $c.OwningProcess
        if (-not $ownerPid -or $ownerPid -le 4) { continue }
        $p = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
        if (-not $p) { continue }
        if ($p.ProcessName -like 'com.docker*') {
            Write-Warn2 "端口 $port 由 $($p.ProcessName) 持有（Docker Desktop 宿主），不动它——容器由下面的 compose stop 处理"
            continue
        }
        Write-Warn2 "端口 $port 仍被 PID $ownerPid（$($p.ProcessName)）占用，结束它"
        Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
    }
}

# 兜底 2：按命令行特征清扫本项目遗留进程。
# 为什么需要：Celery worker **不监听任何端口**，上面的端口兜底抓不到它；一旦它的宿主
# 窗口先死，uv / celery.exe / python 就成了孤儿（2026-09-22 实测曾两套 Celery 并存、
# 抢同一个 Redis 队列）。这里只匹配本项目独有的命令行特征，不会误伤其他进程。
# 第三个特征 `WindowTitle='潮声智库` 匹配的是**宿主窗口自己**：`-NoExit` 的 PowerShell
# 在它的子进程（npm run dev → node）被杀完之后仍会活着变成空壳窗口，几轮 start/stop
# 之后桌面上一排「潮声智库 · 前端」的空窗口，看着像服务还在跑（实测 2026-10-23 抓到）。
$stale = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and (
        $_.CommandLine -like '*wuwa_rag.tasks.worker*' -or
        $_.CommandLine -like '*wuwa_rag.api.server*' -or
        $_.CommandLine -like "*WindowTitle='潮声智库*"
    )
})
if ($stale.Count -gt 0) {
    Write-Warn2 "清扫本项目遗留进程 $($stale.Count) 个"
    foreach ($sp in $stale) {
        Write-Info "  PID $($sp.ProcessId)  $($sp.Name)"
        Stop-Process -Id $sp.ProcessId -Force -ErrorAction SilentlyContinue
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
