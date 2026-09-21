<#
.SYNOPSIS
    潮声智库 · 一键启动（Docker 基础设施 + 后端 Celery/FastAPI + 前端 Vite）
.DESCRIPTION
    启动链路：
      1) docker compose up -d            —— postgres / neo4j / redis / rustfs
      2) 等待基础设施就绪               —— PG 健康检查 + Redis PING + 端口探测
      3) 幂等应用建表 SQL               —— pgsql/001_init.sql（全 IF NOT EXISTS）
      4) Celery worker                  —— 独立窗口，--pool=solo（Windows 必须）
      5) FastAPI                        —— 独立窗口，:8000
      6) 前端 Vite                      —— 独立窗口，:5173
      7) 轮询 /health 就绪 + 检查 Ollama aemeath 模型
    各后台窗口 PID 记录到 .runtime/dev.pids.json，供 stop.ps1 精准关闭。
    注：生成模型已弃用 serve_amis.py，改走 Ollama 的 aemeath（LLM_URL=:11434）。
.PARAMETER NoDocker
    跳过 docker compose（容器已在运行时用）
.PARAMETER NoMigrate
    跳过建表 SQL 应用
.PARAMETER NoFront
    不启动前端
.PARAMETER Open
    全部就绪后自动打开浏览器 http://localhost:5173
.EXAMPLE
    .\scripts\start.ps1
    .\scripts\start.ps1 -NoDocker -Open
#>
[CmdletBinding()]
param(
    [switch]$NoDocker,
    [switch]$NoMigrate,
    [switch]$NoFront,
    [switch]$Open
)

$ErrorActionPreference = 'Stop'

# ---------- 路径解析（脚本在 scripts/ 下，项目根是其父目录）----------
$Root     = Split-Path -Parent $PSScriptRoot
$LogDir   = Join-Path $Root 'logs'
$RunDir   = Join-Path $Root '.runtime'
$PidFile  = Join-Path $RunDir 'dev.pids.json'   # 运行时状态放 .runtime/，与日志分离
$SqlFile  = Join-Path $Root 'pgsql\001_init.sql'
$FrontDir = Join-Path $Root 'front'

foreach ($d in $LogDir, $RunDir) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

# ---------- 输出助手 ----------
function Write-Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Info([string]$msg) { Write-Host "    $msg" -ForegroundColor Gray }
function Write-Warn2([string]$msg){ Write-Host "    [!] $msg" -ForegroundColor Yellow }
function Write-Err2([string]$msg) { Write-Host "    [X] $msg" -ForegroundColor Red }

# ---------- TCP 端口快速探测（比 Test-NetConnection 快）----------
function Test-TcpPort([string]$ComputerName, [int]$Port, [int]$TimeoutMs = 1500) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $client.BeginConnect($ComputerName, $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
        if ($ok -and $client.Connected) { $client.EndConnect($iar); return $true }
        return $false
    } catch { return $false }
    finally { $client.Close() }
}

# ---------- 轮询等待条件 ----------
function Wait-Until {
    param([scriptblock]$Test, [int]$TimeoutSec = 60, [int]$IntervalMs = 1000, [string]$What = '条件')
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        if (& $Test) { return $true }
        Start-Sleep -Milliseconds $IntervalMs
    }
    return $false
}

# ---------- 安全调用原生命令 ----------
# PowerShell 5.1 陷阱：$ErrorActionPreference='Stop' 下重定向原生命令 stderr（2>$null）
# 会把每行 stderr 包成 ErrorRecord 抛出。这里临时切 Continue 吞掉 stderr，
# 返回合并输出 + 退出码，避免探测类命令（容器未就绪时必写 stderr）导致脚本崩溃。
function Invoke-Native {
    param([scriptblock]$Block)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $Block 2>$null | Out-String
        return [pscustomobject]@{ Output = $out.Trim(); ExitCode = $LASTEXITCODE }
    } finally { $ErrorActionPreference = $prev }
}

# ---------- 后台窗口启动 + PID 记录 ----------
$pids = @()
function Start-ServiceWindow {
    param([string]$Name, [string]$Title, [string]$WorkDir, [string]$Command)
    $full = "`$host.UI.RawUI.WindowTitle='$Title'; cd '$WorkDir'; $Command"
    $p = Start-Process -FilePath 'powershell.exe' `
        -ArgumentList '-NoExit', '-NoProfile', '-Command', $full `
        -WorkingDirectory $WorkDir -PassThru
    $script:pids += [pscustomobject]@{ name = $Name; pid = $p.Id; title = $Title }
    Write-Ok "$Name 已启动（窗口 PID $($p.Id)）"
    return $p.Id
}

# PID 文件落盘（无论中途是否报错，尽量保留已启动的记录）
# 注意：$pids 为空时不能写文件——ConvertTo-Json 会产出空字符串，
# stop.ps1 那边 ConvertFrom-Json 会抛异常；改为清除旧记录。
function Save-Pids {
    if ($pids.Count -eq 0) {
        if (Test-Path $PidFile) { Remove-Item $PidFile -Force -ErrorAction SilentlyContinue }
        return
    }
    # 单元素时 ConvertTo-Json 不输出数组，强制包一层保证 stop.ps1 读回形态一致
    ConvertTo-Json -InputObject @($pids) -Depth 4 | Set-Content -Path $PidFile -Encoding UTF8
}

Write-Host @"

  潮声智库 · 一键启动
  项目根: $Root
"@ -ForegroundColor Magenta

try {
    # ========== 0. 前置检查 ==========
    Write-Step '检查依赖命令'
    foreach ($cmd in 'docker', 'uv', 'npm') {
        if (Get-Command $cmd -ErrorAction SilentlyContinue) { Write-Ok "$cmd 可用" }
        else { Write-Err2 "缺少命令 $cmd，请先安装并加入 PATH"; exit 1 }
    }

    # ========== 1. Docker 基础设施 ==========
    if (-not $NoDocker) {
        Write-Step '启动 Docker 容器（postgres/neo4j/redis/rustfs）'
        $dinfo = Invoke-Native { docker info }
        if ($dinfo.ExitCode -ne 0) {
            Write-Err2 'Docker 守护进程未运行，请先启动 Docker Desktop（或加 -NoDocker 跳过）'
            exit 1
        }
        Push-Location $Root
        try {
            docker compose up -d
            if ($LASTEXITCODE -ne 0) { Write-Err2 'docker compose up 失败'; exit 1 }
        } finally { Pop-Location }
        Write-Ok '容器已拉起'

        Write-Step '等待基础设施就绪'
        # PostgreSQL 健康检查
        $pgReady = Wait-Until -TimeoutSec 90 -What 'PostgreSQL' -Test {
            $s = (Invoke-Native { docker inspect -f '{{.State.Health.Status}}' wuwa-pg }).Output
            return ($s -eq 'healthy')
        }
        if ($pgReady) { Write-Ok 'PostgreSQL healthy' } else { Write-Warn2 'PostgreSQL 健康检查超时，继续尝试' }

        # Redis PING
        $redisReady = Wait-Until -TimeoutSec 30 -What 'Redis' -Test {
            $r = (Invoke-Native { docker exec wuwa-redis redis-cli ping }).Output
            return ($r -eq 'PONG')
        }
        if ($redisReady) { Write-Ok 'Redis PONG' } else { Write-Warn2 'Redis 未就绪' }

        # Neo4j bolt / RustFS 端口
        if (Wait-Until -TimeoutSec 60 -Test { Test-TcpPort 'localhost' 7687 }) { Write-Ok 'Neo4j :7687 可达' } else { Write-Warn2 'Neo4j 端口未就绪' }
        if (Wait-Until -TimeoutSec 40 -Test { Test-TcpPort 'localhost' 9000 }) { Write-Ok 'RustFS :9000 可达' } else { Write-Warn2 'RustFS 端口未就绪' }
    } else {
        Write-Step '跳过 Docker（-NoDocker）'
    }

    # ========== 2. 建表 SQL（幂等）==========
    if (-not $NoMigrate) {
        Write-Step '应用建表 SQL（幂等，可重复执行）'
        if (Test-Path $SqlFile) {
            # 容器名固定 wuwa-pg；-i 保留 stdin，把 SQL 文件喂给 psql
            # ON_ERROR_STOP=0：表/索引已存在时报错也不中断（幂等）
            $mig = Invoke-Native { Get-Content -Raw -Encoding UTF8 $SqlFile |
                docker exec -i wuwa-pg psql -U wuwa -d wuwa -v ON_ERROR_STOP=0 }
            if ($mig.ExitCode -eq 0) { Write-Ok '数据库结构已就绪' }
            else { Write-Warn2 'psql 返回非零（多为表已存在），继续' }
        } else {
            Write-Warn2 "未找到 $SqlFile，跳过建表"
        }
    } else {
        Write-Step '跳过建表（-NoMigrate）'
    }

    # ========== 3. Celery worker ==========
    Write-Step '启动 Celery worker（--pool=solo，Windows 必须）'
    Start-ServiceWindow -Name 'celery' -Title '潮声智库 · Celery Worker' -WorkDir $Root `
        -Command 'uv run celery -A wuwa_rag.worker:celery_app worker --pool=solo --loglevel=info' | Out-Null

    # ========== 4. FastAPI ==========
    Write-Step '启动 FastAPI（:8000）'
    Start-ServiceWindow -Name 'api' -Title '潮声智库 · FastAPI :8000' -WorkDir $Root `
        -Command 'uv run python -m wuwa_rag.api.server' | Out-Null

    # ========== 5. 前端 Vite ==========
    if (-not $NoFront) {
        Write-Step '启动前端 Vite（:5173）'
        if (Test-Path (Join-Path $FrontDir 'node_modules')) {
            Start-ServiceWindow -Name 'front' -Title '潮声智库 · 前端 :5173' -WorkDir $FrontDir `
                -Command 'npm run dev' | Out-Null
        } else {
            Write-Warn2 '前端依赖未安装，先跑 npm install（在新窗口）'
            Start-ServiceWindow -Name 'front' -Title '潮声智库 · 前端 :5173' -WorkDir $FrontDir `
                -Command 'npm install; npm run dev' | Out-Null
        }
    } else {
        Write-Step '跳过前端（-NoFront）'
    }

    Save-Pids

    # ========== 6. 等待 API 就绪 ==========
    Write-Step '等待后端 /health 就绪'
    $apiReady = Wait-Until -TimeoutSec 120 -IntervalMs 1500 -What 'API' -Test {
        try {
            $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 3
            return ($r.StatusCode -eq 200)
        } catch { return $false }
    }
    if ($apiReady) { Write-Ok 'FastAPI /health 返回 200' }
    else { Write-Warn2 'API 未在 120s 内就绪，请看「FastAPI :8000」窗口日志（首次加载模型较慢）' }

    # ========== 7. Ollama 模型检查 ==========
    Write-Step '检查 Ollama 生成模型（aemeath）'
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        # 服务未启动时 ollama list 会写 stderr，必须走 Invoke-Native 才不会被 Stop 模式抛中断
        $ollamaRes = Invoke-Native { ollama list }
        if ($ollamaRes.ExitCode -ne 0) {
            Write-Warn2 "ollama list 执行失败（服务未启动？）：$($ollamaRes.Output)"
            Write-Info '  请启动 Ollama 后重试；问答生成依赖模型 aemeath'
        } elseif ($ollamaRes.Output -match 'aemeath') {
            Write-Ok 'Ollama 已就绪：aemeath'
        } else {
            Write-Warn2 '未在 ollama list 中找到 aemeath，问答生成会失败。请执行：'
            Write-Info '  ollama pull aemeath   （或确认 .env 的 LLM_MODEL / LLM_URL）'
        }
    } else {
        Write-Warn2 '未找到 ollama 命令，若生成模型走 Ollama 请先安装并启动'
    }

    # ========== 8. 汇总 ==========
    Write-Host @"

  ── 全部启动完成 ──────────────────────────────
    前端     http://localhost:5173
    后端 API http://127.0.0.1:8000  (/docs 看接口)
    Ollama   http://localhost:11434
    RustFS   http://localhost:9001  (控制台)
    Neo4j    http://localhost:7474

  停止全部： .\scripts\stop.ps1   （或 .\dev.bat stop）
  PID 记录： $PidFile
"@ -ForegroundColor Green

    if ($Open) {
        Start-Process 'http://localhost:5173'
    }
}
catch {
    Write-Err2 "启动过程出错：$($_.Exception.Message)"
    Save-Pids   # 尽量保存已启动的，方便 stop 清理
    exit 1
}
finally {
    Save-Pids
}
