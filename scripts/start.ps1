<#
.SYNOPSIS
    潮声智库 · 一键启动（Docker 基础设施 + 后端 Celery/FastAPI + 前端 Vite）
.DESCRIPTION
    启动链路：
      1) docker compose up -d            —— postgres / neo4j / redis / rustfs
      2) 等待基础设施就绪               —— PG 健康检查 + Redis PING + 端口探测
      3) 幂等应用建表 SQL               —— pgsql/001_init.sql（全 IF NOT EXISTS）
      4) Celery worker                  —— 独立窗口，--pool=solo（Windows 必须）
      5) FastAPI                        —— 独立窗口（端口 = .env 的 API_PORT）
      6) 等待 /health 200 后再起前端      —— 顺序反了会刷一片 ECONNREFUSED
      7) 前端 Vite                      —— 独立窗口（端口 = .env 的 WEB_PORT）
      7) 轮询 /health 就绪 + 检查 Ollama aemeath 模型
    各后台窗口 PID 记录到 .runtime/dev.pids.json，供 stop.ps1 精准关闭。
    ⚠️ 本脚本**不写死任何端口**：一律从根 .env 读（见 scripts/ports.ps1）。
       写死的下场是「改了 .env 忘了改脚本」——探测的是旧端口，表现成启动超时，
       而真实服务其实好着，排查方向会被整个带偏。
    生成模型由 Ollama 承载（.env 的 LLM_URL，默认 11434，模型名 aemeath），
    无需另起本地推理服务。Ollama 是外部依赖，端口不由本项目发布，故不改它。
.PARAMETER NoDocker
    跳过 docker compose（容器已在运行时用）
.PARAMETER NoMigrate
    跳过建表 SQL 应用
.PARAMETER NoFront
    不启动前端
.PARAMETER Open
    全部就绪后自动打开浏览器（前端地址，端口 = 根 .env 的 WEB_PORT）
.EXAMPLE
    .\scripts\start.ps1
    .\scripts\start.ps1 -NoDocker -Open
#>
[CmdletBinding()]
param(
    [switch]$NoDocker,
    [switch]$NoMigrate,
    [switch]$NoFront,
    [switch]$Open,
    # 默认**不弹控制台窗口**：celery / api / 前端 三个进程在隐藏窗口里跑，
    # 各自的 stdout+stderr 落到 `.runtime/<名字>.out.log`（不是 logs/ ——
    # 那里只有走 ww_logger 的业务日志，而 uvicorn/celery/vite 自己抛的
    # 绑定失败、traceback 只在控制台上，隐藏窗口不能把它弄丢）。
    # 要实时看输出时显式开：`.\dev.bat -ShowWindows`。
    [switch]$ShowWindows
)

$ErrorActionPreference = 'Stop'

# ---------- 路径解析（脚本在 scripts/ 下，项目根是其父目录）----------
$Root     = Split-Path -Parent $PSScriptRoot
$LogDir   = Join-Path $Root 'logs'
$RunDir   = Join-Path $Root '.runtime'
$PidFile  = Join-Path $RunDir 'dev.pids.json'   # 运行时状态放 .runtime/，与日志分离
$SqlFile  = Join-Path $Root 'pgsql\001_init.sql'
$FrontDir = Join-Path $Root 'front'

# 端口从根 .env 现取，脚本里一个数字都不写（口径与 dotenv/Compose 一致，见 ports.ps1）
. (Join-Path $PSScriptRoot 'ports.ps1')
$Ports = Get-WuwaPorts $Root

# 隐藏窗口模式要把各服务的控制台输出写到 .runtime/<名字>.out.log，
# 而 Tee-Object **不会自建目录**（目录不存在时整条启动命令报错）——先确保它在。
if (-not (Test-Path $RunDir)) { New-Item -ItemType Directory -Force -Path $RunDir | Out-Null }

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

# ---------- 端口占用预检：Get-PortOwner / Get-WuwaPortClashes 见 scripts/ports.ps1 ----------

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
    # 不弹窗口 ≠ 看不见输出：隐藏模式下把 stdout+stderr 同时落盘。
    # logs/ 里只有走 ww_logger 的业务日志，而「uvicorn 抢不到端口」「import 就崩」
    # 这类信息只出现在控制台上 —— 没有这份文件就变成「服务静默没起来」，最难排查。
    # 三处实测坑（每一处都是先踩后改，别再改回去）：
    #   · Out-File 而不是 Tee-Object —— 5.1 的 Tee 写出 UTF-16LE，grep/head 当二进制看不了；
    #   · 必须再过一道 `[string]$_` —— 原生命令的 stderr 会被 PowerShell 包成 ErrorRecord，
    #     直接写文件就成了「uv : INFO: Started server process / 所在位置 行:1 字符:121 / +~~~」
    #     这种格式化噪声（uvicorn、celery 的正常 INFO 全走 stderr，等于整份日志被污染）；
    #   · EAP 设 Continue —— 默认 Stop 时第一条 stderr 就可能被当异常中断服务。
    $outLog = Join-Path $RunDir "$Name.out.log"
    if (-not $ShowWindows) {
        $Command = "`$ErrorActionPreference='Continue'; $Command 2>&1 |" +
            " ForEach-Object { [string]`$_ } | Out-File -FilePath '$outLog' -Encoding utf8 -Width 512"
    }
    $full = "`$host.UI.RawUI.WindowTitle='$Title'; cd '$WorkDir'; $Command"
    $p = Start-Process -FilePath 'powershell.exe' `
        -ArgumentList '-NoExit', '-NoProfile', '-Command', $full `
        -WorkingDirectory $WorkDir -PassThru `
        -WindowStyle $(if ($ShowWindows) { 'Normal' } else { 'Hidden' })
    $script:pids += [pscustomobject]@{ name = $Name; pid = $p.Id; title = $Title }
    if ($ShowWindows) {
        Write-Ok "$Name 已启动（窗口 PID $($p.Id)）"
    } else {
        Write-Ok "$Name 已启动（PID $($p.Id)，输出 → .runtime/$Name.out.log）"
    }
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
        # ---- 端口占用预检：只检本项目要**发布**到宿主机的那几个端口 ----
        # 容器内部端口（冒号右侧）不参与，撞不到宿主机。
        # 豁免名单 = 本项目**正在跑的容器实际发布的端口号**（见 ports.ps1 的说明：
        # 按进程豁免会连别家容器的占用一起放过，等于预检失明）。
        $ownPublished = @(Get-WuwaOwnPublishedPorts)
        $clashes = @(Get-WuwaPortClashes $Ports -OwnPublished $ownPublished)
        if ($clashes.Count -gt 0) {
            Write-Err2 '端口已被占用，docker compose 起不来。关掉占用者，或改根 .env 里对应的 *_PORT：'
            foreach ($c in $clashes) {
                Write-Info ("  {0,-14} :{1}  <= PID {2} {3}" -f $c.Service, $c.Port, $c.Owner.Pid, $c.Owner.Name)
                if ($c.Owner.Cmd) { Write-Info ("      {0}" -f $c.Owner.Cmd) }
            }
            exit 1
        }
        Write-Ok "端口占用预检通过（本项目已发布端口 $($ownPublished -join ', ')）"

        Push-Location $Root
        try {
            # 必须**裸调用**（stderr 不重定向）：compose 的彩色进度渲染靠检测真终端，
            # 一旦把 stderr 接进管道（2>&1 / 2>$null）它就退化成灰白纯文本。
            # 实测（PS 5.1，2026-09-22）：只有「重定向原生 stderr」才会产生
            # ErrorRecord 并在 $ErrorActionPreference='Stop' 下抛错（2>&1 实测抛
            # RemoteException）；**裸调用不抛**，$LASTEXITCODE 照常可读。
            # 所以这里既不要包装、也不要重定向。
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

        # Neo4j bolt / RustFS S3 API 端口探测。
        # Redis 不在这里探：上面已经 `docker exec ... redis-cli ping` 走容器内部验过，
        # 宿主机发布端口只服务外部客户端，再探一次只是多等一轮没意义的超时。
        $bolt = $Ports.Neo4jBolt
        $s3Api = $Ports.S3Api
        if (Wait-Until -TimeoutSec 60 -Test { Test-TcpPort 'localhost' $bolt }) { Write-Ok "Neo4j :$bolt 可达" } else { Write-Warn2 "Neo4j 端口 :$bolt 未就绪" }
        if (Wait-Until -TimeoutSec 40 -Test { Test-TcpPort 'localhost' $s3Api }) { Write-Ok "RustFS :$s3Api 可达" } else { Write-Warn2 "RustFS 端口 :$s3Api 未就绪" }
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
    # 日志角色由启动方显式声明（ww_logger 的 argv 推断只作兜底）——只用于给每行
    # 日志的 role 列打标签；两个进程共写同一个 rag.log，跨进程轮转安全靠
    # concurrent-log-handler 的文件锁（**不再按 role 拆文件**）。
    # 改这里要同步 ww_logger._process_tag 的注释。
    Start-ServiceWindow -Name 'celery' -Title '潮声智库 · Celery Worker' -WorkDir $Root `
        -Command '$env:WUWA_LOG_ROLE=''worker''; uv run celery -A wuwa_rag.tasks.worker:celery_app worker --pool=solo --loglevel=info' | Out-Null

    # ========== 4. FastAPI ==========
    $apiPort = $Ports.Api
    $webPort = $Ports.Web
    Write-Step "启动 FastAPI（:$apiPort）"
    Start-ServiceWindow -Name 'api' -Title "潮声智库 · FastAPI :$apiPort" -WorkDir $Root `
        -Command '$env:WUWA_LOG_ROLE=''api''; uv run python -m wuwa_rag.api.server' | Out-Null

    # ========== 5. 等后端就绪，再开前端 ==========
    # ⚠️ 顺序很关键：**必须先等 /health 返回 200 再启动前端**。
    # 反过来的话 vite 秒级就绪，浏览器（以及早就开着的旧标签页）立刻开始轮询
    # /health、/auth/me、/conversations…，而后端还在启动 —— 那 2~3 秒的空窗里
    # 代理会刷出一片 ECONNREFUSED（用户实测「问题依旧存在」的根因）。
    Write-Step '等待后端 /health 就绪'
    $healthUrl = "http://127.0.0.1:$apiPort/health"
    $apiReady = Wait-Until -TimeoutSec 120 -IntervalMs 1500 -What 'API' -Test {
        try {
            $r = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 3
            return ($r.StatusCode -eq 200)
        } catch { return $false }
    }
    if ($apiReady) { Write-Ok 'FastAPI /health 返回 200' }
    else {
        $hint = if ($ShowWindows) { "「FastAPI :$apiPort」窗口日志" } else { ".runtime/api.out.log" }
        Write-Warn2 "API 未在 120s 内就绪，请看 $hint（首次加载模型较慢）"
        # uvicorn 抢不到端口时是**静默**起不来的（只在它自己窗口里抛一句 bind error），
        # 这里顺手把端口当前的占用者打出来，省得对着超时日志猜。
        $apiOwner = Get-PortOwner $apiPort
        if ($apiOwner) {
            Write-Info ("  :$apiPort 现由 PID {0} {1} 占用{2}" -f $apiOwner.Pid, $apiOwner.Name, `
                $(if ($apiOwner.Cmd) { "  <- $($apiOwner.Cmd)" } else { '' }))
        }
    }

    # ========== 6. 前端 Vite ==========
    if (-not $NoFront) {
        Write-Step "启动前端 Vite（:$webPort）"
        # 前端开了 strictPort（见 front/vite.config.ts）：端口被占时直接失败而不是
        # 悄悄漂到别的端口——origin 一变，浏览器 localStorage 就是另一套，已登录用户
        # 会莫名其妙被要求重新登录。所以这里预先点出占用者。
        $webOwner = Get-PortOwner $webPort
        if ($webOwner) {
            Write-Warn2 ("端口 :$webPort 已被 PID {0} {1} 占用，Vite 会直接拒绝启动" -f $webOwner.Pid, $webOwner.Name)
        }
        if (Test-Path (Join-Path $FrontDir 'node_modules')) {
            Start-ServiceWindow -Name 'front' -Title "潮声智库 · 前端 :$webPort" -WorkDir $FrontDir `
                -Command 'npm run dev' | Out-Null
        } else {
            Write-Warn2 '前端依赖未安装，先跑 npm install（在新窗口）'
            Start-ServiceWindow -Name 'front' -Title "潮声智库 · 前端 :$webPort" -WorkDir $FrontDir `
                -Command 'npm install; npm run dev' | Out-Null
        }
    } else {
        Write-Step '跳过前端（-NoFront）'
    }

    Save-Pids

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
    前端        http://localhost:$webPort
    后端 API    http://127.0.0.1:$apiPort  (/docs 看接口)
    Neo4j       http://localhost:$($Ports.Neo4jHttp)   （Bolt 端口 $($Ports.Neo4jBolt)）
    RustFS      http://localhost:$($Ports.S3Console)   （S3 API 端口 $($Ports.S3Api)）
    PostgreSQL  localhost:$($Ports.Pg)      Redis  localhost:$($Ports.Redis)
    Ollama      http://localhost:$($Ports.Llm)  （外部依赖，端口不由本项目发布）

  端口一律来自根目录 .env 顶部的 *_PORT，改端口只改那一个文件。

  $(if ($ShowWindows) { '窗口模式：三个服务各占一个控制台窗口（实时看输出）。' } else { @(
    '服务模式：不弹窗口。各服务的控制台输出（含 uvicorn/celery/vite 自己的报错）在',
    '  .runtime\celery.out.log / api.out.log / front.out.log，\stop.ps1 会一并清掉。',
    '  想实时看输出：.\dev.bat -ShowWindows',
    '  [注意] 若 .env 没配 ADMIN_PASSWORD，首次启动的那次性管理员口令只会出现在',
    '     .runtime\api.out.log 里（故意不落 logs/）——读到后请删掉该文件。'
  ) -join [Environment]::NewLine })

  停止全部： .\scripts\stop.ps1   （或 .\dev.bat stop）
  PID 记录： $PidFile
"@ -ForegroundColor Green

    if ($Open) {
        Start-Process "http://localhost:$webPort"
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
