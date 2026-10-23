<#
.SYNOPSIS
    读取根目录 .env 里的服务端口（本项目端口的唯一来源）。
.DESCRIPTION
    start.ps1 / stop.ps1 要探测「端口通了没」、要兜底清理端口占用，而端口本身
    只在 .env 里定义一次。脚本里写死数字的后果是「改了 .env 忘了改脚本」：
    探测的是已经没人监听的旧端口，表现成启动超时/一直等待，真实服务其实好着——
    这类不一致不报错，只会把排查方向整个带偏，所以端口一律从 .env 现取。

    只解析 `KEY = 数字` 这种标量，不做 ${VAR} 插值：带插值的是 URL
    （NEO4J_URI / REDIS_URL / S3_ENDPOINT），脚本用不到，Python 侧由 dotenv 展开。

    ⚠️ 兜底默认值必须与 .env、docker-compose.yml、config.py 四处一致；
       由 tests/test_config_and_prompt.py 的「端口四处对账」用例钉住。
    ⚠️ 语法保持 PowerShell 5.1 可用：没有 ?? / ?. 运算符，取值一律走 Get-EnvValue。
#>

function Get-EnvValue {
    param([hashtable]$Map, [string]$Key, [string]$Default)
    if ($Map.ContainsKey($Key) -and $Map[$Key] -ne '') { return $Map[$Key] }
    return $Default
}

# .env 的行形态很松：允许 `KEY = VALUE`（本项目就这么写）、允许值带双引号、
# 允许行尾 `# 注释`、允许行尾空格。python-dotenv 与 Compose 都能吃这种写法，
# 这里必须按**同一口径**解析，三边看到的才是同一个值——特别是行尾注释：
# dotenv 对未加引号的值会在「空白 + #」处截断（`25432   # PostgreSQL` → `25432`），
# 而引号内的 `#` 原样保留（`"1 # x"` → `1 # x`）。解析器少做这一步，
# `[int]` 转换就会炸在端口上，表现成脚本报错而服务其实好着。
function Read-WuwaEnv {
    param([string]$Root)
    $map = @{}
    $file = Join-Path $Root '.env'
    if (-not (Test-Path $file)) { return $map }
    foreach ($line in @(Get-Content -Path $file -Encoding UTF8)) {
        if ($line -match '^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
            $key = $Matches[1]
            $val = $Matches[2]
            # 与 python-dotenv 同口径：**先判断有没有前引号**，有就取到配对闭引号为止
            # （引号内的 `#` 属于值本身），没有才在「空白 + #」处截断行尾注释。
            # 原先先整串匹配 `^"(.*)"$`：遇到 `PG_PORT = "25432" # 注释` 时整串匹配失败
            # （结尾不是引号）→ 落到剥注释分支 → 剩下带引号的 `"25432"` → `[int]` 抛
            # InvalidCast，start/stop 直接中断（2026-10-23 实测复现；Compose 与 dotenv
            # 都能正确解析成 25432，三边就此不一致）。
            if ($val -match '^"([^"]*)"') { $val = $Matches[1] }
            elseif ($val -match "^'([^']*)'") { $val = $Matches[1] }
            else { $val = ($val -replace '\s+#.*$', '').Trim() }
            $map[$key] = $val
        }
    }
    return $map
}

function Get-WuwaPorts {
    param([string]$Root)
    $e = Read-WuwaEnv $Root
    return [pscustomobject]@{
        Pg        = [int](Get-EnvValue $e 'PG_PORT'         '25432')
        Neo4jBolt = [int](Get-EnvValue $e 'NEO4J_BOLT_PORT' '27687')
        Neo4jHttp = [int](Get-EnvValue $e 'NEO4J_HTTP_PORT' '27474')
        Redis     = [int](Get-EnvValue $e 'REDIS_PORT'      '26380')
        S3Api     = [int](Get-EnvValue $e 'S3_API_PORT'     '29000')
        S3Console = [int](Get-EnvValue $e 'S3_CONSOLE_PORT' '29001')
        Api       = [int](Get-EnvValue $e 'API_PORT'        '28000')
        Web       = [int](Get-EnvValue $e 'WEB_PORT'        '25173')
        Llm       = [int](Get-EnvValue $e 'LLM_PORT'        '11434')
    }
}

# 谁在监听这个端口。返回 $null 表示空闲。
# 撞端口时 docker compose 只给一行 `Bind for 127.0.0.1:29000 failed: port is already
# allocated`——不说谁占的，也分不清是本项目上一轮没关干净还是别的项目。本机实测就是
# 后者（9000 被另一个项目的常驻服务占死）。所以把 PID + 进程名 + 完整命令行一起报出来，
# 下一步动作才有依据（关它，还是改 .env 的 *_PORT）。
function Get-PortOwner {
    param([int]$Port)
    $conns = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    if ($conns.Count -eq 0) { return $null }
    $ownerPid = $conns[0].OwningProcess
    $proc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
    $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId=$ownerPid" -ErrorAction SilentlyContinue).CommandLine
    return [pscustomobject]@{
        Pid      = $ownerPid
        Name     = if ($proc) { $proc.ProcessName } else { 'unknown' }
        Cmd      = $cmdLine
        IsDocker = [bool]($proc -and $proc.ProcessName -like 'com.docker*')
    }
}

# compose 将要发布的端口里，哪些已经被占。
# ⚠️ 「豁免」必须按**端口号**判，不能按「是不是 Docker 占的」判：Windows 上 Docker Desktop
# 用同一个 `com.docker.backend` 进程替**所有项目**的容器转发端口（本机实测：Dify、
# srtc-mysql 与本项目都挂在同一个 PID 上）。按进程豁免，等于在本项目容器已启动时
# 把「别家容器占了我们的 2xxxx 口」也当成自己上一轮 —— 预检当场失明，
# 要等 `docker compose up` 在 bind 时才炸（2026-10-23 审查抓到）。
# 正确做法：调用方从 `docker ps` 里取**本项目容器实际发布的端口号**传进来，只豁免这些。
function Get-WuwaPortClashes {
    param([pscustomobject]$Ports, [int[]]$OwnPublished = @())
    $map = [ordered]@{
        'PostgreSQL'    = $Ports.Pg
        'Neo4j Bolt'    = $Ports.Neo4jBolt
        'Neo4j HTTP'    = $Ports.Neo4jHttp
        'Redis'         = $Ports.Redis
        'RustFS API'    = $Ports.S3Api
        'RustFS 控制台' = $Ports.S3Console
    }
    $clashes = @()
    foreach ($svc in $map.Keys) {
        $port = [int]$map[$svc]
        $owner = Get-PortOwner $port
        if (-not $owner) { continue }
        if ($owner.IsDocker -and $OwnPublished -contains $port) { continue }
        $clashes += [pscustomobject]@{ Service = $svc; Port = $port; Owner = $owner }
    }
    return $clashes
}

# 本项目正在跑的容器实际发布到宿主机的端口号（只有这些算「自己上一轮留下的」）。
# 从 `docker ps` 的 Ports 列里取 `127.0.0.1:25432->5432/tcp` 这种形态的**左侧端口**。
function Get-WuwaOwnPublishedPorts {
    $out = @()
    $lines = @(docker ps --format '{{.Names}}|{{.Ports}}' 2>$null)
    foreach ($line in $lines) {
        $parts = $line -split '\|', 2
        if ($parts.Count -lt 2) { continue }
        if ($parts[0] -notlike 'wuwa-*') { continue }
        foreach ($m in [regex]::Matches($parts[1], ':([0-9]+)->')) {
            $p = 0
            if ([int]::TryParse($m.Groups[1].Value, [ref]$p)) { $out += $p }
        }
    }
    return @($out | Select-Object -Unique)
}
