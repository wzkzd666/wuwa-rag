"""配置默认值与 prompt 模块拆分。

① 配置默认值
-----------
2026-10-06 收紧了三个默认值，都是「clone 即用」场景下的坑：
  · `DEBUG=True` → `False`：生产环境不该默认开调试；
  · `HF_HOME="D:/hf_cache/huggingface"` → `~/.cache/huggingface`：硬编码 D 盘，
    换台机器直接失效（本机看到的 D 盘路径来自**系统环境变量**，不是代码默认值）；
  · `CLOUD_ALLOW_PRIVATE_NET=True` → `False`：这是 SSRF 防护开关，
    默认放开等于给所有 clone 的人留了个口子。开发者自用再在 .env 里打开。

⚠️ 必须断言 `model_fields[...].default`，**不能**断言 `get_settings()` 的实例值
--------------------------------------------------------------------------
`get_settings()` 会读 .env，而 .env 是本地文件（gitignore、不入库）。
断言实例值会让「有 .env 的机器绿、CI 上红」，或反过来——两头都不可信。
`model_fields[...].default` 是代码里写死的默认值，才是这条测试要守的东西。

② prompt 模块拆分
----------------
`graph.py` 原本 1573 行混了八种职责，把纯字符串构建拆到了 `dialog/prompt.py`。
拆分最大的风险不是「没拆干净」，而是**再导出断链**：`api/app.py` 一直
`from wuwa_rag.dialog.graph import doc_sources`，拆分后 graph 必须把它再导出，
否则 app 导入期就炸。所以下面专门有一条断言「graph.doc_sources is prompt.doc_sources」
——验的是同一个对象，不只是同名。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from wuwa_rag.config import Settings, get_settings
from wuwa_rag.dialog import graph as graph_mod
from wuwa_rag.dialog import prompt as prompt_mod


def _default(name: str):
    """取代码里写死的字段默认值（不读 .env）。"""
    return Settings.model_fields[name].default


# ---------- ① 配置默认值 ----------

def test_DEBUG_默认关闭() -> None:
    assert _default("DEBUG") is False


def test_HF_HOME_不硬编码盘符且位于用户目录() -> None:
    """断言「在用户目录下」而非具体字符串——它由 Path.home() 在类定义时求值，
    不同机器/不同用户名下值不同，写死字符串会让测试只在作者机器上绿。
    """
    got = Path(_default("HF_HOME"))
    assert got == Path.home() / ".cache" / "huggingface"
    # 回归钉桩：绝不能再退回硬编码 D 盘那个路径
    assert got != Path("D:/hf_cache/huggingface")
    assert "hf_cache" not in str(got), f"HF_HOME 又退回硬编码路径了：{got}"


def test_SSRF_防护默认收紧() -> None:
    """CLOUD_ALLOW_PRIVATE_NET 默认必须是 False——放开等于给所有 clone 的人留 SSRF 口子。"""
    assert _default("CLOUD_ALLOW_PRIVATE_NET") is False


# ---------- 限流配置（档位数字被 test_ratelimit 用作断言依据）----------

@pytest.mark.parametrize(
    "field,expected",
    [
        ("RATE_LIMIT_ENABLED", True),
        ("RATE_LIMIT_STORAGE", ""),            # 留空 = 进程内存，不强制依赖 Redis
        ("RATE_LIMIT_AUTH", "10/minute"),
        ("RATE_LIMIT_ASK", "20/minute"),
        ("RATE_LIMIT_TTS", "10/minute"),
        ("RATE_LIMIT_OUTBOUND", "20/hour"),    # SSRF 面，按小时收得最紧
        ("RATE_LIMIT_DEFAULT", "200/minute"),
    ],
)
def test_限流配置默认值(field: str, expected) -> None:
    assert _default(field) == expected


# ---------- ② prompt 模块拆分 ----------

@pytest.mark.parametrize("name", ["SYSTEM_PROMPT", "build_context", "build_prompt", "doc_sources"])
def test_prompt_模块导出齐全(name: str) -> None:
    assert hasattr(prompt_mod, name), f"prompt.py 缺 {name}"


def test_graph_再导出_all_sources_是同一对象() -> None:
    """api/app.py 依赖 `from ...graph import all_sources`，拆分后必须再导出且是同一对象。

    用 `is` 而非「都能调通」：若哪天变成两份实现，同名同签名也能各自跑，
    但行为迟早分叉——那正是项目文档反复警告的「双判据必然不同步」。
    2026-10-23：引用来源改成「文档面包屑 + 联网溯源」的统一口径 `all_sources`，
    旧的 `doc_sources` 再导出随唯一消费者消失而删除（留着就是无人使用的转发层）。
    """
    assert graph_mod.all_sources is prompt_mod.all_sources


def test_graph_不再残留已迁出的私有实现() -> None:
    """旧私有名必须真的删掉，不能「新模块有了、旧函数还留着」——留着就是死代码 + 分叉源。

    历史上真出过这个状态：prompt.py 建好了、graph.py 也导入了，但旧 `_build_prompt`
    还在且仍引用已删除的 `_SYSTEM`，主链路一跑就 NameError。
    """
    for stale in ("_build_prompt", "_build_context", "_lock_focus", "_SYSTEM"):
        assert not hasattr(graph_mod, stale), f"graph.py 仍残留已迁出的 {stale}"


def test_联网溯源进引用面板() -> None:
    """`all_sources` = 文档面包屑 + 联网溯源，两条出口共用。

    2026-10-23 前的状态：千帆的 `- title: url` 只拼进 prompt 文本，`done.sources`
    里没有它 —— 靠联网答出来的内容，引用面板是空的。而「这句从哪查来的」
    恰恰是联网答案最需要被看见的部分（引用面板的立项理由就是「建立信任」）。
    """
    docs = [{"breadcrumb": "守岸人 › 角色攻略 › 配队"},
            {"breadcrumb": "守岸人 › 角色攻略 › 配队"},      # 重复面包屑要去重
            {"breadcrumb": ""}]
    web = [{"title": "鸣潮 守岸人 攻略", "url": "https://a.example/x"},
           {"title": "", "url": "https://b.example/y"},
           {"title": "缺 url 的项要丢掉", "url": ""},
           {"title": "重复链接", "url": "https://a.example/x"}]
    got = prompt_mod.all_sources(docs, web)
    assert got[0] == "守岸人 › 角色攻略 › 配队"
    assert sum(1 for x in got if x == "守岸人 › 角色攻略 › 配队") == 1, "文档来源没去重"
    assert any(x.startswith("联网 › ") and "a.example/x" in x for x in got)
    assert any(x.startswith("联网 › ") and "b.example/y" in x for x in got), "无标题时也要留 url"
    assert sum(1 for x in got if "a.example/x" in x) == 1, "联网来源没按 url 去重"
    assert not any("缺 url" in x for x in got)
    # 只有文档时行为与旧 doc_sources 一致（不能因为加了联网分支就改文档侧形态）
    assert prompt_mod.all_sources(docs) == prompt_mod.doc_sources(docs)


async def test_联网失败要说清原因且不留脏上下文(monkeypatch) -> None:
    """失败降级必须**带原因**、且三个字段互斥。

    旧契约所有失败都返回空串，用户只看到人设化的「翻了本地也上网找了，确实没有」——
    而真相可能是 Key 过期、根本没查成。这两种情况对用户的意义完全不同。
    """
    from wuwa_rag.dialog import graph as graph_mod

    async def _fail(_q: str):
        return False, "联网搜索没成功（API Key 无效或已过期）", []

    monkeypatch.setattr(graph_mod, "web_search", _fail)
    out = await graph_mod.web_node({"question": "随便", "search_query": "随便", "refined_query": ""})
    assert out["used_web"] is True
    assert out["web_facts"] == "", "失败时不能把原因当资料喂给模型"
    assert out["web_sources"] == []
    assert "API Key" in out["web_reason"], "失败原因丢了，用户又只能看到「确实没有」"

    # 跨轮清零：checkpointer 会把上一轮的 web_* 带进本轮，不清就挂到别的回答上
    fresh = graph_mod._fresh_state("下一句", [])
    for key in ("web_facts", "web_sources", "web_reason"):
        assert fresh.get(key) in ([], ""), f"{key} 没有每轮清零"


def test_done事件键集与落库meta同源() -> None:
    """SSE `done` 的形状由 `_done_event` 单点保证，且 `_meta_of` 必须把它**全部**落库。

    两个各自会漂移的地方钉在一起：
    ① `ask_stream` 有两条出口产出 done（正常收尾 / 角色不在库的早退）。早退那条一度
       少 `emotion`、`web_reason` 两键——同一个「不知道」在两条路径上形状不一样，
       前端只能靠 optional 兜着，缺的键要到界面上才看得出来。
    ② `api/app._meta_of` 把 done 挑出来存进 `messages.meta`，是**白名单**式的
       （逐个 `evt.get(...)`）。给 `_done_event` 加了新键而忘了加进这里，
       表现是「流式当场有、刷新历史就没了」——最难查的一类不一致。
    """
    from wuwa_rag.api.app import _meta_of

    done = graph_mod._done_event(answer="不知道", characters=["某角色"], thread_id="t1")
    # done 是事件信封、thread_id 是会话认领、message_id 是这条回答的 DB 行号（前端用它把
    # 气泡的本地随机 uid 换成真 id，反馈才关联得上）、answer 落在 messages.content，
    # 三者都不属于「这条回答的元数据」——除它们以外，done 里有的键 meta 必须都有。
    assert set(_meta_of(done)) == set(done) - {"done", "thread_id", "answer", "message_id"}, \
        "done 与 meta 的键集脱钩：新键没落库，或 meta 凭空多了 done 里没有的键"
    assert set(done) == {
        "done", "answer", "intent", "slots", "characters", "docs", "sources",
        "truncated", "emotion", "web_reason", "message_id", "thread_id",
    }
    # 早退路径没跑检索/生成：没发生的字段必须是**空值而不是缺失**，前端不必特判
    assert done["docs"] == 0 and done["sources"] == [] and done["emotion"] == ""
    assert done["web_reason"] == "" and done["truncated"] is False
    # 没落库就没有行号：None 而不是 0（0 会被前端当成一条真实消息 id）
    assert done["message_id"] is None


def test_SYSTEM_PROMPT_含云端不需要的本地约束() -> None:
    """SYSTEM_PROMPT 是给**本地 aemeath** 的（术语对照 + 作答约束）。

    云端模型走另一条路径不带它。这里只钉一个不变量：它非空且是字符串——
    措辞本身经多轮实测校准（约束与理由见 `dialog/prompt.py` 的模块注释），
    不适合在单测里逐字断言。
    """
    assert isinstance(prompt_mod.SYSTEM_PROMPT, str)
    assert prompt_mod.SYSTEM_PROMPT.strip()


# ---------- 本问解析（角色 / 内容 / 召回板块，三行各自可缺）----------

def test_本问解析_角色与内容都确定时三行齐全() -> None:
    """用户实测：问「心声骸推荐」，模型把「心声骸」当一个词、把「心」吃掉了。
    角色与槽位在检索侧本就确定性可解，写进上下文即可省掉模型自己猜分词。"""
    ctx = prompt_mod.build_context({
        "question": "心声骸推荐",
        "characters": ["心"],
        "slots": ["声骸"],
        "docs": [{"component": "声骸套装推荐", "breadcrumb": "心 › 声骸套装推荐", "text": "…"}],
    })
    assert "## 本问解析" in ctx
    assert "- 被问到的角色：心" in ctx
    assert "- 要问的内容：声骸" in ctx
    assert "- 召回资料所属板块：声骸套装推荐" in ctx


def test_本问解析_各行可缺且全缺时不生成() -> None:
    """角色认不出就不写角色行；没槽位就改用召回板块；两者都没有则整段不生成。"""
    # 角色没认出来 —— 不硬凑
    ctx = prompt_mod.build_context({
        "question": "声骸怎么配", "characters": [], "slots": ["声骸"],
        "docs": [{"component": "角色养成", "breadcrumb": "x", "text": "…"}],
    })
    assert "被问到的角色" not in ctx
    assert "- 要问的内容：声骸" in ctx

    # 无槽位 —— 改用实际召回的板块
    ctx = prompt_mod.build_context({
        "question": "这个角色怎么玩", "characters": ["心"], "slots": [],
        "docs": [{"component": "技能介绍", "breadcrumb": "…", "text": "…"},
                 {"component": "角色机制", "breadcrumb": "…", "text": "…"}],
    })
    assert "要问的内容" not in ctx
    assert "- 召回资料所属板块：技能介绍、角色机制" in ctx

    # 闲聊轮：全缺 → 整段不生成（不留空标题）
    ctx = prompt_mod.build_context({"question": "你好呀", "characters": [], "slots": [], "docs": []})
    assert "## 本问解析" not in ctx


# ---------- ③ 端口对账（2026-10-22）----------
# 背景：服务全部改用不常用端口（2xxxx 段），因为 9000 在本机被另一个项目的常驻服务
# 占死，而 5432/6379/8000/5173 是每台开发机都在抢的端口。端口从此**只在根 .env 定义一次**，
# 由四个消费方各自读取：docker-compose.yml（`${VAR}` 插值）、Python（dotenv 插值）、
# scripts/ports.ps1（探测与兜底 kill）、front/vite.config.ts（dev server 与代理目标）。
#
# ⚠️ 这五个文件里端口出现了两份「值」：.env 里的真值 + 各消费方缺席时的兜底默认。
# 不一致的后果不是报错，而是**静默连到旧端口**（compose 起了 25432、脚本还在探 5432，
# 于是永远超时）。正是 AGENTS.md 反复记录的「同一语义两处判据」，所以用测试钉死。

_RE_ROOT = Path(__file__).resolve().parents[1]

# 项目自己对外发布的服务端口，一个都不许再用这些常用值（Ollama 11434 是外部依赖，不在此列）
_LEGACY_PORTS = frozenset({5432, 6379, 7687, 7474, 9000, 9001, 8000, 5173})

_ENV_ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_ENV_QUOTED = re.compile(r"^\"(.*)\"$")
_ENV_TRAILING_COMMENT = re.compile(r"\s+#.*$")
_URL_PORT = re.compile(r":(\d{2,5})(?:[/?]|$)")
# 形如 "127.0.0.1:25432:5432" 或 "[::1]:${PG_PORT:?}:5432"；host_ip 段可能含冒号，单独吃掉。
# 发布端口那段允许 `${...}` 整体（里面有 `:?` 冒号，不能再按冒号切）
_COMPOSE_PUB = re.compile(
    r"""["']
         (?:\[[^\]]*\]|[^:"'\s]+)              # host_ip（127.0.0.1 / [::1]）
         :((?:\$\{[^}]*\})|(?:[^:"']+))        # 宿主机发布端口（应是 ${VAR}，允许带 :?）
         :(\d{2,5})                             # 容器内部端口
         ["']
    """,
    re.VERBOSE,
)
_COMPOSE_VAR = re.compile(r"^\$\{([A-Za-z0-9_]+)(?::[\?=][^}]*)?\}")
_PS_FALLBACK = re.compile(r"(\w+)\s*=\s*\[int\]\(Get-EnvValue\s+\$e\s+'([A-Z0-9_]+)'\s+'(\d+)'\)")


def _dotenv_map(text: str) -> dict[str, str]:
    """按 python-dotenv 的口径解析 .env：允许 `KEY = VALUE`、允许引号、
    未加引号的值在「空白 + #」处截断（与 Compose/ports.ps1 三边一致）。
    不做 `${VAR}` 插值——对账只比裸端口，插值另有专门断言。
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = _ENV_ASSIGN.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        quoted = _ENV_QUOTED.match(val)
        val = quoted.group(1) if quoted else _ENV_TRAILING_COMMENT.sub("", val).strip()
        out[key] = val
    return out


def _url_port(value: str) -> int:
    """从 bolt://host:27687 / redis://host:26380/0 这类串里取端口。"""
    m = _URL_PORT.search(value)
    assert m, f"没能从 {value!r} 里解析出端口"
    return int(m.group(1))


def _code_ports() -> dict[str, int]:
    """config.py 里写死的默认端口（不读 .env，与 `_default` 同一口径）。"""
    got = {
        "PG_PORT": int(_default("PG_PORT")),
        "API_PORT": int(_default("API_PORT")),
        "WEB_PORT": int(_default("WEB_PORT")),
    }
    for key in ("NEO4J_URI", "REDIS_URL", "S3_ENDPOINT", "LLM_URL"):
        got[key] = _url_port(str(_default(key)))
    return got


def test_端口_不再使用被占用的常用端口() -> None:
    """本项目发布的每个端口都必须离开常用段。

    踩过的坑：9000 被 `Starry River Tea Chronicles` 的 `main.py serve` 常驻占用，
    RustFS 一起就 `port is already allocated`。8000/5173 同理由 Vite/其它后端随时抢。
    """
    ports = _code_ports()
    # LLM_URL 是唯一豁免：Ollama 是外部依赖，不由本项目发布端口
    owned = {k: v for k, v in ports.items() if k != "LLM_URL"}
    clash = {k: v for k, v in owned.items() if v in _LEGACY_PORTS}
    assert not clash, f"这些服务端口又退回常用端口了：{clash}"
    # 也绝不落进 Windows 动态端口区（49152-65535）：那段由系统给出站连接随机分配，
    # 落在里面等于「端口偶尔被别人占一下」，是最难复现的一类故障。
    in_dyn = {k: v for k, v in owned.items() if v >= 49152}
    assert not in_dyn, f"端口落进了 Windows 动态区：{in_dyn}"
    assert len(set(owned.values())) == len(owned), f"端口彼此撞了：{owned}"


def test_端口_compose_只引用变量不写死宿主机端口() -> None:
    """docker-compose.yml 的宿主机发布端口必须是 `${VAR}`，容器内部端口才允许写死。

    写死左边 = 改 .env 后 compose 仍绑旧端口，而 Python/前端已经连新端口，
    表现成「连不上数据库」却看不出为什么。
    """
    text = (_RE_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    published = [(m.group(1), m.group(2)) for m in _COMPOSE_PUB.finditer(text)]
    assert published, "compose 里一个端口映射都没解析到，多半是这里的正则先坏了"
    literals = [(pub, target) for pub, target in published if not _COMPOSE_VAR.match(pub)]
    assert not literals, f"compose 宿主机端口写死了（应改为 .env 的 ${{VAR:?}}）：{literals}"
    # 必须是 `:?` 必填形态：普通 `${VAR}` 缺变量时静默展开成空串，Docker 会随手分配一个
    # 随机宿主端口，容器照常起来而 Python/前端按 .env 连 —— 最难查的那种「起来了但连不上」。
    not_required = [pub for pub, _ in published if ":?" not in pub]
    assert not not_required, f"这些映射没用 ${{VAR:?}}，缺变量时会静默换端口：{not_required}"
    names = {_COMPOSE_VAR.match(pub).group(1) for pub, _ in published}
    assert names == {
        "PG_PORT", "NEO4J_BOLT_PORT", "NEO4J_HTTP_PORT",
        "REDIS_PORT", "S3_API_PORT", "S3_CONSOLE_PORT",
    }, f"compose 引用的端口变量集合变了：{sorted(names)}"


def test_端口_ports_ps1_兜底值与代码默认一致() -> None:
    """scripts/ports.ps1 里 .env 缺席时的兜底端口，必须等于 config.py 的默认值。

    这两份各自独立写死，漂了的后果是「脚本探的端口和服务实际听的不是同一个」，
    启动脚本一直超时误导排查。
    """
    text = (_RE_ROOT / "scripts" / "ports.ps1").read_text(encoding="utf-8-sig")
    got = {env_key: int(fb) for _name, env_key, fb in _PS_FALLBACK.findall(text)}
    assert got, "ports.ps1 里没解析到任何兜底端口，多半是 _PS_FALLBACK 正则先坏了"
    code = _code_ports()
    # URL 类字段在这里都是裸端口变量，按名字逐一对上
    mapping = {
        "PG_PORT": "PG_PORT",
        "NEO4J_BOLT_PORT": "NEO4J_URI",
        "REDIS_PORT": "REDIS_URL",
        "S3_API_PORT": "S3_ENDPOINT",
        "API_PORT": "API_PORT",
        "WEB_PORT": "WEB_PORT",
        "LLM_PORT": "LLM_URL",
    }
    drift = {k: (got[k], code[v]) for k, v in mapping.items() if got[k] != code[v]}
    assert not drift, f"ports.ps1 兜底值与 config.py 默认值不一致（端口:脚本值,代码值）：{drift}"
    assert got["NEO4J_HTTP_PORT"] not in _LEGACY_PORTS
    assert got["S3_CONSOLE_PORT"] not in _LEGACY_PORTS


def test_端口_vite_兜底值与代码默认一致() -> None:
    """front/vite.config.ts 的兜底端口必须等于 config.py 的 API_PORT / WEB_PORT。

    漂了的形态很具体：Vite 把 /api 代理到旧端口，页面全 502，而后端日志里
    一个请求都没收到——最容易误判成「后端挂了」。
    """
    text = (_RE_ROOT / "front" / "vite.config.ts").read_text(encoding="utf-8")
    code = _code_ports()
    for env_key, field in (("API_PORT", "API_PORT"), ("WEB_PORT", "WEB_PORT")):
        m = re.search(rf"env\.{env_key}\s*\|\|\s*'(\d+)'", text)
        assert m, f"vite.config.ts 里没找到 {env_key} 的兜底值，读取方式改了要同步本用例"
        assert int(m.group(1)) == code[field], (
            f"vite 兜底 {env_key}={m.group(1)} != config.py {field}={code[field]}"
        )


@pytest.mark.skipif(
    not (_RE_ROOT / ".env").exists(), reason=".env 是本地文件（gitignore），CI 上无此项"
)
def test_端口_env_是唯一真源且与各消费方一致() -> None:
    """根 .env 的 *_PORT ↔ config.py 默认值 ↔ compose 引用的变量名 三方对账。

    ⚠️ .env 不入库，所以本用例是「作者机器上的门禁」；上面三条才是在 CI 上真正
    兜住的那几条。这里额外抓的是 dotenv 插值的顺序坑：被引用的 *_PORT 必须排在
    引用它的 URL **之前**，否则展开成空串（实测写出过 redis://localhost:/0）。
    """
    env = _dotenv_map((_RE_ROOT / ".env").read_text(encoding="utf-8-sig"))
    code = _code_ports()

    for key in ("PG_PORT", "API_PORT", "WEB_PORT", "NEO4J_BOLT_PORT", "NEO4J_HTTP_PORT",
                "REDIS_PORT", "S3_API_PORT", "S3_CONSOLE_PORT", "LLM_PORT"):
        assert key in env, f".env 缺端口变量 {key}（compose 会插值成空串直接起不来）"
        assert env[key].isdigit(), f".env 的 {key} 不是裸端口：{env[key]!r}（行尾注释要留空格才对）"

    assert int(env["PG_PORT"]) == code["PG_PORT"]
    assert int(env["API_PORT"]) == code["API_PORT"]
    assert int(env["WEB_PORT"]) == code["WEB_PORT"]

    # URL 一律写成「引用端口变量」的形态，不再抄第二份数字——这才是「只定义一次」的落点。
    # 写死的话，改 *_PORT 后 URL 仍指旧端口，表现成「连不上数据库」而看不出为什么。
    _derived = (("NEO4J_URI", "NEO4J_BOLT_PORT"), ("REDIS_URL", "REDIS_PORT"),
                ("S3_ENDPOINT", "S3_API_PORT"), ("LLM_URL", "LLM_PORT"))
    for key, port_key in _derived:
        assert f"${{{port_key}}}" in env[key], \
            f".env 的 {key} 又写死端口了（应写成 ${{{port_key}}}）：{env[key]!r}"

    # 并且 dotenv 确实把它展开了：插值失效（顺序错、变量名打错）时 URL 会变成
    # `redis://localhost:/0` 这种看着像、连不通的形态，只有实测展开结果才抓得住。
    s = get_settings()
    for field, port_key in _derived:
        resolved = _url_port(getattr(s, field))
        assert resolved == int(env[port_key]), \
            f"{field} 插值后端口={resolved}，与 {port_key}={env[port_key]} 不符（插值没生效？）"

    # 插值顺序：URL 行引用的变量，必须已经在它之前定义
    lines = (_RE_ROOT / ".env").read_text(encoding="utf-8-sig").splitlines()
    seen: set[str] = set()
    for line in lines:
        m = _ENV_ASSIGN.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        for ref in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", raw):
            assert ref in seen, f".env 第 {lines.index(line) + 1} 行引用了尚未定义的 {ref}（dotenv 按行求值）"
        seen.add(key)

    # CORS 源必须跟着 WEB_PORT：写成 `${WEB_PORT}` 最好；万一写死，也必须等于它。
    cors_raw = next(
        (ln for ln in lines
         if _ENV_ASSIGN.match(ln) and _ENV_ASSIGN.match(ln).group(1) == "CORS_ORIGINS"),
        "",
    )
    if "${WEB_PORT}" not in cors_raw:
        assert all(_url_port(o) == int(env["WEB_PORT"]) for o in env["CORS_ORIGINS"].split(",")), \
            "CORS_ORIGINS 里的前端端口和 WEB_PORT 不一致（应写成 http://localhost:${WEB_PORT}）"


# ---------- 日志大类名单必须与 get_logger 调用点对齐 ----------

def test_日志大类名单与调用点一致() -> None:
    """`_MODULE_LOGS` 漏一个名字**不报错**，只会让那个 logger 静默落进 app.log。

    后果是排查时的误判：按业务大类去翻 `<name>.log` 发现文件不存在，读成「这条链路没跑」。
    `music` / `auth` 就这样长期混在 app.log 里（2026-10-23 补全）。
    反向也要成立：名单里不许留从未被引用的名字（曾经的 ai / pg 就是这么清理掉的）。
    """
    from wuwa_rag.ww_logger import _MODULE_LOGS

    src_root = Path(__file__).resolve().parent.parent / "src"
    used = {
        m.group(2)
        for f in src_root.rglob("*.py")
        # 只认合法标识符（首字母后跟字母/数字/下划线）：注释/文档里举例写的
        # get_logger("...") 不是调用点。两点容易漏：
        #   · 名字里带数字（neo4j / bm25），别写成 [a-z_]+；
        #   · 调用点引号两种风格都有（`get_logger('upload')` 与 `get_logger("rag")`），
        #     只匹配一种会让那半边名单被误判成「从未被引用」。
        # ⚠️ 正则在**整份文件文本**上跑，分不清代码与注释：注释里若要举一个已删掉的名字，
        #    写成 `get_logger("那个名字")` 这种完整调用形态会被当调用点、当场红
        #    （2026-10-24 改名 celery→worker 时，就红在解释这次改名的注释上）。
        #    说名字请只写名字，别写成调用形态。
        for m in re.finditer(r"""get_logger\((['"])([a-z_][a-z0-9_]*)\1\)""",
                             f.read_text(encoding="utf-8"))
    }
    # app 是 root 兜底文件本身，不进独立名单
    missing = sorted(used - set(_MODULE_LOGS) - {"app"})
    stale = sorted(set(_MODULE_LOGS) - used)
    assert not missing, f"这些 logger 名没有独立文件，会静默落进 app.log：{missing}"
    assert not stale, f"名单里有从未被 get_logger 引用的名字：{stale}"


def test_问句长度帽后端只有一处且与前端成对() -> None:
    """`/ask` 与 `/debug/retrieval` 各自都有一个问句长度上限；两处字面量迟早会漂
    （漂了的后果不是报错，而是**同一个问题在两个入口一边能发一边被拒**）。

    前端 `Composer.tsx` 的 `QUESTION_MAX` 是**跨语言**的第二道：它先挡是手感
    （超限输入不了，比发出去收 422 好），后端才是判据 —— 所以两边数值必须相等。
    与端口四处对账同一类做法：读源码文本比常量，不起服务、不碰网络。
    """
    import re

    import wuwa_rag.api.app as app_mod
    from wuwa_rag.api.app import AskIn

    backend = app_mod.QUESTION_MAX_CHARS
    assert AskIn.model_fields["question"].metadata, "问句字段没有约束了"
    caps = [m.max_length for m in AskIn.model_fields["question"].metadata if hasattr(m, "max_length")]
    assert caps == [backend], f"AskIn 的帽没引用常量：{caps}"

    src = (_RE_ROOT / "src/wuwa_rag/api/app.py").read_text(encoding="utf-8")
    lit = re.findall(r"max_length=\d+", src)
    assert not lit, f"app.py 里还有写死的 max_length 字面量（应引用常量）：{lit}"
    dbg = src[src.index('@app.get("/debug/retrieval")'):][:600]
    assert "max_length=QUESTION_MAX_CHARS" in dbg, "调试出口没走同一份帽"

    front = (_RE_ROOT / "front/src/components/Composer.tsx").read_text(encoding="utf-8")
    m = re.search(r"const QUESTION_MAX = (\d+)", front)
    assert m, "前端那道 QUESTION_MAX 没了？两头都得有（前端管手感、后端管判据）"
    assert int(m.group(1)) == backend, f"前后端帽不一致：前端 {m.group(1)} vs 后端 {backend}"

    # 重命名那条：入口必须 ≥ 落库截断，否则用户写了一半会被无声砍短
    from wuwa_rag.core.conversations import MAX_TITLE_CHARS
    assert app_mod.RENAME_MAX_CHARS >= MAX_TITLE_CHARS, (
        f"入口 {app_mod.RENAME_MAX_CHARS} 比落库截断 {MAX_TITLE_CHARS} 还窄")
