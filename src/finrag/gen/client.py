"""生成侧调用：DeepSeek（走 OpenAI 协议）的流式与非流式入口。

流式与非流式共用同一份参数装配；错误在此翻译成带排查方向的 :class:`LLMError`。
不做自动重试，Key 缺失在进入网络之前就报。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from finrag.config import AppConfig, get_config

__all__ = [
    "ChatReply",
    "Delta",
    "LLMError",
    "chat",
    "get_client",
    "stream_chat",
]

logger = logging.getLogger(__name__)

# 加载锁：客户端是进程级单例（内部有连接池），并发首调会各建一份。
_CLIENT_LOCK = threading.Lock()
_CLIENT: Any = None
_CLIENT_KEY: str | None = None


class LLMError(RuntimeError):
    """调用失败 —— 带排查方向；``kind`` 机器可读，``hint`` 是给人的下一步。"""

    def __init__(self, kind: str, message: str, *, hint: str = "", status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.hint = hint
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "message": str(self), "hint": self.hint, "status": self.status}


@dataclass(frozen=True, slots=True)
class Delta:
    """流式的一帧。``text`` 为空且 ``finish_reason`` 有值才是结束帧，不能只看空 ``text``。

    ``reasoning`` 是思考模式下的思维链，走 ``delta.reasoning_content``，与 ``text``
    （``delta.content``）是两条**并行**的流，不要拼接；非思考模型恒为空串。
    """

    text: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


@dataclass(slots=True)
class ChatReply:
    """非流式的一次完整回答。``reasoning`` 与 ``text`` 同样分属两个字段。"""

    text: str
    reasoning: str = ""
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""


# ------------------------------------------------------------------
# 客户端（进程级单例）
# ------------------------------------------------------------------
def get_client(cfg: AppConfig | None = None) -> Any:
    """进程级 ``openai.OpenAI`` 单例；``base_url`` 或 ``api_key`` 变化会重建客户端。"""
    cfg = cfg or get_config()
    global _CLIENT, _CLIENT_KEY

    if cfg.llm.api_key is None:
        # 在此就报，不要等 401：401 会被读成"Key 无效"，真相往往是环境变量名对不上。
        raise LLMError(
            "no_key",
            "未读到 DeepSeek API Key，无法调用生成模型。",
            hint="设环境变量 DEEPSEEKFLASH_KEY 或 FINRAG_LLM__API_KEY；"
                 "诊断用 scripts/check_llm_key.py --probe",
        )

    key = cfg.llm.api_key.get_secret_value()
    sig = f"{cfg.llm.base_url}|{key[-6:]}|{cfg.llm.timeout}"
    with _CLIENT_LOCK:
        if _CLIENT is None or _CLIENT_KEY != sig:
            from openai import OpenAI

            logger.info("构建 LLM 客户端：base_url=%s timeout=%ds",
                        cfg.llm.base_url, cfg.llm.timeout)
            _CLIENT = OpenAI(
                api_key=key,
                base_url=cfg.llm.base_url,
                timeout=cfg.llm.timeout,
                # max_retries=0 是刻意的：重试的单位是"一次完整问答"，由前端发起
                max_retries=0,
            )
            _CLIENT_KEY = sig
        return _CLIENT


def _params(cfg: AppConfig, *, stream: bool) -> dict[str, Any]:
    """装配请求参数，流式与非流式共用这一处；不传 ``stream_options``（代价是流式没有 usage，
    但避免未验证参数的报错歧义）。

    **刻意不传思考模式开关**：``deepseek-flash`` 服务端默认就是 ``thinking=enabled`` +
    ``reasoning_effort=high``，不显式声明 = 沿用默认 = 想关掉它的前端开关只影响是否展示，
    不影响答案。若哪天要显式固定，写作
    ``extra_body={"thinking": {"type": "enabled"}}`` 加 ``reasoning_effort="high"``
    （关掉思考用 ``reasoning_effort="none"``，但会掉事实性问答的准确率，别默认开）。
    """
    return dict(
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        stream=stream,
    )


def _translate(exc: Exception) -> LLMError:
    """把 provider 异常翻成带排查方向的 :class:`LLMError`；按 ``status_code`` 分派（比类名稳定），取不到再按类名兜底。"""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    text = str(exc)

    if status == 401 or name == "AuthenticationError":
        return LLMError("auth", f"Key 被拒绝（HTTP {status}）：{text[:200]}",
                        hint="Key 读到了但服务端不认。核对 DEEPSEEKFLASH_KEY 的值是否完整、"
                             "账户是否有余额；读取路径见 scripts/check_llm_key.py",
                        status=status)
    if status in (400, 404):
        # 请求侧错误的唯一出口：把"我们发错了"从兜底的 unknown 里分出来
        # （网络问题另有 timeout / connection 两个分支，不靠这里）。
        return LLMError("bad_model_or_request", f"HTTP {status}：{text[:200]}",
                        hint="400/404 = 请求被拒，不是网络问题。按两步核："
                             "① llm.model —— 它可被 configs/default.yaml、环境变量 "
                             "FINRAG_LLM__MODEL、或消融用的另一份 yaml 覆盖，未必还是默认值；"
                             "实际取值与是否现役名，跑 scripts/check_llm_key.py 看。"
                             "② 请求体 —— 如 max_tokens 是否超服务端上限、temperature 越界。",
                        status=status)
    if status == 429 or name == "RateLimitError":
        return LLMError("rate_limit", f"被限流（HTTP {status}）：{text[:200]}",
                        hint="等一会儿再发；本模块刻意不自动重试（见模块说明）",
                        status=status)
    if name in ("APITimeoutError", "Timeout", "ReadTimeout"):
        return LLMError("timeout", f"超时：{text[:200]}",
                        hint="当前 llm.timeout 见配置；流式下它是**两帧之间**的容忍时间。"
                             "长回答卡住可抬高该值",
                        status=status)
    if name in ("APIConnectionError", "ConnectError", "ConnectTimeout"):
        return LLMError("connection", f"连不上：{text[:200]}",
                        hint="核对 llm.base_url（当前值见配置）。网络连通性可跑 "
                             "scripts/check_env.py --online",
                        status=status)
    return LLMError("unknown", f"{name}: {text[:200]}",
                    hint="未归类的失败，先看原始报错全文", status=status)


# ------------------------------------------------------------------
# 两个入口
# ------------------------------------------------------------------
def chat(
    messages: Sequence[dict[str, str]],
    cfg: AppConfig | None = None,
) -> ChatReply:
    """非流式。返回完整回答，供评测逐字核对。"""
    cfg = cfg or get_config()
    client = get_client(cfg)
    try:
        resp = client.chat.completions.create(
            messages=list(messages), **_params(cfg, stream=False)
        )
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001 —— 必须在这里收口并翻译
        raise _translate(exc) from exc

    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    # content 可能为 None（只回 reasoning 或触发内容过滤），用 ``or ""`` 兜底
    text = choice.message.content or ""
    return ChatReply(
        text=text,
        reasoning=getattr(choice.message, "reasoning_content", None) or "",
        finish_reason=getattr(choice, "finish_reason", None),
        usage={
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        } if usage else {},
        model=getattr(resp, "model", ""),
    )


def stream_chat(
    messages: Sequence[dict[str, str]],
    cfg: AppConfig | None = None,
) -> Iterator[Delta]:
    """流式逐帧吐 :class:`Delta`，最后给一个带 ``finish_reason`` 的结束帧；异常发生在迭代过程中
    （HTTP 已 200），调用方须自行包成 ``event: error``。
    """
    cfg = cfg or get_config()
    client = get_client(cfg)
    try:
        stream = client.chat.completions.create(
            messages=list(messages), **_params(cfg, stream=True)
        )
    except Exception as exc:  # noqa: BLE001
        raise _translate(exc) from exc

    try:
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                # 有些 provider 末尾发一帧只带 usage、无 choices：跳过，它不是结束帧
                continue
            ch = choices[0]
            delta = getattr(ch, "delta", None)
            text = getattr(delta, "content", None) if delta is not None else None
            # 思维链在独立字段里，不取就直接丢了（服务端仍照常计费）。非思考模型恒为 None。
            reasoning = (
                getattr(delta, "reasoning_content", None) if delta is not None else None
            )
            fr = getattr(ch, "finish_reason", None)
            if text or reasoning or fr:
                yield Delta(
                    text=text or "",
                    reasoning=reasoning or "",
                    finish_reason=fr,
                )
    except Exception as exc:  # noqa: BLE001
        raise _translate(exc) from exc
