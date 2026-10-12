"""实测 ``llm.api_key`` 的两个环境变量名是否都能读到（只读、不联网、不调 API）。

用法：
    python scripts/check_llm_key.py
    python scripts/check_llm_key.py --probe   # 逐个名字单独试
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from pydantic import AliasChoices, SecretStr  # noqa: E402

from finrag.config import AppConfig, get_config  # noqa: E402
from finrag.console import use_utf8_stdout  # noqa: E402

#: 应当都能读到 Key 的环境变量名。顺序即优先级（``AliasChoices`` 的顺序）。
EXPECTED_ENV_NAMES = ("FINRAG_LLM__API_KEY", "DEEPSEEKFLASH_KEY")

#: 刻意不接受的历史名：.env 占位符用的就是它，接上会让假 Key 顶掉真 Key。
RETIRED_ENV_NAMES = ("DEEPSEEK_API_KEY",)

out: list[str] = []


def emit(s: str = "") -> None:
    out.append(s)
    print(s)


def _mask(v: str | None) -> str:
    return f"{v[:7]}...（{len(v)} 位）" if v else "(None)"


def main() -> int:
    use_utf8_stdout()
    cfg = get_config()
    ap = argparse.ArgumentParser(description="检查 LLM Key 的读取路径（只读、不联网）")
    ap.add_argument("--out", default=str(cfg.paths.resolve(cfg.paths.log_dir) / "llm_key_check.txt"),
                    help="报告落盘路径，默认取 paths.log_dir")
    ap.add_argument("--probe", action="store_true",
                    help="逐个环境变量名单独构造配置，验别名是否真的生效")
    args = ap.parse_args()

    emit("FinRAG LLM Key 检查（只读、不联网）")
    emit("")

    # ---- 1. 正式路径：走完整优先级链（env > .env > yaml > 默认值）----
    emit("=== 1. 正式路径 load_config() ===")
    key = cfg.llm.api_key
    got = key.get_secret_value() if key else None
    emit(f"  llm.model          {cfg.llm.model}")
    emit(f"  llm.base_url       {cfg.llm.base_url}")
    emit(f"  llm.temperature    {cfg.llm.temperature}")
    emit(f"  llm.api_key        {_mask(got)}")

    ok = bool(got) and got.startswith("sk-") and "xxxx" not in got
    emit(f"  ⇒ {'通过：拿到真实 Key' if ok else '**失败：没拿到可用 Key**'}")

    emit("")
    emit("=== 2. 环境变量侧的实际状态 ===")
    for name in EXPECTED_ENV_NAMES:
        v = os.environ.get(name)
        emit(f"  {name:26s} {'已设置' if v else '未设置'}"
             f"{'  ' + _mask(v) if v else ''}")
    for name in RETIRED_ENV_NAMES:
        v = os.environ.get(name)
        emit(f"  {name:26s} {'已设置' if v else '未设置'}"
             f"{'  ← 刻别名列表，历史名（.env 占位符用的就是它）' if v else ''}")

    # ---- 3. 逐名探针：确认两个名字各自都能命中 ----
    # 探针须对 AppConfig（BaseSettings，读环境变量）构造；用 LLMConfig 会恒不命中。
    if args.probe:
        emit("")
        emit("=== 3. 逐名探针（对 AppConfig 构造，与真实读取路径同构）===")
        emit("  ⚠ 进程内改 os.environ 只影响本进程；每轮结束都会还原。")

        def probe(active: dict[str, str]) -> str | None:
            saved = {n: os.environ.get(n) for n in EXPECTED_ENV_NAMES}
            for n in EXPECTED_ENV_NAMES:
                os.environ.pop(n, None)
            os.environ.update(active)
            try:
                # 绕过 lru_cache，否则拿到上一轮的结果
                from finrag.config import AppConfig as _AC

                c = _AC()
                k = c.llm.api_key
                return k.get_secret_value() if k else None
            finally:
                for n, v in saved.items():
                    os.environ.pop(n, None)
                    if v is not None:
                        os.environ[n] = v

        for target in EXPECTED_ENV_NAMES:
            v = probe({target: "sk-probe-" + target})
            hit = v == "sk-probe-" + target
            emit(f"  只设 {target:26s} -> {_mask(v)}  {'命中' if hit else '**没命中**'}")

        v_none = probe({})
        emit(f"  两个都不设{'':18s} -> {_mask(v_none)}")
        emit(f"  ⇒{'正确（None，不伪造）' if v_none is None else '**异常：不该有值**'}")

        # 优先级：两个都设时必须取 AliasChoices 里靠前的那个
        both = probe({
            "FINRAG_LLM__API_KEY": "sk-probe-FIRST",
            "DEEPSEEKFLASH_KEY": "sk-probe-SECOND",
        })
        ok_first = both == "sk-probe-FIRST"
        emit(f"  两个都设（正式名在前）  -> {_mask(both)}"
             f"  {'取正式名，正确' if ok_first else '**优先级不符**'}")

    emit("")
    emit("=== 4. 顺带核模型名 ===")
    emit(f"  当前 {cfg.llm.model!r}")
    if cfg.llm.model == "deepseek-flash":
        emit("  ✓ DeepSeek 当前的 API 名（写错名会拿到语义不明的 400/404）")
    else:
        emit("  ⚠ 不是当前的 API 名。官方现役为 deepseek-flash。")

    text = "\n".join(out)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    emit("")
    emit(f"报告：{p}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())