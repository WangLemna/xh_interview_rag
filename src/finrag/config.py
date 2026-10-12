"""FinRAG 配置加载：环境变量 / .env > configs/default.yaml > 代码默认值。

嵌套字段用双层下划线表示，如 ``FINRAG_LLM__MODEL``、``FINRAG_PG_DSN``。
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any
# yaml 由 pydantic-settings 的 YamlConfigSettingsSource 加载，此处不直接 import yaml
from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# src/finrag/config.py -> 项目根目录
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default.yaml"
ENV_FILE: Path = PROJECT_ROOT / ".env"

#: 配置文件与 ``.env`` 一律按 UTF-8 解码。pydantic-settings 不指定编码时会退回
#: ``open()`` 的 locale 默认值，Windows 中文版终端里那是 GBK(cp936)，
#: 读到 yaml 里的中文注释即抛 UnicodeDecodeError（本项目的 config.py 本身全是中文注释）。
#: 这类故障只在 GBK 环境下出现、在 UTF-8 环境下静默正确，属于必须显式钉死的一类默认值。
FILE_ENCODING: str = "utf-8"

#: 本次构造要读的 yaml 路径。``BaseSettings`` 没有 ``_yaml_file`` 构造参数，
#: ``settings_customise_sources`` 也拿不到实例，故用 ContextVar 传递（线程/协程安全）。
_yaml_file_override: ContextVar[Path | None] = ContextVar("finrag_yaml_file", default=None)


def resolve_model_path(model: str) -> str:
    """模型标识 → 可加载路径：

    1. ``model`` 是绝对路径 → 原样返回（**不检查存在性**：写错时让加载器直说"找不到"，
       好过这个字符串被当成 HuggingFace repo id 拿去联网找）；
    2. ``PROJECT_ROOT/<model>`` 存在 → 用它（项目内自带的权重）；
    3. 都不是 → 原样返回，交由加载器按 HF repo id 处理。

    ⚠ 第 3 条有个反直觉之处：相对路径写错（如 ``models/bge-m3`` 而该目录不存在）时
    **不会**回落到 ``BAAI/bge-m3``，而是把这个相对路径当 repo id 去联网找，报的是
    "repo 找不到" —— 这与"忘了放权重"看起来毫无关系。
    """
    p = Path(model)
    if p.is_absolute():
        return str(p)
    local = PROJECT_ROOT / p
    return str(local) if local.exists() else model


class ChunkConfig(BaseModel):
    """分块策略。消融实验的主要自变量之一。"""

    max_tokens: int = Field(512, ge=64, le=4096)
    min_tokens: int = Field(80, ge=0)
    overlap_tokens: int = Field(64, ge=0)
    respect_headings: bool = True
    # 硬切时优先把切点吸附到句末标点上，而不是从字符中间劈开。
    # 表格块不走这条路（其语义单元是"行/记录"，句号不构成边界），关掉受影响的是段落部分。
    split_on_sentence: bool = True
    # 父块的 token 预算；设 0 = 完全不生成父块。
    # 节内总量 ≤ 此值 → 整节配一个父块；超过 → 按**窗口**拆成多个父块（见下），
    # 而不是"干脆不配"（年报里"一个标题 + 几百页"是常态，直接放弃会让长节命中时毫无上下文）。
    parent_max_tokens: int = Field(2048, ge=0)
    # 超长节的窗口：每个窗口最多挂多少片、相邻窗口重叠多少片。
    # 重叠是为了不让"恰好落在窗口边界"的片失去上下文；窗口同时受 ``parent_max_tokens``
    # 的 token 上限兜底（片本身 ≤ ``max_tokens``，故默认 4 片 ≈ 2048 token，正好贴合预算）。
    # 重叠数 ≥ 窗口片数时会被钳到"窗口片数 − 1"（否则步进为 0、死循环）。
    parent_window_pieces: int = Field(4, ge=2)
    parent_window_overlap: int = Field(1, ge=0)
    # 把结构化表格当作独立检索单元（默认开）：表格块独占片，被切成 ≥2 片的表
    # 另配一条只含表头行的 ``role='child'`` 片（数据片的 ``parent_index`` 指向它）。
    # 关掉即整份退回"表与同节块共片、无表头片"的形态。小表（< ``min_tokens``）不
    # 参与合并而留在碎片档：表格作为独立检索单元，不为碎片率与邻居粘成一片。
    table_units: bool = True

    #: 把 markdown 首行（表头行）复制给被硬切出来的非首片。
    #: 只补结构化表格块（``Block.kind == "table"`` 且 ``Block.table`` 非空）；
    #: 来源用 markdown 首行而非 ``TableData.header``（后者补错列名比不补更糟）。
    #: 关掉 = 只有首片带列名；会改变 ``content_fingerprint``，触发增量重入库。
    repeat_header: bool = True


class ParseConfig(BaseModel):
    """解析层的可调策略。"""

    # 正文区跨页重复块的强噪声线（块占比 / 块最长字符数）
    repeat_strong_ratio: float = Field(0.5, ge=0.0, le=1.0)
    repeat_strong_maxlen: int = Field(12, ge=1)
    # 无 TOC 文档的版式兜底及其质量门控
    fallback_enabled: bool = True
    min_body_share: float = Field(0.85, ge=0.0, le=1.0)
    # 栏重排，默认关：实测收益与误判风险不成比例
    reorder_columns: bool = False
    # 只停结构化、不停检出：`find_tables` 照跑、kind 照标，消融测的是"有行列数据"
    # 值多少分。打开时 `_to_ir_blocks` 把通过守门的表折叠成一个 ``Block``
    # （``Block.table`` 非空、``text`` 是整表 markdown）。
    structure_tables: bool = True


class EmbedConfig(BaseModel):
    """向量化配置。BGE-M3 同时产出 dense 与 sparse 两种表示。"""

    dense_model: str = "BAAI/bge-m3"
    dim: int = Field(1024, gt=0)
    device: str = "cpu"
    # GPU 上用 fp16 省一半显存、约翻倍速度；CPU 推理时该字段被忽略
    use_fp16: bool = True
    batch_size: int = Field(8, ge=1)
    max_length: int = Field(8192, ge=128)
    # 向量缓存：以 (model, text_hash) 为键，避免重复编码。
    # ⚠ 这里配的是**缓存根**，不是 embeddings 目录本身 —— `embed.bge_m3._cache_dir` 会再拼一层
    #   `CACHE_SUBDIR="embeddings"`（与 `parse/cache.py` 拼 `parse` 同构）。若这里也写成
    #   `data/cache/embeddings`，落点会变成 `data/cache/embeddings/embeddings/`（双层）。
    #   旧双层目录**故意保留**：改名迁移前不要删，它是已有向量缓存的所在地。
    cache_dir: Path = Path("data/cache")

    @property
    def cache_path(self) -> Path:
        p = self.cache_dir
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def model_path(self) -> str:
        """解析后的模型路径：绝对路径原样用；相对路径按项目根解；都不成立则当 HF repo id。"""
        return resolve_model_path(self.dense_model)


class RetrieveConfig(BaseModel):
    """检索与融合参数，M5 消融实验的自变量；``min_rerank_score`` 恒为 0.0（不拒答）——
    阈值待实测 Cross-Encoder 分数分布后再定。
    """

    dense_top_k: int = Field(100, ge=1)
    sparse_top_k: int = Field(100, ge=1)
    #: 关掉稀疏路（M5 消融项：± BGE-M3 learned sparse）。
    #: 用独立 ``bool`` 而非 ``sparse_top_k=0``：后者有 ``ge=1`` 约束，且 ``bool`` 会进
    #: :class:`RecallResult.skipped`，报告能印出"这一路被刻意关掉"。关掉时不动 ``sparse_top_k``。
    use_sparse: bool = True
    # RRF 平滑常数，论文( Cormack et al. 2009 )经验值 60，越大越弱化头部排名差异
    rrf_k: int = Field(60, ge=1)
    # 送入 Cross-Encoder 重排的候选数：既是重排射程上限，也是重排前后两栏读数的
    # 共同分母（两栏都在这批候选内部排名次，否则两栏不可比）。2026-10-11 由 40 放大到
    # 100（配合 dense/sparse_top_k 30→100，验证"召回池太小是不是召回瓶颈"）；代价是
    # CE 重排耗时线性增长（≈28.8ms/条），检索侧 P50 从 ~1255ms 涨到 ~2900ms。
    # 上限受融合并集约束（dense_top_k + sparse_top_k = 200，此处只取 100）。
    rerank_candidates: int = Field(100, ge=1)
    final_top_k: int = Field(15, ge=1)
    #: 重排分数低于该阈值时触发拒答，用于抑制幻觉。消费链路是通的，只是阈值**刻意取0.0
    #: （永不拒答）** —— 这不是待实现的空壳，而是一个已定的保守决策：
    #: `multi_entity` 拒答正确率本就偏低（25%），再加一道分数闸**很可能误杀正确答案**。
    #: **启用前提**：先拿到 Cross-Encoder 的分数分布，分开"gold 片"与"非 gold 片"，
    #: 找到那条真正能分开两者的分界线再定阈值；**不可试错式拍脑袋调**，调高必误拒。
    #: 详见 ``FinRAG-优化清单.md`` Q9（拒答口径）。刻意留在 yaml 里，便于将来一行启用。
    min_rerank_score: float = 0.0

    # --- 多实体（对比类问句）---
    # 命中 2 家以上公司时，按公司分组各召回一次（否则片段多的一家会把另一方整个挤出
    # 候选池），重排后按组保底选取，保证对比问句每一方都有片段进 prompt。
    #
    #: 每组是否**各取满** ``rerank_candidates``（而不是把 100 均分给各组）。
    #: 开：两家 → 200 条进重排（约 2× 耗时，≈5.8s）、三家 → 300、依此类推。多家对比本来就更慢，
    #: 故默认开；想省时间就设 false，各组共享一个总池。
    group_pool_full: bool = True
    #: 用"满额池"的组数上限。超过此值改回按组数均分，但**仍然分组召回**（不会退回并集）。
    #: 纯粹是防止一次提问把重排池撑得过大。
    full_pool_max_groups: int = Field(4, ge=2)
    #: 最终注入 prompt 的条数硬上限。组数一多，多实体放大后线性增长（按
    #: ``final_top_k × 组数``），既费 token 也让"每条都配得上引用"名不副实；
    #: 压到 20 与 ``AskRequest.top_k`` 的 ``le=20`` 对齐。
    #: ⚠ 与 ``api/server.py`` 的 ``le=20`` 是**两处独立的 20**（这里可改、那里写死在
    #: pydantic 校验里），改本项不会带动那边 ⇒ 会出现"服务端内部能取 30、但客户端传
    #: 25 被拒"的情况。
    top_k_cap: int = Field(20, ge=1, le=50)

    # HNSW 图遍历的候选池大小，必须 ≥ dense_top_k，否则近似召回率会在候选池
    # 边缘显著下滑且不报错。设 0 = 沿用 pgvector 默认（仅调试用）。
    # 200 是**已定值、不要再当调优项**：pgvector 默认 40 偏小（见 store/schema.sql），
    # 现为 dense_top_k(100) 的 2.0x。而实测延迟大头是 **CE 重排（占检索侧 91%）**，
    # 稠密 28ms + 稀疏 2ms —— HNSW 遍历根本不是瓶颈，降它只会掉召回、省不下时间。
    ef_search: int = Field(200, ge=0)


class RerankConfig(BaseModel):
    """Cross-Encoder 精排参数；``device`` 需经 FlagEmbedding 的 ``devices=``（复数）传入，且只在构造时生效。"""

    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    use_fp16: bool = True
    batch_size: int = Field(8, ge=1)

    # 配对后（query + passage）的总长上限，不是片长上限；超长时从片尾静默截断。
    # 片最长 512 token，抬到 1024 留一倍余量；改它会让分数分布整体漂移。
    max_length: int = Field(1024, ge=1)

    @property
    def model_path(self) -> str:
        """解析后的模型路径，规则同 ``EmbedConfig.model_path``。"""
        return resolve_model_path(self.model)


class FuzzyConfig(BaseModel):
    """别名模糊匹配（Q2）：精确子串之外的容错，治"错别字 → 静默退回全库"。

    **顺序铁律**：先用拼音索引把候选集缩小，**再**对缩小后的候选算编辑距离。
    反过来对全表 879 个别名逐个算距离才是真慢（与已作废的 AC 自动机犯同一个错）。

    **裁决权始终在别名表**：模糊只负责"提名"，采不采纳由 ``resolve`` 按下列阈值决定，
    LLM 只有提议权。匹配不到时仍走原路径（进 ``missing`` → 拦截），没有"兜底全库"。
    """

    #: 总开关。关掉 = 纯精确子串（Q2 之前的行为）。留开关是为了出问题时能一键回退。
    enabled: bool = True

    # --- 拼音索引（缩小候选）---
    #: 拼音索引最多带出多少个候选别名去做距离比对。再多只会让第2 步变慢，且低质量候选
    #: 本来就会被阈值挡掉。50 对应"首字母撞车的公司数"量级，足够覆盖同音错别字。
    pinyin_candidates: int = Field(50, ge=1, le=500)
    #: 候选字符串短于该长度时**不做**距离比对。1~2 个字的别名（"茅台""平安"）太容易撞车，
    #: 距离阈值对它们几乎没有区分度，宁可漏过也不误匹配 —— 误匹配的代价（答成别家公司）
    #: 远大于漏匹配（退回原有拦截路径）。
    min_len: int = Field(3, ge=2, le=10)

    # --- 编辑距离（比对缩小后的候选）---
    #: 拼音串上允许的最大编辑距离。同音/近音错别字（"格力电器"→"格力申器"）距离 1；
    #: 给到 2 会开始误伤真正不同的公司，故保守取 1。
    max_pinyin_dist: int = Field(1, ge=0, le=3)
    #: 汉字串上允许的最大编辑距离。汉字层比拼音层**更严**（默认 0=不启用）：
    #: 拼音相同但汉字差很多（如"格力"/"格林"都是 geli 段），说明是不同公司，不是错别字。
    max_char_dist: int = Field(0, ge=0, le=3)
    #: 多个候选距离相同时的处置：``True``=取拼音最短的那个（更可能是简称，更常被问到），
    #: ``False``=全部丢弃（宁可漏匹配也不猜）。默认丢弃。
    tie_break: bool = Field(False)
    #: 一个候选被至少这么多路证据支持才算命中（拼音精确 / 拼音近 / 首字母）。
    #: 调高更保守。当前逻辑：拼音全串相同才算强证据，否则要求唯一最优候选。
    require_strong: bool = Field(True)


class LLMConfig(BaseModel):
    """生成模型配置。DeepSeek 无 embedding 接口，仅承担生成职责。"""

    # 当前 API 名为 ``deepseek-flash``（V4.1-Flash，1M 上下文）。模型下线不会体现在
    # 本地报错里，请求只拿到语义不明的 400/404，排查时先核这一项。
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    # Key 由 ``AppConfig.llm_api_key_env`` 统一读取后灌入，此处不接环境变量别名。
    api_key: SecretStr | None = None
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1)
    timeout: int = Field(60, ge=1)
    stream: bool = True


class MultiTurnConfig(BaseModel):
    """多轮记忆配置：两层机制，各自可关，互不替代。

    1) **LLM 问句改写**（``enabled``）：把"它 / 省略主语"的问句补全成自足问句
       （``gen/rewrite.py``）。**话题判定并入同一次调用** —— 模型判为不依赖上文时
       原样输出，不另起第二个模型。
    2) **确定性范围继承**（``inherit_scope``）：本轮没认出公司时，沿用上一轮解析出的
       公司/年份重新定位文档（``store/resolve.py``）。是改写关闭或失败时的兜底。

    注意：**上一轮答案正文永不进入 prompt** —— 记忆只传"结构化实体 + 用户问句"，
    见 ``gen/rewrite.py`` 与 ``gen/prompt.py`` 的反幻觉不变式。
    """

    #: LLM 问句改写总开关。关掉 = 多轮只做确定性的范围继承，不调模型。
    enabled: bool = True
    #: 确定性范围继承开关。关掉 = 本轮认不出公司就退回全库（改写仍可独立生效）。
    inherit_scope: bool = True
    #: 服务端**保留**的历史轮数上限（也是 ``AskRequest.history`` 的接受上限）。
    #: 保留本身几乎不花钱（只是一个列表），但要注意：目前**只有两处消费历史** ——
    #: 范围继承（只看最近一轮）与问句改写（看 ``rewrite_turns`` 轮）。调大它本身不改变
    #: 检索行为，真正的成本在 ``rewrite_turns``。
    max_turns: int = Field(100, ge=0, le=100)
    #: **送进改写 prompt 的候选窗口**（取自保留历史的最近若干轮）。注意它不等于"依据窗口"：
    #: 模型在这个窗口内**自己挑**用哪几轮来补全指代，并把挑了哪几轮回报到 ``rewrite.used_turns``。
    #: 窗口大小的取舍：太小则"跳了一轮话题又跳回来"时用不到那轮（近 ≠ 相关）；太大则每次提问
    #: 多付 token（约 30~40/轮）且模型要扫更多噪声。5~20 之间的成本差可忽略，**精确取值需要
    #: 评测数据来定**；没有评测时取 10（覆盖常见的话题往复跨度）。
    #: 设 0 = 不调改写模型（退化为纯确定性范围继承，同时**失去库外公司识别** —— 首轮问句里的
    #: 库外公司名将不再进 `missing`，会静默退回全库检索）。
    rewrite_turns: int = Field(10, ge=0, le=100)
    #: 改写产物的字符上限；超长视为失败、回落原问句（防模型把解释写进来）。
    max_chars: int = Field(200, ge=1)


class GenConfig(BaseModel):
    """生成侧的组装策略。上下文 token 预算不在（恒不触发：6 片 × 512 token 远小于 1M 上下文），
    拒答话术也不在（它是契约，只在 ``finrag/gen/prompt.py`` 定义一次）。
    """

    #: 引用标签是否带 ``heading_path``（章节路径），影响引用可读性与章节归属判断。
    #: 它不改变编码文本，因而不影响召回率。
    include_heading_path: bool = True

    #: 是否对答案做数字溯源审计（``gen.prompt.audit_numbers``）；只影响观测，不参与拒答、不改 prompt。
    audit_numbers: bool = True


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)


class PoolConfig(BaseModel):
    """数据库连接池（``store/pool.py``）。**只影响"连接怎么复用"，不改任何 SQL 语义。**

    FastAPI 的同步路由跑在线程池里，而 psycopg 的 ``Connection`` **非线程安全**
    ⇒ 必须"一借一还"、不跨请求/线程共享。池大小按并发量调，不是检索层的消融自变量，
    故不进 ``configs/default.yaml`` 的检索参数区；调优用环境变量覆盖
    （如 ``FINRAG_POOL__MAX_SIZE=16``）。
    """

    #: 预热连接数（进程启动时即建好这么多，避免首个请求等建连）。
    min_size: int = Field(1, ge=0)
    #: 连接上限。超过后借连接会等待 ``timeout`` 秒。
    max_size: int = Field(8, ge=1)
    #: 借不到连接时最多等多久（秒），超时抛异常 → 由路由转成 503 / SSE error 事件。
    timeout: float = Field(30.0, gt=0.0)
    #: 空闲连接最多存活多久（秒）后被回收重建；0 = 不限。仅 ``psycopg_pool`` 路径生效。
    max_idle: float = Field(300.0, ge=0.0)


class PathConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    log_dir: Path = Path("logs")
    # 中间产物缓存的根目录，不参与任何判据：删掉只会让下一次跑得慢。被 .gitignore 排除。
    cache_dir: Path = Path("data/cache")
    # 语料清单（CSV）：身份(doc_id)与元数据(code/company/year)的唯一来源。
    # **必填** —— 入库时找不到即 fail-closed，不退回"按文件名猜"。可用 FINRAG_PATHS__MANIFEST 覆盖。
    manifest: Path = Path("data/corpus_manifest.csv")

    def resolve(self, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value


class AppConfig(BaseSettings):
    """完整配置树。env / .env 的优先级高于 yaml。"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding=FILE_ENCODING,
        env_prefix="FINRAG_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """注册 yaml source 并排在 env / .env 之后，顺序即优先级：
        ``init > env > dotenv > yaml > secrets > 默认值``。

        路径与编码都显式给全：``load_config(path)`` 借 ``_yaml_file_override`` 换文件，
        编码不给则退回 locale 默认值（见 :data:`FILE_ENCODING`）。
        """
        yaml_source = YamlConfigSettingsSource(
            settings_cls,
            yaml_file=_yaml_file_override.get() or DEFAULT_CONFIG_PATH,
            yaml_file_encoding=FILE_ENCODING,
        )
        return (init_settings, env_settings, dotenv_settings, yaml_source, file_secret_settings)

    # DSN 含密码，用 SecretStr 避免日志泄露
    pg_dsn: SecretStr = SecretStr(
        "postgresql://postgres:postgres@localhost:5432/finrag"
    )

    parse: ParseConfig = Field(default_factory=ParseConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    fuzzy: FuzzyConfig = Field(default_factory=FuzzyConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    multiturn: MultiTurnConfig = Field(default_factory=MultiTurnConfig)
    gen: GenConfig = Field(default_factory=GenConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    pool: PoolConfig = Field(default_factory=PoolConfig)
    paths: PathConfig = Field(default_factory=PathConfig)

    # ------------------------------------------------------------------
    # LLM Key 的读取口（唯一入口）
    # ------------------------------------------------------------------
    # 必须在 ``AppConfig`` 上：子模型 ``LLMConfig`` 是普通 ``BaseModel``，不读环境变量。
    # 两个别名都显式列出（``validation_alias`` 会顶替派生的 env 名），
    # ``.env`` 里的 ``DEEPSEEK_API_KEY`` 占位符是有意排除的历史名。
    llm_api_key_env: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("FINRAG_LLM__API_KEY", "DEEPSEEKFLASH_KEY"),
    )

    def model_post_init(self, __context: Any) -> None:
        """把顶层读到的 Key 灌进 ``llm``，仅在 ``llm.api_key`` 为空时写入（不覆盖显式值）。"""
        if self.llm_api_key_env and not self.llm.api_key:
            self.llm.api_key = self.llm_api_key_env


def load_config(path: Path | str | None = None) -> AppConfig:
    """加载配置；yaml 由已注册的 ``YamlConfigSettingsSource`` 加载，这里不直接读盘。
    ``path`` 用于临时指向另一份 yaml（消融临时基线 / 对照实验）。

    原实现传的是 ``AppConfig(_yaml_file=...)``，但 ``BaseSettings`` 没有这个构造参数
    —— 它会被当成普通 init 字段吞进 ``extra="ignore"``，读到的仍是默认那份 yaml，
    且不报错（消融实验会静默跑错基线）。现改由 ContextVar 传给 yaml source。
    """
    if path is None:
        return AppConfig()
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"配置 yaml 不存在：{p}")
    token = _yaml_file_override.set(p)
    try:
        return AppConfig()
    finally:
        _yaml_file_override.reset(token)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """进程内单例，避免重复读盘。"""
    return load_config()
